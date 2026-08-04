"""Qt presentation and lifecycle integration for plugin environment tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import WeakSet, ref

from superqt import ensure_main_thread

from napari.plugins.environments import (
    PluginEnvironmentOperation,
    PluginTaskPhase,
    PluginTaskState,
    _add_task_observer,
    _set_task_dispatcher,
    list_active_plugin_environment_tasks,
)
from napari.utils.notifications import notification_manager
from napari.utils.progress import cancelable_progress

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QApplication

    from napari.plugins.environments import PluginTask, PluginTaskProgress

_installed = False
_LIFECYCLE_PHASES = {
    PluginTaskPhase.PREPARING,
    PluginTaskPhase.PROVISIONING,
    PluginTaskPhase.STARTING,
    PluginTaskPhase.CLEANING_UP,
}


class _PluginTaskActivityProgress(cancelable_progress):
    """Activity progress whose cancellation reaches related plugin tasks."""

    def __init__(self, group: _EnvironmentActivityGroup, label: str) -> None:
        self._group_reference = ref(group)
        self._label = label
        self._activity_closed = False
        super().__init__(total=0, desc='Managed plugin environment')

    def apply(
        self,
        update: PluginTaskProgress,
        description: str,
    ) -> None:
        """Apply one absolute plugin task progress snapshot."""

        self.set_description(f'{self._label}: {description}')
        self.total = update.total or 0
        self.n = update.current or 0
        self.events.value(value=self.n)

    def cancel(self) -> None:
        """Request backend cancellation without waiting for iteration."""

        if self.is_canceled:
            return
        self.is_canceled = True
        if group := self._group_reference():
            group.cancel()

    def close(self) -> None:
        """Remove the Activity item exactly once."""

        if self._activity_closed:
            return
        self._activity_closed = True
        super().close()


def _dispatch_to_main_thread(callback: Callable[[], None]) -> None:
    ensure_main_thread(callback)()


def _activity_label(task: PluginTask[Any]) -> str:
    metadata = task.metadata
    if metadata is None:
        return 'Managed plugin environment'
    plugin = metadata.plugin
    environment = metadata.environment_id
    if environment is not None and plugin is not None:
        environment_prefix = f'{plugin}.'
        if environment.startswith(environment_prefix):
            environment = environment.removeprefix(environment_prefix)
        return f'{plugin} · {environment}'
    return plugin or environment or 'Managed plugin environment'


def _lifecycle_description(
    operation: PluginEnvironmentOperation,
    phase: PluginTaskPhase,
    message: str,
) -> str:
    if phase is PluginTaskPhase.PREPARING:
        if operation is PluginEnvironmentOperation.EXECUTE:
            return 'Checking environment'
        return 'Preparing environment'
    if phase is PluginTaskPhase.PROVISIONING:
        return 'Installing environment'
    if phase is PluginTaskPhase.STARTING:
        return 'Starting worker'
    if operation is PluginEnvironmentOperation.STOP:
        return 'Stopping workers'
    if operation is PluginEnvironmentOperation.REMOVE:
        return 'Removing environment'
    if message.startswith('Removing previous environment'):
        return 'Cleaning up previous environment'
    return 'Finalizing environment'


_PHASE_PRIORITY = {
    PluginTaskPhase.PREPARING: 1,
    PluginTaskPhase.PROVISIONING: 2,
    PluginTaskPhase.CLEANING_UP: 3,
    PluginTaskPhase.STARTING: 4,
}
_activity_groups: dict[
    tuple[str | None, tuple[str, ...]], _EnvironmentActivityGroup
] = {}
_presented_tasks: WeakSet[PluginTask[Any]] = WeakSet()


class _EnvironmentActivityGroup:
    """Present concurrent lifecycle tasks for one environment as one item."""

    def __init__(
        self,
        key: tuple[str | None, tuple[str, ...]],
        label: str,
    ) -> None:
        self.key = key
        self.label = label
        self.tasks: dict[str, PluginTask[Any]] = {}
        self.latest: dict[
            str, tuple[int, PluginTask[Any], PluginTaskProgress]
        ] = {}
        self.sequence = 0
        self.progress: _PluginTaskActivityProgress | None = None

    def attach(self, task: PluginTask[Any]) -> None:
        """Attach a task and observe its lifecycle until execution or done."""

        self.tasks[task.task_id] = task

        def receive_progress(update: PluginTaskProgress) -> None:
            if (
                task.metadata is not None
                and task.metadata.operation
                is PluginEnvironmentOperation.EXECUTE
                and update.phase is PluginTaskPhase.EXECUTING
            ):
                self.detach(task)
                return
            if update.phase not in _LIFECYCLE_PHASES:
                return
            self.sequence += 1
            self.latest[task.task_id] = (self.sequence, task, update)
            self.render()

        task.add_progress_callback(receive_progress)
        task.add_done_callback(self.detach)

    def cancel(self) -> None:
        """Request cancellation for every task represented by this item."""

        for task in tuple(self.tasks.values()):
            task.cancel()

    def detach(self, task: PluginTask[Any]) -> None:
        """Stop representing a task and close an empty group."""

        if self.tasks.pop(task.task_id, None) is None:
            return
        self.latest.pop(task.task_id, None)
        if self.tasks:
            self.render()
            return
        if self.progress is not None:
            self.progress.close()
            self.progress = None
        if _activity_groups.get(self.key) is self:
            del _activity_groups[self.key]

    def render(self) -> None:
        """Render the most advanced active lifecycle phase."""

        if not self.latest:
            if self.progress is not None:
                self.progress.close()
                self.progress = None
            return
        _, task, update = max(
            self.latest.values(),
            key=lambda item: (_PHASE_PRIORITY[item[2].phase], item[0]),
        )
        metadata = task.metadata
        if metadata is None:
            return
        if self.progress is None:
            self.progress = _PluginTaskActivityProgress(self, self.label)
        self.progress.apply(
            update,
            _lifecycle_description(
                metadata.operation,
                update.phase,
                update.message,
            ),
        )


@ensure_main_thread
def _notify_task_failure(task: PluginTask[Any]) -> None:
    """Report unhandled task failures without choosing a progress surface."""

    def receive_done(done_task: PluginTask[Any]) -> None:
        if (
            done_task.state is PluginTaskState.FAILED
            and done_task.error is not None
        ):
            error = done_task.error
            notification_manager.receive_error(
                type(error),
                error,
                error.__traceback__,
            )

    task.add_done_callback(receive_done)


@ensure_main_thread
def _present_task_activity(task: PluginTask[Any]) -> None:
    """Present only environment lifecycle phases in existing Activity UI."""

    metadata = task.metadata
    if metadata is None or task.done or task in _presented_tasks:
        return
    _presented_tasks.add(task)
    key = (metadata.plugin, metadata.environment_ids)
    group = _activity_groups.get(key)
    if group is None:
        group = _EnvironmentActivityGroup(key, _activity_label(task))
        _activity_groups[key] = group
    group.attach(task)


def _observe_plugin_task(task: PluginTask[Any]) -> None:
    _notify_task_failure(task)
    _present_task_activity(task)


def _shutdown_with_notification() -> None:
    from napari.plugins._environment_manager import (
        shutdown_plugin_environments,
    )

    try:
        shutdown_plugin_environments()
    except Exception as error:  # noqa: BLE001
        notification_manager.receive_error(
            type(error),
            error,
            error.__traceback__,
        )


def install_plugin_environment_qt_support(app: QApplication) -> None:
    """Install Qt dispatch, failure notification, and shutdown support once."""

    global _installed
    if _installed:
        return
    _installed = True
    _set_task_dispatcher(_dispatch_to_main_thread)
    _add_task_observer(_observe_plugin_task)
    for task in list_active_plugin_environment_tasks():
        _observe_plugin_task(task)
    app.aboutToQuit.connect(_shutdown_with_notification)


__all__ = ('install_plugin_environment_qt_support',)

"""Qt presentation and lifecycle integration for plugin environment tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import ref

from superqt import ensure_main_thread

from napari.plugins.environments import (
    PluginEnvironmentOperation,
    PluginTaskPhase,
    PluginTaskState,
    _add_task_observer,
    _set_task_dispatcher,
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
    """Activity progress whose cancellation immediately reaches its task."""

    def __init__(self, task: PluginTask[Any], label: str) -> None:
        self._task_reference = ref(task)
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
        if task := self._task_reference():
            task.cancel()

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
) -> str:
    if phase is PluginTaskPhase.PREPARING:
        return 'Preparing environment'
    if phase is PluginTaskPhase.PROVISIONING:
        return 'Installing environment'
    if phase is PluginTaskPhase.STARTING:
        return 'Starting worker'
    if operation is PluginEnvironmentOperation.STOP:
        return 'Stopping workers'
    if operation is PluginEnvironmentOperation.REMOVE:
        return 'Removing environment'
    return 'Cleaning up previous environment'


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
    if metadata is None or task.done:
        return
    activity_progress: _PluginTaskActivityProgress | None = None
    active = True

    def close_activity(_task: PluginTask[Any] | None = None) -> None:
        nonlocal active
        active = False
        if activity_progress is not None:
            activity_progress.close()

    def receive_progress(update: PluginTaskProgress) -> None:
        nonlocal activity_progress
        if not active:
            return
        if (
            metadata.operation is PluginEnvironmentOperation.EXECUTE
            and update.phase is PluginTaskPhase.EXECUTING
        ):
            close_activity()
            return
        if update.phase not in _LIFECYCLE_PHASES:
            return
        if activity_progress is None:
            activity_progress = _PluginTaskActivityProgress(
                task, _activity_label(task)
            )
        activity_progress.apply(
            update,
            _lifecycle_description(metadata.operation, update.phase),
        )

    task.add_progress_callback(receive_progress)
    task.add_done_callback(close_activity)


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
    app.aboutToQuit.connect(_shutdown_with_notification)


__all__ = ('install_plugin_environment_qt_support',)

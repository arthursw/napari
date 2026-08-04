from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari._qt import _plugin_environments as qt_environments
from napari.plugins.environments import (
    PluginEnvironmentOperation,
    PluginEnvironmentProvisioningError,
    PluginTask,
    PluginTaskCanceledError,
    PluginTaskMetadata,
    PluginTaskPhase,
)
from napari.utils.progress import progress


def _managed_task(
    operation: PluginEnvironmentOperation = PluginEnvironmentOperation.PREPARE,
) -> PluginTask[None]:
    return PluginTask(
        PluginTaskMetadata(
            operation,
            'example-plugin',
            ('example-plugin.worker',),
            (
                'example-plugin.compute'
                if operation is PluginEnvironmentOperation.EXECUTE
                else None
            ),
        )
    )


def _activity_progresses():
    return tuple(
        item
        for item in progress._all_instances
        if isinstance(item, qt_environments._PluginTaskActivityProgress)
    )


def test_task_observer_only_presents_structured_failure(
    qapp, monkeypatch
) -> None:
    received_errors: list[BaseException] = []
    monkeypatch.setattr(
        qt_environments.notification_manager,
        'receive_error',
        lambda exc_type, error, traceback: received_errors.append(error),
    )
    task: PluginTask[None] = PluginTask()

    qt_environments._notify_task_failure(task)
    error = PluginEnvironmentProvisioningError(
        'Environment installation failed',
        details='pixi exited with status 1',
    )
    task._set_error(error)

    assert received_errors == [error]


def test_lifecycle_progress_uses_existing_activity_ui(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    activity_dialog = viewer.window._qt_window._activity_dialog
    task = _managed_task()

    qt_environments._present_task_activity(task)
    task._set_running(PluginTaskPhase.PREPARING, 'Preparing environment')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()
    pbar = activity_dialog.get_pbar_from_prog(activity_progress)
    assert pbar is not None
    assert (
        pbar.description_label.text()
        == 'example-plugin · worker: Preparing environment: '
    )

    task._report_progress(
        PluginTaskPhase.PROVISIONING,
        'Installing packages',
        2,
        5,
    )
    qtbot.waitUntil(
        lambda: (
            pbar.description_label.text()
            == 'example-plugin · worker: Installing environment: '
        )
    )
    assert pbar.qt_progress_bar.value() == 2
    assert pbar.qt_progress_bar.maximum() == 5

    task._report_progress(
        PluginTaskPhase.PROVISIONING,
        'Resolving packages without a known total',
    )
    qtbot.waitUntil(lambda: pbar.qt_progress_bar.maximum() == 0)
    assert pbar.qt_progress_bar.minimum() == 0

    task._report_progress(
        PluginTaskPhase.PROVISIONING,
        'Installing more packages',
        3,
        7,
    )
    qtbot.waitUntil(lambda: pbar.qt_progress_bar.maximum() == 7)
    assert pbar.qt_progress_bar.value() == 3

    task._set_result(None)
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)


def test_long_backend_message_does_not_hide_activity_controls(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    activity_dialog = viewer.window._qt_window._activity_dialog
    task: PluginTask[None] = PluginTask(
        PluginTaskMetadata(
            PluginEnvironmentOperation.REMOVE,
            'napari-wsegmenter',
            ('napari-wsegmenter.stardist',),
        )
    )
    physical_name = (
        'napari-wsegmenter-napari-wsegmenter.stardist-'
        '94ad049c6caf-57f2617bd1945f85'
    )
    qt_environments._present_task_activity(task)

    task._set_running(
        PluginTaskPhase.CLEANING_UP,
        f"environment_removal: Removing managed environment '{physical_name}'",
    )
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()
    pbar = activity_dialog.get_pbar_from_prog(activity_progress)
    assert pbar is not None

    assert pbar.description_label.text() == (
        'napari-wsegmenter · stardist: Removing environment: '
    )
    assert physical_name not in pbar.description_label.text()
    assert pbar.layout().itemAt(0).widget() is pbar.description_label
    controls = pbar.layout().itemAt(1).layout()
    assert controls.itemAt(0).widget() is pbar.qt_progress_bar
    assert controls.itemAt(2).widget() is pbar.cancel_button
    assert not pbar.cancel_button.isHidden()

    task._set_result(None)
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)


def test_plugin_wide_activity_description_has_plugin_identity(qtbot) -> None:
    task: PluginTask[None] = PluginTask(
        PluginTaskMetadata(
            PluginEnvironmentOperation.STOP,
            'example-plugin',
        )
    )
    qt_environments._present_task_activity(task)
    task._set_running(PluginTaskPhase.CLEANING_UP, 'Stopping workers')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    assert activity_progress.desc == 'example-plugin: Stopping workers: '

    task._set_result(None)
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)


def test_finalization_is_not_presented_as_previous_environment_cleanup(
    qtbot,
) -> None:
    task = _managed_task(PluginEnvironmentOperation.EXECUTE)
    qt_environments._present_task_activity(task)

    task._set_running(
        PluginTaskPhase.CLEANING_UP,
        'Finalizing managed plugin environment',
    )
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    assert activity_progress.desc.endswith('Finalizing environment: ')

    task._set_result(None)
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)


def test_concurrent_tasks_for_environment_share_one_activity_item(
    qtbot,
) -> None:
    prepare = _managed_task(PluginEnvironmentOperation.PREPARE)
    execute = _managed_task(PluginEnvironmentOperation.EXECUTE)
    qt_environments._present_task_activity(prepare)
    qt_environments._present_task_activity(execute)

    prepare._set_running(
        PluginTaskPhase.PROVISIONING,
        'Installing packages',
    )
    execute._set_running(
        PluginTaskPhase.PREPARING,
        'Waiting for environment',
    )
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()
    assert activity_progress.desc.endswith('Installing environment: ')

    prepare._set_result(None)
    qtbot.waitUntil(
        lambda: activity_progress.desc.endswith('Checking environment: ')
    )

    execute._report_progress(
        PluginTaskPhase.EXECUTING,
        'Running worker',
    )
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)
    execute._set_result(None)


def test_shared_activity_cancel_requests_all_tasks(qtbot) -> None:
    first = _managed_task(PluginEnvironmentOperation.PREPARE)
    second = _managed_task(PluginEnvironmentOperation.EXECUTE)
    qt_environments._present_task_activity(first)
    qt_environments._present_task_activity(second)
    first._set_running(PluginTaskPhase.PROVISIONING, 'Installing')
    second._set_running(PluginTaskPhase.PREPARING, 'Waiting')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    activity_progress.cancel()

    assert first.cancellation_requested
    assert second.cancellation_requested
    first._set_canceled()
    second._set_canceled()


def test_presenting_same_task_twice_is_idempotent(qtbot) -> None:
    task = _managed_task()
    qt_environments._present_task_activity(task)
    qt_environments._present_task_activity(task)

    task._set_running(PluginTaskPhase.PREPARING, 'Preparing')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    task._set_result(None)
    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)


def test_activity_cancel_immediately_requests_task_cancellation(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    activity_dialog = viewer.window._qt_window._activity_dialog
    task = _managed_task()
    qt_environments._present_task_activity(task)
    task._set_running(PluginTaskPhase.PROVISIONING, 'Installing packages')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()
    pbar = activity_dialog.get_pbar_from_prog(activity_progress)
    assert pbar is not None

    qtbot.mouseClick(pbar.cancel_button, Qt.MouseButton.LeftButton)

    assert task.cancellation_requested
    task._set_canceled()
    with pytest.raises(PluginTaskCanceledError):
        task.result()


def test_execution_phase_is_excluded_from_activity(qtbot) -> None:
    task = _managed_task(PluginEnvironmentOperation.EXECUTE)
    qt_environments._present_task_activity(task)
    task._set_running(PluginTaskPhase.PREPARING, 'Checking environment')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    task._report_progress(PluginTaskPhase.EXECUTING, 'Running model', 1, 2)

    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)
    task._set_result(None)


def test_terminal_failure_closes_activity_item(qtbot) -> None:
    task = _managed_task()
    qt_environments._present_task_activity(task)
    task._set_running(PluginTaskPhase.CLEANING_UP, 'Cleaning environment')
    qtbot.waitUntil(lambda: len(_activity_progresses()) == 1)
    [activity_progress] = _activity_progresses()

    error = PluginEnvironmentProvisioningError('Cleanup failed')
    task._set_error(error)

    qtbot.waitUntil(lambda: activity_progress not in progress._all_instances)
    with pytest.raises(PluginEnvironmentProvisioningError):
        task.result()


def test_qt_support_installation_is_idempotent(monkeypatch) -> None:
    dispatchers = []
    observers = []
    shutdown_callbacks = []
    app = SimpleNamespace(
        aboutToQuit=SimpleNamespace(
            connect=shutdown_callbacks.append,
        )
    )
    monkeypatch.setattr(qt_environments, '_installed', False)
    monkeypatch.setattr(
        qt_environments,
        '_set_task_dispatcher',
        dispatchers.append,
    )
    monkeypatch.setattr(
        qt_environments,
        '_add_task_observer',
        observers.append,
    )

    qt_environments.install_plugin_environment_qt_support(app)
    qt_environments.install_plugin_environment_qt_support(app)

    assert dispatchers == [qt_environments._dispatch_to_main_thread]
    assert observers == [qt_environments._observe_plugin_task]
    assert shutdown_callbacks == [qt_environments._shutdown_with_notification]

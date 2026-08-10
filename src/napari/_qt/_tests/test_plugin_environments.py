from __future__ import annotations

import threading
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from napari._qt import _plugin_environments as qt_environments
from napari._qt._qapp_model.qactions._plugins import Q_PLUGINS_ACTIONS
from napari.plugins import _environment_manager as manager_module
from napari.plugins._environment_manager import PluginEnvironmentManager
from napari.plugins._tests.test_environments import _FakeBackend, _snapshot
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentUnavailableError,
    PluginTask,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def environment_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr(
        manager_module, '_build_snapshot', lambda backend: _snapshot()
    )
    manager = PluginEnvironmentManager(
        tmp_path, backend_factory=lambda root: backend
    )
    monkeypatch.setattr(
        qt_environments, 'get_plugin_environment_manager', lambda: manager
    )
    yield manager, backend
    with suppress(Exception):
        manager.close()


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
    error = PluginEnvironmentUnavailableError('Environment unavailable')

    task._set_error(error)

    assert received_errors == [error]


def test_qt_support_installation_is_idempotent(monkeypatch) -> None:
    dispatchers = []
    observers = []
    shutdown_callbacks = []
    app = SimpleNamespace(
        aboutToQuit=SimpleNamespace(connect=shutdown_callbacks.append)
    )
    monkeypatch.setattr(qt_environments, '_installed', False)
    monkeypatch.setattr(
        qt_environments, '_set_task_dispatcher', dispatchers.append
    )
    monkeypatch.setattr(
        qt_environments, '_add_task_observer', observers.append
    )

    qt_environments.install_plugin_environment_qt_support(app)
    qt_environments.install_plugin_environment_qt_support(app)

    assert dispatchers == [qt_environments._dispatch_to_main_thread]
    assert observers == [qt_environments._notify_task_failure]
    assert shutdown_callbacks == [qt_environments._shutdown_with_notification]


def test_reuse_only_startup_stays_silent(qtbot, environment_manager) -> None:
    manager, backend = environment_manager
    backend.names.add('example-plugin.worker')
    backend.mutate = False
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)

    manager.start_reconciliation()
    qtbot.waitUntil(lambda: not manager.setup_running())

    assert not dialog.isVisible()


def test_setup_attention_does_not_request_focus(
    qtbot, environment_manager, monkeypatch
) -> None:
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)
    focus_requests: list[str] = []
    monkeypatch.setattr(
        dialog, 'raise_', lambda: focus_requests.append('raise')
    )
    monkeypatch.setattr(
        dialog, 'activateWindow', lambda: focus_requests.append('activate')
    )

    dialog._show_for_attention()
    dialog._show_for_attention()

    assert dialog.isVisible()
    assert focus_requests == []


def test_mutating_startup_shows_one_modal_dialog(
    qtbot, environment_manager
) -> None:
    manager, _backend = environment_manager
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)

    manager.start_reconciliation()
    qtbot.waitUntil(lambda: not manager.setup_running())
    qtbot.waitUntil(lambda: not dialog.isVisible())

    assert dialog.windowModality().value == 2
    assert 'Environment ready' in dialog._logs.toPlainText()


def test_failed_setup_offers_retry_and_continue(
    qtbot, environment_manager
) -> None:
    manager, backend = environment_manager
    backend.fail_provision = True
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)

    manager.start_reconciliation()
    qtbot.waitUntil(lambda: not manager.setup_running())

    assert dialog.isVisible()
    assert dialog._retry.isVisible()
    assert dialog._continue.isVisible()
    backend.fail_provision = False
    dialog._retry.click()
    qtbot.waitUntil(lambda: not manager.setup_running())
    assert not manager.setup_has_failures()


def test_canceled_setup_offers_retry_and_continue(
    qtbot, environment_manager, monkeypatch
) -> None:
    manager, _backend = environment_manager
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()

    def build_snapshot(backend):
        snapshot_started.set()
        assert release_snapshot.wait(2)
        return _snapshot()

    monkeypatch.setattr(manager_module, '_build_snapshot', build_snapshot)
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)

    manager.start_reconciliation()
    assert snapshot_started.wait(1)
    assert manager.cancel_setup()
    release_snapshot.set()
    qtbot.waitUntil(lambda: not manager.setup_running())
    qtbot.waitUntil(lambda: dialog.isVisible())

    assert dialog._retry.isVisible()
    assert dialog._continue.isVisible()


def test_cancel_setup_button_waits_for_cleanup_and_continues(
    qtbot, environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()

    def build_snapshot(backend):
        snapshot_started.set()
        assert release_snapshot.wait(2)
        return _snapshot()

    monkeypatch.setattr(manager_module, '_build_snapshot', build_snapshot)
    dialog = qt_environments._PluginSetupDialog()
    qtbot.addWidget(dialog)

    manager.start_reconciliation()
    assert snapshot_started.wait(1)
    dialog.show()
    dialog._refresh()

    assert dialog._cancel.isVisible()
    dialog._cancel.click()
    assert not dialog._cancel.isEnabled()
    assert dialog._label.text().startswith('Canceling setup')

    release_snapshot.set()
    qtbot.waitUntil(lambda: not manager.setup_running())
    qtbot.waitUntil(lambda: not dialog.isVisible())

    assert not manager.setup_has_failures()
    assert backend.provisioned == []
    assert all(
        view.setup_state is manager_module._SetupState.SKIPPED
        for view in manager.environment_views()
    )


def test_automatic_shutdown_is_coalesced_and_explicit_close_retries(
    environment_manager, monkeypatch
) -> None:
    manager, backend = environment_manager
    manager.start_reconciliation().result(timeout=2)
    manager.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.close_error = RuntimeError('still alive')
    close_timeouts = []
    received_errors: list[BaseException] = []
    original_close = manager.close

    def close(*, timeout=manager_module._SHUTDOWN_TIMEOUT) -> None:
        close_timeouts.append(timeout)
        original_close(timeout=timeout)

    monkeypatch.setattr(manager, 'close', close)
    monkeypatch.setattr(manager_module, '_manager', manager)
    monkeypatch.setattr(manager_module, '_automatic_shutdown_started', False)
    monkeypatch.setattr(
        qt_environments.notification_manager,
        'receive_error',
        lambda exc_type, error, traceback: received_errors.append(error),
    )

    qt_environments._shutdown_with_notification()
    qt_environments._shutdown_with_notification()
    manager_module._shutdown_plugin_environments_at_exit()

    assert close_timeouts == [manager_module._SHUTDOWN_TIMEOUT]
    assert pool.close_calls == 1
    assert len(received_errors) == 1
    assert isinstance(received_errors[0], PluginEnvironmentError)

    pool.close_error = None
    manager.close(timeout=1)

    assert close_timeouts == [manager_module._SHUTDOWN_TIMEOUT, 1]
    assert pool.close_calls == 2
    assert backend.closed


def test_workers_dialog_replays_state_and_logs(
    qtbot, environment_manager
) -> None:
    manager, _backend = environment_manager
    manager.start_reconciliation().result(timeout=2)
    manager.execute('example-plugin.run', (1,), {}).result(timeout=2)
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)

    dialog.show()
    qtbot.waitUntil(lambda: 'example-plugin.worker' in dialog._cards)

    assert dialog.windowTitle() == 'Plugin Environments'
    assert [label.text() for label in dialog._group_titles] == [
        '<b>Example Plugin</b>'
    ]
    card = dialog._cards['example-plugin.worker']
    assert card._title.text() == '<b>Worker</b>'
    assert card._identifier.text() == 'example-plugin.worker'
    assert card._status.text() == 'Setup: Ready · Worker: Idle'
    assert not card._diagnostic.isVisible()
    assert card._action_button.text() == 'Stop worker'
    assert card._action_button.isEnabled()
    assert 'Executing example-plugin.run' in dialog._logs.toPlainText()
    card._show_logs_button.click()
    assert dialog._filter.currentData() == 'example-plugin.worker'

    dialog.close()
    replacement = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(replacement)
    replacement.show()
    assert 'Executing example-plugin.run' in replacement._logs.toPlainText()


def test_workers_dialog_groups_environments_by_plugin(
    qtbot, environment_manager, monkeypatch
) -> None:
    manager, _backend = environment_manager
    snapshot = _snapshot()
    second = replace(
        snapshot.environments[0],
        plugin='another-plugin',
        plugin_display_name='Another Plugin',
        environment_id='another-plugin.worker',
        display_name='Another Worker',
        recipe=replace(
            snapshot.environments[0].recipe,
            plugin='another-plugin',
            environment_id='another-plugin.worker',
        ),
    )
    grouped_snapshot = replace(
        snapshot, environments=(*snapshot.environments, second)
    )
    monkeypatch.setattr(
        manager_module, '_build_snapshot', lambda backend: grouped_snapshot
    )
    manager.start_reconciliation().result(timeout=2)
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)

    dialog.show()

    assert [label.text() for label in dialog._group_titles] == [
        '<b>Another Plugin</b>',
        '<b>Example Plugin</b>',
    ]
    assert set(dialog._cards) == {
        'another-plugin.worker',
        'example-plugin.worker',
    }


def test_workers_dialog_logs_are_filtered_copied_and_cleared(
    qtbot, environment_manager
) -> None:
    manager, _backend = environment_manager
    manager.start_reconciliation().result(timeout=2)
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    card = dialog._cards['example-plugin.worker']
    card._show_logs_button.click()

    manager._log(
        'visible environment message',
        plugin='example-plugin',
        environment_id='example-plugin.worker',
    )
    manager._log('global message')

    assert 'visible environment message' in dialog._logs.toPlainText()
    assert 'global message' not in dialog._logs.toPlainText()
    dialog._copy_button.click()
    assert (
        'visible environment message'
        in qt_environments.QApplication.clipboard().text()
    )

    dialog._clear_button.click()

    assert manager.log_records() == ()
    assert dialog._logs.toPlainText() == ''


def test_worker_state_refresh_does_not_rebuild_log(
    qtbot, environment_manager, monkeypatch
) -> None:
    manager, _backend = environment_manager
    manager.start_reconciliation().result(timeout=2)
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    rebuilds = []
    monkeypatch.setattr(dialog, '_rebuild_logs', lambda: rebuilds.append(True))

    manager._changed()
    qtbot.waitUntil(
        lambda: (
            dialog._cards['example-plugin.worker']._status.text()
            == 'Setup: Ready · Worker: Stopped'
        )
    )

    assert rebuilds == []


def test_workers_dialog_empty_state_and_close_button(
    qtbot, environment_manager
) -> None:
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._empty.isVisible()
    assert dialog._cards == {}

    dialog._close_button.click()

    assert not dialog.isVisible()


def test_managed_plugin_workers_menu_title() -> None:
    action = next(
        action
        for action in Q_PLUGINS_ACTIONS
        if action.id == 'napari.window.plugins.plugin_workers'
    )

    assert action.title == 'Manage Plugin Environments...'


def test_stop_state_is_owned_by_manager_when_window_closes(
    qtbot, environment_manager
) -> None:
    manager, backend = environment_manager
    manager.start_reconciliation().result(timeout=2)
    manager.execute('example-plugin.run', (1,), {}).result(timeout=2)
    dialog = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    backend.pools[0].close_error = RuntimeError('cleanup failed')

    dialog._cards['example-plugin.worker']._action_button.click()
    dialog.close()
    qtbot.waitUntil(
        lambda: (
            manager.environment_views()[0].worker_state.value
            == 'cleanup_failed'
        )
    )

    replacement = qt_environments.PluginWorkersDialog()
    qtbot.addWidget(replacement)
    replacement.show()
    qtbot.waitUntil(
        lambda: (
            replacement._cards['example-plugin.worker']._action_button.text()
            == 'Retry worker cleanup'
        )
    )
    card = replacement._cards['example-plugin.worker']
    assert card._diagnostic.text() == 'cleanup failed'
    assert not card._diagnostic.isHidden()
    assert card._action_button.isEnabled()

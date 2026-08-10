from __future__ import annotations

import hashlib
import threading
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from napari.plugins import _environment_manager as manager_module
from napari.plugins._environment_manager import (
    PluginEnvironmentManager,
    _SetupState,
    _SnapshotEnvironment,
    _StartupSnapshot,
    _WorkerState,
)
from napari.plugins._environment_types import (
    BackendFailure,
    BackendPhase,
    BackendProgress,
    EnvironmentRecipe,
    WorkerCommand,
)
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentUnavailableError,
    PluginTaskCanceledError,
    PluginTaskPhase,
    PluginTaskState,
    PluginWorkerError,
    WorkerContext,
)

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        CancelCallbackSetter,
        ProgressCallback,
    )


def _recipe(
    plugin: str = 'example-plugin',
    environment: str = 'example-plugin.worker',
    requirement: str = 'example-dependency==1',
) -> EnvironmentRecipe:
    return EnvironmentRecipe(
        plugin=plugin,
        plugin_version='1.0',
        environment_id=environment,
        python='3.12',
        conda=(),
        pypi=(requirement,),
        channels=('conda-forge',),
        worker_package=None,
        worker_content_identity=None,
    )


def _command(
    environment: str = 'example-plugin.worker',
) -> WorkerCommand:
    return WorkerCommand(
        plugin='example-plugin',
        environment_id=environment,
        command_id='example-plugin.run',
        target='worker:run',
        accepts_context=True,
    )


def _snapshot() -> _StartupSnapshot:
    recipe = _recipe()
    return _StartupSnapshot(
        environments=(
            _SnapshotEnvironment(
                plugin='example-plugin',
                plugin_display_name='Example Plugin',
                plugin_enabled=True,
                environment_id=recipe.environment_id,
                display_name='Worker',
                recipe=recipe,
            ),
        ),
        commands=(_command(),),
    )


@dataclass
class _FakeEnvironment:
    name: str


class _FakePool:
    def __init__(self, environment: str) -> None:
        self.environment = environment
        self.closed = False
        self.executed: list[Any] = []
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.close_error: Exception | None = None
        self.failure: BackendFailure | None = None
        self.block_close = False
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        self.close_calls = 0

    def execute(
        self,
        target: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        accepts_context: bool,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any:
        cancel = threading.Event()
        set_cancel_callback(lambda: cancel.set() or True)
        self.entered.set()
        if self.block:
            self.release.wait(2)
        if cancel.is_set():
            from napari.plugins._environment_types import BackendCanceled

            raise BackendCanceled
        if self.failure is not None:
            raise self.failure
        self.executed.append((target, args, kwargs, accepts_context))
        progress(
            BackendProgress(BackendPhase.EXECUTING, 'worker progress', 1, 1)
        )
        return args[0] if args else None

    def close(self) -> None:
        self.close_calls += 1
        self.close_entered.set()
        if self.block_close:
            self.close_release.wait(2)
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class _FakeBackend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: set[str] = set()
        self.provisioned: list[str] = []
        self.removed: list[str] = []
        self.pools: list[_FakePool] = []
        self.fail_provision = False
        self.environment_names_error: Exception | None = None
        self.environment_names_calls = 0
        self.remove_error: Exception | None = None
        self.mutate = True
        self.closed = False

    def worker_content_identity(self, source: Path) -> str:
        return f'sha256:{"a" * 64}'

    def provision_environment(
        self,
        name: str,
        recipe: EnvironmentRecipe,
        *,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
        on_mutation_started=None,
    ) -> _FakeEnvironment:
        self.provisioned.append(name)
        set_cancel_callback(lambda: True)
        if self.mutate and on_mutation_started is not None:
            on_mutation_started()
        if self.fail_provision:
            raise BackendFailure('provision failed')
        self.names.add(name)
        return _FakeEnvironment(name)

    def start_pool(
        self, environment: _FakeEnvironment, *, progress: ProgressCallback
    ) -> _FakePool:
        pool = _FakePool(environment.name)
        self.pools.append(pool)
        return pool

    def remove_environment(
        self,
        name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None:
        self.removed.append(name)
        if self.remove_error is not None:
            raise self.remove_error
        self.names.discard(name)

    def environment_names(self) -> tuple[str, ...]:
        self.environment_names_calls += 1
        if self.environment_names_error is not None:
            raise self.environment_names_error
        return tuple(self.names)

    def close(self, *, timeout: float | None = None) -> None:
        self.closed = True


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = _FakeBackend(tmp_path)
    monkeypatch.setattr(
        manager_module, '_build_snapshot', lambda backend: _snapshot()
    )
    value = PluginEnvironmentManager(
        tmp_path, backend_factory=lambda root: backend, max_parallel_tasks=2
    )
    yield value, backend
    with suppress(Exception):
        value.close()


def test_plugin_author_api_has_only_worker_conveniences() -> None:
    import napari.plugins as plugin_api

    assert plugin_api.WorkerContext is WorkerContext
    assert set(plugin_api.__all__) == {
        'WorkerContext',
        'execute_worker_command',
        'menu_item_template',
        'plugin_manager',
    }


def test_default_environment_root_is_scoped_to_host_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, 'user_data_dir', lambda: str(tmp_path))
    monkeypatch.setattr(
        manager_module, 'PREFIX_PATH', '/host/installation/one'
    )
    first_manager = PluginEnvironmentManager()
    first = first_manager.root
    first_manager.close()
    same = manager_module._default_environment_root()

    monkeypatch.setattr(
        manager_module, 'PREFIX_PATH', '/host/installation/two'
    )
    second_manager = PluginEnvironmentManager()
    second = second_manager.root
    second_manager.close()

    assert first == same
    assert first.parent == second.parent
    assert first != second
    assert first.parent == tmp_path / 'plugin-environments' / 'installations'
    assert first.name == hashlib.sha256(b'/host/installation/one').hexdigest()


def test_public_task_phases_are_runtime_only() -> None:
    from napari.plugins import environments

    assert set(PluginTaskPhase) == {
        PluginTaskPhase.STARTING,
        PluginTaskPhase.EXECUTING,
    }
    assert 'PluginTaskPhase' in environments.__all__


def test_disabled_plugin_declarations_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import npe2

    backend = _FakeBackend(tmp_path)
    manifest = SimpleNamespace(
        name='disabled-plugin',
        display_name='Disabled Plugin',
        package_version='1.0',
        contributions=SimpleNamespace(
            worker_package=None,
            environments=(
                SimpleNamespace(
                    id='disabled-plugin.worker',
                    display_name='Worker',
                    python='3.12',
                    conda=(),
                    pypi=(),
                    channels=('conda-forge',),
                ),
            ),
            commands=(
                SimpleNamespace(
                    id='disabled-plugin.run',
                    environment='disabled-plugin.worker',
                    python_name='worker:run',
                    accepts_worker_context=True,
                ),
            ),
        ),
    )
    monkeypatch.setattr(manager_module, '_iter_manifests', lambda: (manifest,))
    monkeypatch.setattr(npe2.plugin_manager, 'is_disabled', lambda name: True)

    snapshot = manager_module._build_snapshot(backend)

    assert snapshot.environments == ()
    assert snapshot.commands == ()


def test_startup_removes_orphans_then_provisions_sequentially(manager) -> None:
    value, backend = manager
    backend.names.add('old-physical-generation')
    attention = []
    value.add_attention_callback(lambda: attention.append(True))

    value.start_reconciliation().result(timeout=2)

    assert backend.removed == ['old-physical-generation']
    assert backend.provisioned == ['example-plugin.worker']
    assert value.environment_views()[0].setup_state is _SetupState.READY
    assert attention


def test_reusable_startup_does_not_request_attention(manager) -> None:
    value, backend = manager
    backend.names.add('example-plugin.worker')
    backend.mutate = False
    attention = []
    value.add_attention_callback(lambda: attention.append(True))

    value.start_reconciliation().result(timeout=2)

    assert attention == []


def test_orphan_inventory_failure_can_be_retried(manager) -> None:
    value, backend = manager
    backend.environment_names_error = RuntimeError('inventory unavailable')

    value.start_reconciliation().result(timeout=2)

    assert backend.provisioned == ['example-plugin.worker']
    assert value.environment_views()[0].setup_state is _SetupState.READY
    assert value.setup_has_failures()
    with pytest.raises(
        PluginEnvironmentUnavailableError, match='has not completed'
    ):
        value.execute('example-plugin.run', (), {}).result()

    backend.environment_names_error = None
    value.retry_setup().result(timeout=2)

    assert backend.environment_names_calls == 2
    assert backend.provisioned == ['example-plugin.worker']
    assert not value.setup_has_failures()
    assert value.execute('example-plugin.run', (1,), {}).result(timeout=2) == 1


def test_orphan_removal_failure_can_be_retried(manager) -> None:
    value, backend = manager
    backend.names.add('old-physical-generation')
    backend.remove_error = RuntimeError('environment is busy')

    value.start_reconciliation().result(timeout=2)

    assert backend.removed == ['old-physical-generation']
    assert 'old-physical-generation' in backend.names
    assert value.environment_views()[0].setup_state is _SetupState.READY
    assert value.setup_has_failures()

    backend.remove_error = None
    value.retry_setup().result(timeout=2)

    assert backend.removed == [
        'old-physical-generation',
        'old-physical-generation',
    ]
    assert 'old-physical-generation' not in backend.names
    assert backend.provisioned == ['example-plugin.worker']
    assert not value.setup_has_failures()


def test_orphan_removal_failure_can_be_explicitly_ignored(manager) -> None:
    value, backend = manager
    backend.names.add('old-physical-generation')
    backend.remove_error = RuntimeError('environment is busy')

    value.start_reconciliation().result(timeout=2)
    assert value.setup_has_failures()

    value.continue_without_failed()

    assert 'old-physical-generation' in backend.names
    assert not value.setup_has_failures()
    assert value.execute('example-plugin.run', (1,), {}).result(timeout=2) == 1


def test_failed_setup_can_continue_or_retry(manager) -> None:
    value, backend = manager
    backend.fail_provision = True
    value.start_reconciliation().result(timeout=2)
    assert value.setup_has_failures()

    value.continue_without_failed()
    with pytest.raises(PluginEnvironmentUnavailableError, match='unavailable'):
        value.execute('example-plugin.run', (), {}).result()

    backend.fail_provision = False
    value.retry_setup().result(timeout=2)
    assert value.environment_views()[0].setup_state is _SetupState.READY


def test_invalid_snapshot_can_be_fixed_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _FakeBackend(tmp_path)
    calls = 0

    def build_snapshot(backend):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError('duplicate environment aliases')
        return _snapshot()

    monkeypatch.setattr(manager_module, '_build_snapshot', build_snapshot)
    value = PluginEnvironmentManager(
        tmp_path, backend_factory=lambda root: backend
    )
    try:
        value.start_reconciliation().result(timeout=2)
        assert value.setup_has_failures()

        value.retry_setup().result(timeout=2)

        assert not value.setup_has_failures()
        assert value.environment_views()[0].setup_state is _SetupState.READY
    finally:
        value.close()


def test_setup_cancellation_is_latched_before_backend_callback(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, backend = manager
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()

    def build_snapshot(backend):
        snapshot_started.set()
        assert release_snapshot.wait(2)
        return _snapshot()

    monkeypatch.setattr(manager_module, '_build_snapshot', build_snapshot)
    future = value.start_reconciliation()
    assert snapshot_started.wait(1)
    assert value.cancel_setup()
    release_snapshot.set()
    future.result(timeout=2)

    assert value.setup_has_failures()
    assert backend.provisioned == []

    value.retry_setup().result(timeout=2)
    assert not value.setup_has_failures()
    assert backend.provisioned == ['example-plugin.worker']


def test_concurrent_reconciliation_callers_share_one_future(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, _backend = manager
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(9)
    futures: list[Any] = []

    def build_snapshot(backend):
        nonlocal calls
        with calls_lock:
            calls += 1
        return _snapshot()

    def start() -> None:
        barrier.wait()
        futures.append(value.start_reconciliation())

    monkeypatch.setattr(manager_module, '_build_snapshot', build_snapshot)
    threads = [threading.Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    futures[0].result(timeout=2)
    assert len({id(future) for future in futures}) == 1
    assert calls == 1


def test_execution_never_provisions(manager) -> None:
    value, backend = manager

    with pytest.raises(
        PluginEnvironmentUnavailableError, match='not completed'
    ):
        value.execute('example-plugin.run', (), {}).result()

    assert backend.provisioned == []


def test_commands_are_fifo_and_queued_cancel_isolated(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    first = value.execute('example-plugin.run', (1,), {})
    while not backend.pools:
        threading.Event().wait(0.01)
    pool = backend.pools[0]
    pool.block = True
    pool.entered.wait(1)
    second = value.execute('example-plugin.run', (2,), {})
    third = value.execute('example-plugin.run', (3,), {})
    assert second.cancel()
    pool.release.set()

    assert first.result(timeout=2) == 1
    with pytest.raises(PluginTaskCanceledError):
        second.result(timeout=2)
    assert third.result(timeout=2) == 3
    assert [call[1][0] for call in pool.executed] == [1, 3]


def test_idle_stop_blocks_new_work_and_restarts_lazily(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    assert value.execute('example-plugin.run', (1,), {}).result(timeout=2) == 1

    future = value.stop_worker('example-plugin.worker')
    future.result(timeout=2)
    assert value.environment_views()[0].worker_state is _WorkerState.STOPPED
    assert value.execute('example-plugin.run', (2,), {}).result(timeout=2) == 2
    assert len(backend.pools) == 2


def test_failed_stop_retains_retryable_cleanup_state(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.close_error = RuntimeError('still alive')

    with pytest.raises(RuntimeError, match='still alive'):
        value.stop_worker('example-plugin.worker').result(timeout=2)
    assert (
        value.environment_views()[0].worker_state
        is _WorkerState.CLEANUP_FAILED
    )
    with pytest.raises(PluginEnvironmentUnavailableError, match='cleanup'):
        value.execute('example-plugin.run', (2,), {}).result()

    pool.close_error = None
    value.stop_worker('example-plugin.worker').result(timeout=2)
    assert value.environment_views()[0].worker_state is _WorkerState.STOPPED


def test_remote_failure_does_not_discard_warm_pool(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.failure = BackendFailure(
        'bad call', diagnostics={'category': 'remote_exception'}
    )

    with pytest.raises(PluginWorkerError, match='bad call'):
        value.execute('example-plugin.run', (2,), {}).result(timeout=2)
    assert value.environment_views()[0].worker_state is _WorkerState.IDLE


def test_fatal_pool_is_not_restartable_until_cleanup_finishes(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.failure = BackendFailure(
        'worker exited', diagnostics={'category': 'worker_exit'}
    )
    pool.block_close = True

    with pytest.raises(PluginWorkerError, match='worker exited'):
        value.execute('example-plugin.run', (2,), {}).result(timeout=2)
    assert pool.close_entered.wait(1)
    assert value.environment_views()[0].worker_state is _WorkerState.STOPPING
    with pytest.raises(PluginEnvironmentUnavailableError, match='stopping'):
        value.execute('example-plugin.run', (3,), {}).result()

    pool.close_release.set()
    stop_future = value._runtime['example-plugin.worker'].stop_future
    assert stop_future is not None
    stop_future.result(timeout=2)
    assert value.environment_views()[0].worker_state is _WorkerState.STOPPED
    assert value.execute('example-plugin.run', (4,), {}).result(timeout=2) == 4
    assert len(backend.pools) == 2


def test_dispatcher_relinquishes_ownership_before_idle(
    manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    reservation_finished = threading.Event()
    release_dispatcher = threading.Event()
    dispatcher_exited = threading.Event()
    original_finish = value._finish_reservation
    original_dispatch = value._dispatch_environment

    def finish_reservation(runtime, reservation) -> None:
        original_finish(runtime, reservation)
        reservation_finished.set()
        assert release_dispatcher.wait(2)

    def dispatch_environment(environment_id: str) -> None:
        try:
            original_dispatch(environment_id)
        finally:
            dispatcher_exited.set()

    monkeypatch.setattr(value, '_finish_reservation', finish_reservation)
    monkeypatch.setattr(value, '_dispatch_environment', dispatch_environment)

    task = value.execute('example-plugin.run', (1,), {})
    assert reservation_finished.wait(1)
    assert task.result(timeout=2) == 1
    runtime = value._runtime['example-plugin.worker']
    with value._lock:
        assert runtime.dispatcher_active
        assert runtime.state is _WorkerState.BUSY

    pool = backend.pools[0]
    pool.close_error = RuntimeError('still alive')
    close_errors: list[BaseException] = []

    def close() -> None:
        try:
            value.close(timeout=1)
        except Exception as error:  # noqa: BLE001
            close_errors.append(error)

    close_thread = threading.Thread(target=close)
    close_thread.start()
    close_thread.join(2)
    assert not close_thread.is_alive()
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], PluginEnvironmentError)
    assert runtime.state is _WorkerState.CLEANUP_FAILED

    release_dispatcher.set()
    assert dispatcher_exited.wait(1)
    assert runtime.state is _WorkerState.CLEANUP_FAILED

    pool.close_error = None
    value.close(timeout=1)


def test_logs_are_bounded_and_replayable(manager) -> None:
    value, _backend = manager
    value.start_reconciliation().result(timeout=2)
    for index in range(2100):
        value._log(str(index))
    replayed = []
    value.add_log_callback(replayed.append)
    assert len(replayed) == 2000
    assert replayed[-1].message == '2099'


def test_shutdown_closes_warm_pools(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)

    value.close()

    assert backend.pools[0].closed
    assert backend.closed


def test_shutdown_reuses_inflight_stop_instead_of_closing_twice(
    manager,
) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.block_close = True

    stop_future = value.stop_worker('example-plugin.worker')
    assert pool.close_entered.wait(1)
    close_error: list[BaseException] = []

    def close() -> None:
        try:
            value.close(timeout=1)
        except Exception as error:  # noqa: BLE001
            close_error.append(error)

    close_thread = threading.Thread(target=close)
    close_thread.start()
    pool.close_release.set()
    close_thread.join(2)

    assert not close_thread.is_alive()
    assert close_error == []
    assert stop_future.done()
    assert pool.close_calls == 1


def test_failed_shutdown_can_retry_pool_cleanup(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.close_error = RuntimeError('still alive')

    with pytest.raises(PluginEnvironmentError, match='clean up'):
        value.close(timeout=1)

    assert pool.close_calls == 1
    assert not backend.closed
    pool.close_error = None

    value.close(timeout=1)

    assert pool.close_calls == 2
    assert backend.closed


def test_shutdown_uses_one_deadline_and_can_resume(manager) -> None:
    value, backend = manager
    value.start_reconciliation().result(timeout=2)
    value.execute('example-plugin.run', (1,), {}).result(timeout=2)
    pool = backend.pools[0]
    pool.block_close = True
    started = manager_module.monotonic()

    with pytest.raises(PluginEnvironmentError, match='clean up'):
        value.close(timeout=0.05)

    assert manager_module.monotonic() - started < 0.5
    assert pool.close_calls == 1
    pool.close_release.set()
    value.close(timeout=1)

    assert pool.close_calls == 1
    assert backend.closed


def test_task_state_is_authoritative(manager) -> None:
    value, _backend = manager
    value.start_reconciliation().result(timeout=2)
    task = value.execute('example-plugin.run', (1,), {})
    assert task.result(timeout=2) == 1
    assert task.state is PluginTaskState.COMPLETED

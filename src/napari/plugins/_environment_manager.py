"""Startup reconciliation and runtime dispatch for plugin workers."""

from __future__ import annotations

import atexit
import logging
import threading
import tomllib
from collections import deque
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendProgress,
    EnvironmentBackend,
    EnvironmentRecipe,
    WorkerCommand,
)
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskPhase,
    PluginWorkerError,
    PluginWorkerFailure,
)
from napari.utils._platformdirs import user_data_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from npe2.manifest import PluginManifest

    from napari.plugins._environment_types import BackendPool

logger = logging.getLogger(__name__)
_SHUTDOWN_TIMEOUT = 30.0


class _SetupState(Enum):
    PENDING = 'pending'
    SETTING_UP = 'setting_up'
    READY = 'ready'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class _WorkerState(Enum):
    STOPPED = 'stopped'
    STARTING = 'starting'
    BUSY = 'busy'
    IDLE = 'idle'
    STOPPING = 'stopping'
    CLEANUP_FAILED = 'cleanup_failed'


@dataclass(frozen=True)
class _SnapshotEnvironment:
    plugin: str
    plugin_display_name: str
    plugin_enabled: bool
    environment_id: str
    display_name: str
    recipe: EnvironmentRecipe


@dataclass(frozen=True)
class _StartupSnapshot:
    environments: tuple[_SnapshotEnvironment, ...]
    commands: tuple[WorkerCommand, ...]


@dataclass(frozen=True)
class _EnvironmentView:
    plugin: str
    plugin_display_name: str
    environment_id: str
    display_name: str
    setup_state: _SetupState
    worker_state: _WorkerState
    diagnostic: str | None


@dataclass(frozen=True)
class _PluginLogRecord:
    sequence: int
    timestamp: datetime
    plugin: str | None
    environment_id: str | None
    message: str
    level: str = 'info'


@dataclass
class _SetupEntry:
    declaration: _SnapshotEnvironment
    state: _SetupState = _SetupState.PENDING
    environment: Any = None
    diagnostic: str | None = None


@dataclass
class _Reservation:
    task: PluginTask[Any]
    command: WorkerCommand
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class _RuntimeEntry:
    setup: _SetupEntry
    state: _WorkerState = _WorkerState.STOPPED
    pool: BackendPool | None = None
    queue: deque[_Reservation] = field(default_factory=deque)
    current: _Reservation | None = None
    dispatcher_active: bool = False
    stop_future: Future[None] | None = None
    diagnostic: str | None = None


def _safe_relative_path(base: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f'{label} must be relative to the plugin manifest')
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise TypeError(
            f'{label} must remain inside the plugin package'
        ) from error
    return resolved


def _manifest_source_directory(manifest: PluginManifest) -> Path:
    source_file = getattr(manifest, '_source_file', None)
    if source_file is None:
        raise ValueError(
            f'Plugin {manifest.name!r} has no manifest source path; managed '
            'worker paths cannot be resolved'
        )
    return Path(source_file).resolve().parent


def _worker_package_path(base: Path, value: str) -> Path:
    path = _safe_relative_path(base, value, 'Worker package path')
    pyproject = path / 'pyproject.toml'
    if not path.is_dir() or not pyproject.is_file():
        raise ValueError(
            f'Worker package must be a directory containing pyproject.toml: {path}'
        )
    try:
        document = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f'Invalid worker package metadata: {pyproject}'
        ) from error
    project = document.get('project')
    if not isinstance(project, dict) or not isinstance(
        project.get('name'), str
    ):
        raise TypeError(
            f'Worker package must declare [project].name: {pyproject}'
        )
    dependencies = project.get('dependencies', [])
    optional = project.get('optional-dependencies', {})
    dynamic = project.get('dynamic', [])
    if (
        dependencies
        or (isinstance(optional, dict) and any(optional.values()))
        or (
            isinstance(dynamic, list)
            and {'dependencies', 'optional-dependencies'} & set(dynamic)
        )
    ):
        raise ValueError(
            'Worker package dependencies must be declared by the environment '
            f'manifest, not {pyproject}'
        )
    return path


def _iter_manifests() -> tuple[PluginManifest, ...]:
    from npe2 import plugin_manager

    return tuple(plugin_manager.iter_manifests(disabled=None))


def _build_snapshot(backend: EnvironmentBackend) -> _StartupSnapshot:
    from npe2 import plugin_manager

    environments: list[_SnapshotEnvironment] = []
    commands: list[WorkerCommand] = []
    environment_aliases: dict[str, str] = {}
    command_aliases: dict[str, str] = {}

    for manifest in _iter_manifests():
        enabled = not plugin_manager.is_disabled(manifest.name)
        if not enabled:
            continue
        plugin_display_name = manifest.display_name or manifest.name
        contributions = manifest.contributions
        declared = {
            str(environment.id): environment
            for environment in (
                getattr(contributions, 'environments', None) or ()
            )
        }
        worker_value = getattr(contributions, 'worker_package', None)
        worker_package = (
            None
            if worker_value is None
            else _worker_package_path(
                _manifest_source_directory(manifest), str(worker_value)
            )
        )
        worker_identity = (
            None
            if worker_package is None
            else backend.worker_content_identity(worker_package)
        )
        recipes: dict[str, EnvironmentRecipe] = {}
        for environment_id, environment in declared.items():
            alias = environment_id.casefold()
            previous = environment_aliases.get(alias)
            if previous is not None:
                raise ValueError(
                    f'Managed environment IDs alias case-insensitively: '
                    f'{previous!r} and {environment_id!r}'
                )
            environment_aliases[alias] = environment_id
            recipe = EnvironmentRecipe(
                plugin=manifest.name,
                plugin_version=manifest.package_version or '0+unknown',
                environment_id=environment_id,
                python=str(environment.python),
                conda=tuple(environment.conda),
                pypi=tuple(environment.pypi),
                channels=tuple(environment.channels),
                worker_package=worker_package,
                worker_content_identity=worker_identity,
            )
            recipes[environment_id] = recipe
            display_name = getattr(environment, 'display_name', None)
            environments.append(
                _SnapshotEnvironment(
                    plugin=manifest.name,
                    plugin_display_name=plugin_display_name,
                    plugin_enabled=enabled,
                    environment_id=environment_id,
                    display_name=str(
                        display_name or environment_id.rsplit('.', 1)[-1]
                    ),
                    recipe=recipe,
                )
            )
        for command in contributions.commands or ():
            environment_value = getattr(command, 'environment', None)
            if environment_value is None:
                continue
            command_id = str(command.id)
            alias = command_id.casefold()
            previous = command_aliases.get(alias)
            if previous is not None:
                raise ValueError(
                    f'Worker command IDs alias case-insensitively: '
                    f'{previous!r} and {command_id!r}'
                )
            command_aliases[alias] = command_id
            environment_id = str(environment_value)
            if environment_id not in recipes or command.python_name is None:
                raise ValueError(
                    f'Worker command {command_id!r} has an invalid environment '
                    'or qualified Python target'
                )
            commands.append(
                WorkerCommand(
                    plugin=manifest.name,
                    environment_id=environment_id,
                    command_id=command_id,
                    target=str(command.python_name),
                    accepts_context=bool(
                        getattr(command, 'accepts_worker_context', False)
                    ),
                )
            )
    return _StartupSnapshot(
        environments=tuple(
            sorted(
                environments, key=lambda item: item.environment_id.casefold()
            )
        ),
        commands=tuple(
            sorted(commands, key=lambda item: item.command_id.casefold())
        ),
    )


def _worker_failure(error: BackendFailure) -> PluginWorkerFailure | None:
    diagnostics = error.diagnostics
    if diagnostics is None:
        return None
    return PluginWorkerFailure(
        category=diagnostics.get('category'),
        message=str(diagnostics.get('message', str(error))),
        target=diagnostics.get('target'),
        traceback=diagnostics.get('traceback'),
        remote_exception_type=diagnostics.get('remote_exception_type'),
        remote_exception_message=diagnostics.get('remote_exception_message'),
        worker_environment=diagnostics.get('worker_environment'),
        worker_pid=diagnostics.get('worker_pid'),
        exit_code=diagnostics.get('exit_code'),
        signal=diagnostics.get('signal'),
        timeout=diagnostics.get('timeout'),
        elapsed=diagnostics.get('elapsed'),
        serialization_context=diagnostics.get('serialization_context'),
    )


def _fatal_worker_failure(error: BackendFailure) -> bool:
    category = (
        None
        if error.diagnostics is None
        else error.diagnostics.get('category')
    )
    return category not in {
        'remote_exception',
        'serialization',
        'invalid_arguments',
    }


class PluginEnvironmentManager:
    """Own one immutable startup snapshot and its warm worker pools."""

    def __init__(
        self,
        root: Path | None = None,
        backend_factory: Callable[[Path], EnvironmentBackend] | None = None,
        max_parallel_tasks: int = 4,
    ) -> None:
        self.root = root or Path(user_data_dir()) / 'plugin-environments'
        self._backend_factory = backend_factory
        self._backend: EnvironmentBackend | None = None
        self._setup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='napari-plugin-setup'
        )
        self._execution_executor = ThreadPoolExecutor(
            max_workers=max_parallel_tasks,
            thread_name_prefix='napari-plugin-worker',
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=max_parallel_tasks,
            thread_name_prefix='napari-plugin-control',
        )
        self._snapshot: _StartupSnapshot | None = None
        self._setup_entries: dict[str, _SetupEntry] = {}
        self._runtime: dict[str, _RuntimeEntry] = {}
        self._commands: dict[str, WorkerCommand] = {}
        self._setup_future: Future[None] | None = None
        self._setup_cancel: Callable[[], Any] | None = None
        self._setup_cancel_requested = False
        self._setup_failure: str | None = None
        self._setup_published = False
        self._attention_callbacks: list[Callable[[], Any]] = []
        self._state_callbacks: list[Callable[[], Any]] = []
        self._log_callbacks: list[Callable[[_PluginLogRecord], Any]] = []
        self._logs: deque[_PluginLogRecord] = deque(maxlen=2000)
        self._log_sequence = 0
        self._closing = False
        self._close_failed = False
        self._closed = False
        self._close_attempt: Future[None] | None = None
        self._backend_close_future: Future[None] | None = None
        self._lock = threading.RLock()

    def _get_backend(self) -> EnvironmentBackend:
        with self._lock:
            if self._backend is not None:
                return self._backend
            factory = self._backend_factory
        if factory is None:
            from napari.plugins._wetlands import WetlandsBackend

            factory = WetlandsBackend
        backend = factory(self.root)
        with self._lock:
            if self._backend is None:
                self._backend = backend
                return backend
            existing = self._backend
        backend.close()
        return existing

    def add_attention_callback(self, callback: Callable[[], Any]) -> None:
        with self._lock:
            if callback not in self._attention_callbacks:
                self._attention_callbacks.append(callback)

    def add_state_callback(
        self, callback: Callable[[], Any], *, replay: bool = True
    ) -> Callable[[], None]:
        with self._lock:
            if callback not in self._state_callbacks:
                self._state_callbacks.append(callback)
        if replay:
            callback()
        return lambda: self._remove_callback(self._state_callbacks, callback)

    def add_log_callback(
        self,
        callback: Callable[[_PluginLogRecord], Any],
        *,
        replay: bool = True,
    ) -> Callable[[], None]:
        with self._lock:
            if callback not in self._log_callbacks:
                self._log_callbacks.append(callback)
            records = tuple(self._logs) if replay else ()
        for record in records:
            callback(record)
        return lambda: self._remove_callback(self._log_callbacks, callback)

    def _remove_callback(self, callbacks: list[Any], callback: Any) -> None:
        with self._lock:
            if callback in callbacks:
                callbacks.remove(callback)

    def _attention(self) -> None:
        with self._lock:
            callbacks = tuple(self._attention_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception('Plugin setup attention callback failed')

    def _changed(self) -> None:
        with self._lock:
            callbacks = tuple(self._state_callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception('Plugin worker state callback failed')

    def _log(
        self,
        message: str,
        *,
        plugin: str | None = None,
        environment_id: str | None = None,
        level: str = 'info',
    ) -> None:
        with self._lock:
            self._log_sequence += 1
            record = _PluginLogRecord(
                self._log_sequence,
                datetime.now(UTC),
                plugin,
                environment_id,
                message,
                level,
            )
            self._logs.append(record)
            callbacks = tuple(self._log_callbacks)
        for callback in callbacks:
            try:
                callback(record)
            except Exception:
                logger.exception('Plugin worker log callback failed')

    def log_records(self) -> tuple[_PluginLogRecord, ...]:
        with self._lock:
            return tuple(self._logs)

    def clear_logs(self) -> None:
        with self._lock:
            self._logs.clear()
        self._changed()

    def start_reconciliation(self) -> Future[None]:
        """Freeze plugin declarations and reconcile them exactly once."""

        with self._lock:
            if self._setup_future is not None:
                return self._setup_future
            if self._closing or self._close_failed:
                future: Future[None] = Future()
                future.set_exception(
                    RuntimeError('Plugin workers are shutting down')
                )
                return future
            self._setup_future = self._setup_executor.submit(
                self._initialize_and_reconcile
            )
            future = self._setup_future
            future.add_done_callback(self._setup_finished)
        self._changed()
        return future

    def _initialize_and_reconcile(self) -> None:
        try:
            backend = self._get_backend()
            snapshot = _build_snapshot(backend)
        except Exception as error:  # noqa: BLE001
            with self._lock:
                self._setup_failure = str(error)
            self._log(
                f'Could not read plugin worker declarations: {error}',
                level='error',
            )
            self._attention()
            self._changed()
            return
        with self._lock:
            self._snapshot = snapshot
            self._setup_failure = None
            self._setup_entries = {
                item.environment_id: _SetupEntry(item)
                for item in snapshot.environments
            }
            self._commands = {
                command.command_id: command for command in snapshot.commands
            }
            self._runtime = {
                environment_id: _RuntimeEntry(entry)
                for environment_id, entry in self._setup_entries.items()
            }
        self._changed()
        self._reconcile()

    def retry_setup(self) -> Future[None]:
        with self._lock:
            if self._closing or self._close_failed:
                future: Future[None] = Future()
                future.set_exception(
                    RuntimeError('Plugin workers are shutting down')
                )
                return future
            if (
                self._setup_future is not None
                and not self._setup_future.done()
            ):
                return self._setup_future
            if self._snapshot is None:
                self._setup_failure = None
                self._setup_cancel_requested = False
                self._setup_future = self._setup_executor.submit(
                    self._initialize_and_reconcile
                )
                future = self._setup_future
                future.add_done_callback(self._setup_finished)
                self._changed()
                return future
            for entry in self._setup_entries.values():
                if entry.state in {_SetupState.FAILED, _SetupState.SKIPPED}:
                    entry.state = _SetupState.PENDING
                    entry.diagnostic = None
            self._setup_published = False
            self._setup_cancel_requested = False
            self._setup_cancel = None
            self._setup_failure = None
            self._setup_future = self._setup_executor.submit(self._reconcile)
            future = self._setup_future
            future.add_done_callback(self._setup_finished)
        self._changed()
        return future

    def _setup_finished(self, _future: Future[None]) -> None:
        """Publish the terminal coordinator state after Future completion."""

        self._changed()

    def continue_without_failed(self) -> None:
        with self._lock:
            if (
                self._setup_future is not None
                and not self._setup_future.done()
            ):
                raise RuntimeError('Plugin environment setup is still running')
            for entry in self._setup_entries.values():
                if entry.state in {
                    _SetupState.PENDING,
                    _SetupState.SETTING_UP,
                    _SetupState.FAILED,
                }:
                    entry.state = _SetupState.SKIPPED
            self._setup_failure = None
            self._setup_cancel_requested = False
            self._setup_cancel = None
            self._setup_published = True
        self._log(
            'Continuing without unavailable plugin workers', level='warning'
        )
        self._changed()

    def cancel_setup(self) -> bool:
        with self._lock:
            if (
                self._setup_future is None
                or self._setup_future.done()
                or self._setup_published
            ):
                return False
            self._setup_cancel_requested = True
            callback = self._setup_cancel
        if callback is not None:
            callback()
        return True

    def _setup_was_canceled(self) -> bool:
        with self._lock:
            return self._setup_cancel_requested

    def _finish_setup_cancellation(
        self, setup: _SetupEntry | None = None
    ) -> None:
        with self._lock:
            self._setup_cancel = None
            if setup is not None:
                setup.state = _SetupState.SKIPPED
                setup.diagnostic = 'Setup was canceled'
            if not (self._closing or self._close_failed):
                self._setup_failure = (
                    'Setup was canceled. Retry setup or continue without the '
                    'affected plugin workers.'
                )
            closing = self._closing or self._close_failed
        if not closing:
            self._attention()
        self._changed()

    def _reconcile(self) -> None:
        backend = self._get_backend()
        orphan_failures: list[str] = []
        if self._setup_was_canceled():
            self._finish_setup_cancellation()
            return
        expected = set(self._setup_entries)
        try:
            orphans = sorted(set(backend.environment_names()) - expected)
        except Exception as error:  # noqa: BLE001
            message = f'Could not inspect managed environments: {error}'
            orphan_failures.append(message)
            self._log(message, level='error')
            self._attention()
            orphans = []
        for name in orphans:
            if self._setup_was_canceled():
                self._finish_setup_cancellation()
                return
            self._attention()
            self._log(
                f'Removing environment left by an uninstalled plugin: {name}'
            )
            try:
                backend.remove_environment(
                    name,
                    progress=lambda update, n=name: self._setup_progress(
                        None, n, update
                    ),
                    set_cancel_callback=self._set_setup_cancel,
                )
            except BackendCanceled:
                self._finish_setup_cancellation()
                return
            except Exception as error:  # noqa: BLE001
                message = (
                    f'Could not remove orphaned environment {name}: {error}'
                )
                orphan_failures.append(message)
                self._log(
                    message,
                    level='error',
                )
                self._attention()
            finally:
                self._clear_setup_cancel()
        for setup in tuple(self._setup_entries.values()):
            with self._lock:
                canceled = self._setup_cancel_requested
                if setup.state is _SetupState.READY:
                    continue
                if canceled:
                    pass
                else:
                    setup.state = _SetupState.SETTING_UP
                    setup.diagnostic = None
            if canceled:
                self._finish_setup_cancellation()
                return
            self._changed()
            declaration = setup.declaration
            self._log(
                f'Checking environment {declaration.display_name}',
                plugin=declaration.plugin,
                environment_id=declaration.environment_id,
            )
            try:
                environment = backend.provision_environment(
                    declaration.environment_id,
                    declaration.recipe,
                    progress=lambda update, item=declaration: (
                        self._setup_progress(
                            item.plugin, item.environment_id, update
                        )
                    ),
                    set_cancel_callback=self._set_setup_cancel,
                    on_mutation_started=self._attention,
                )
            except BackendCanceled:
                self._log(
                    'Environment setup canceled',
                    plugin=declaration.plugin,
                    environment_id=declaration.environment_id,
                    level='warning',
                )
                self._finish_setup_cancellation(setup)
                return
            except Exception as error:  # noqa: BLE001
                with self._lock:
                    setup.state = _SetupState.FAILED
                    setup.diagnostic = str(error)
                self._log(
                    f'Environment setup failed: {error}',
                    plugin=declaration.plugin,
                    environment_id=declaration.environment_id,
                    level='error',
                )
                self._attention()
                self._changed()
                continue
            finally:
                self._clear_setup_cancel()
            with self._lock:
                setup.environment = environment
                setup.state = _SetupState.READY
                setup.diagnostic = None
            self._log(
                'Environment ready',
                plugin=declaration.plugin,
                environment_id=declaration.environment_id,
            )
            self._changed()
        with self._lock:
            self._setup_cancel = None
            canceled = self._setup_cancel_requested
            if not canceled:
                self._setup_failure = '\n'.join(orphan_failures) or None
                failed = any(
                    entry.state is _SetupState.FAILED
                    for entry in self._setup_entries.values()
                )
                if not failed and self._setup_failure is None:
                    self._setup_published = True
        if canceled:
            self._finish_setup_cancellation()
            return
        self._changed()

    def _set_setup_cancel(self, callback: Callable[[], Any]) -> None:
        with self._lock:
            self._setup_cancel = callback
            requested = self._setup_cancel_requested or self._closing
        if requested:
            callback()

    def _clear_setup_cancel(self) -> None:
        with self._lock:
            self._setup_cancel = None

    def _setup_progress(
        self,
        plugin: str | None,
        environment_id: str,
        update: BackendProgress,
    ) -> None:
        self._log(
            update.message,
            plugin=plugin,
            environment_id=environment_id,
        )

    def environment_views(self) -> tuple[_EnvironmentView, ...]:
        with self._lock:
            views = []
            for environment_id, setup in self._setup_entries.items():
                runtime = self._runtime[environment_id]
                declaration = setup.declaration
                views.append(
                    _EnvironmentView(
                        plugin=declaration.plugin,
                        plugin_display_name=declaration.plugin_display_name,
                        environment_id=environment_id,
                        display_name=declaration.display_name,
                        setup_state=setup.state,
                        worker_state=runtime.state,
                        diagnostic=runtime.diagnostic or setup.diagnostic,
                    )
                )
        return tuple(
            sorted(views, key=lambda item: item.environment_id.casefold())
        )

    def setup_running(self) -> bool:
        with self._lock:
            return (
                self._setup_future is not None
                and not self._setup_future.done()
            )

    def setup_has_failures(self) -> bool:
        with self._lock:
            return self._setup_failure is not None or any(
                entry.state is _SetupState.FAILED
                for entry in self._setup_entries.values()
            )

    def execute(
        self,
        command_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> PluginTask[Any]:
        task: PluginTask[Any] = PluginTask()
        with self._lock:
            command = self._commands.get(command_id)
            if self._closing or self._close_failed:
                error = self._unavailable(
                    'Plugin workers are shutting down', command_id=command_id
                )
            elif not self._setup_published:
                error = self._unavailable(
                    'Plugin worker setup has not completed. Restart napari to retry setup.',
                    command=command,
                    command_id=command_id,
                )
            elif command is None:
                error = self._unavailable(
                    f'No enabled plugin worker command {command_id!r} exists in '
                    'this napari session. Restart napari after plugin changes.',
                    command_id=command_id,
                )
            else:
                runtime = self._runtime[command.environment_id]
                if runtime.setup.state is not _SetupState.READY:
                    error = self._unavailable(
                        'This plugin environment is unavailable. Restart napari '
                        'to retry environment setup.',
                        command=command,
                        command_id=command_id,
                    )
                elif runtime.state in {
                    _WorkerState.STOPPING,
                    _WorkerState.CLEANUP_FAILED,
                }:
                    error = self._unavailable(
                        'The plugin worker is stopping or requires cleanup. Try '
                        'again after it has stopped.',
                        command=command,
                        command_id=command_id,
                    )
                else:
                    error = None
                    reservation = _Reservation(task, command, args, kwargs)
                    runtime.queue.append(reservation)
                    task._set_cancel_callback(
                        lambda item=reservation: self._cancel_reservation(item)
                    )
                    if not runtime.dispatcher_active:
                        runtime.dispatcher_active = True
                        self._execution_executor.submit(
                            self._dispatch_environment, command.environment_id
                        )
        if error is not None:
            task._set_error(error)
            return task
        self._log(
            f'Accepted command {command_id}',
            plugin=command.plugin,
            environment_id=command.environment_id,
        )
        self._changed()
        return task

    def _unavailable(
        self,
        message: str,
        *,
        command: WorkerCommand | None = None,
        command_id: str | None = None,
    ) -> PluginEnvironmentUnavailableError:
        return PluginEnvironmentUnavailableError(
            message,
            plugin=None if command is None else command.plugin,
            environment=None if command is None else command.environment_id,
            command=command_id,
        )

    def _cancel_reservation(self, reservation: _Reservation) -> bool:
        with self._lock:
            runtime = self._runtime.get(reservation.command.environment_id)
            if runtime is None:
                return False
            if runtime.current is reservation:
                return True
            try:
                runtime.queue.remove(reservation)
            except ValueError:
                return False
        reservation.task._set_canceled()
        self._log(
            f'Canceled queued command {reservation.command.command_id}',
            plugin=reservation.command.plugin,
            environment_id=reservation.command.environment_id,
        )
        self._changed()
        return True

    def _dispatch_environment(self, environment_id: str) -> None:
        while True:
            with self._lock:
                runtime = self._runtime[environment_id]
                if self._closing or self._close_failed or not runtime.queue:
                    runtime.dispatcher_active = False
                    if (
                        runtime.pool is not None
                        and runtime.current is None
                        and runtime.state
                        not in {
                            _WorkerState.STOPPING,
                            _WorkerState.CLEANUP_FAILED,
                        }
                    ):
                        runtime.state = _WorkerState.IDLE
                    self._changed()
                    return
                reservation = runtime.queue.popleft()
                runtime.current = reservation
                pool = runtime.pool
                if pool is None:
                    runtime.state = _WorkerState.STARTING
                else:
                    runtime.state = _WorkerState.BUSY
            self._changed()
            task = reservation.task
            command = reservation.command
            if task.cancellation_requested:
                task._set_canceled()
                self._finish_reservation(runtime, reservation)
                continue
            if pool is None:
                task._set_running(
                    PluginTaskPhase.STARTING,
                    f'Starting worker for {environment_id}',
                )
                try:
                    pool = self._get_backend().start_pool(
                        runtime.setup.environment,
                        progress=lambda update, task=task: (
                            task._report_progress(
                                PluginTaskPhase.STARTING,
                                update.message,
                                update.current,
                                update.total,
                            )
                        ),
                    )
                except Exception as error:  # noqa: BLE001
                    self._fail_start(runtime, reservation, error)
                    return
                with self._lock:
                    runtime.pool = pool
                    runtime.state = _WorkerState.BUSY
                self._log(
                    'Worker started',
                    plugin=command.plugin,
                    environment_id=environment_id,
                )
            task._set_running(
                PluginTaskPhase.EXECUTING, f'Executing {command.command_id}'
            )
            self._log(
                f'Executing {command.command_id}',
                plugin=command.plugin,
                environment_id=environment_id,
            )
            try:
                result = pool.execute(
                    command.target,
                    reservation.args,
                    reservation.kwargs,
                    accepts_context=command.accepts_context,
                    progress=lambda update, task=task: task._report_progress(
                        PluginTaskPhase.EXECUTING,
                        update.message,
                        update.current,
                        update.total,
                    ),
                    set_cancel_callback=task._set_cancel_callback,
                )
            except BackendCanceled:
                task._set_canceled()
                fatal = False
            except BackendFailure as error:
                task._set_error(
                    PluginWorkerError(
                        str(error),
                        plugin=command.plugin,
                        environment=environment_id,
                        command=command.command_id,
                        phase=PluginTaskPhase.EXECUTING,
                        details=error.details,
                        failure=_worker_failure(error),
                    )
                )
                fatal = _fatal_worker_failure(error)
            except Exception as error:  # noqa: BLE001
                task._set_error(
                    PluginWorkerError(
                        str(error),
                        plugin=command.plugin,
                        environment=environment_id,
                        command=command.command_id,
                        phase=PluginTaskPhase.EXECUTING,
                    )
                )
                fatal = True
            else:
                task._set_result(result)
                fatal = False
            self._log(
                f'Command {command.command_id} finished with {task.state.value}',
                plugin=command.plugin,
                environment_id=environment_id,
                level='error' if task.error is not None else 'info',
            )
            if fatal:
                self._fatal_pool(runtime, reservation)
                return
            self._finish_reservation(runtime, reservation)

    def _finish_reservation(
        self, runtime: _RuntimeEntry, reservation: _Reservation
    ) -> None:
        with self._lock:
            if runtime.current is reservation:
                runtime.current = None
            if runtime.pool is None:
                runtime.state = _WorkerState.STOPPED
        self._changed()

    def _fail_start(
        self,
        runtime: _RuntimeEntry,
        reservation: _Reservation,
        error: Exception,
    ) -> None:
        with self._lock:
            queued = tuple(runtime.queue)
            runtime.queue.clear()
            runtime.current = None
            runtime.dispatcher_active = False
            runtime.state = _WorkerState.STOPPED
            runtime.pool = None
        affected = (reservation, *queued)
        for item in affected:
            item.task._set_error(
                PluginWorkerError(
                    f'Could not start plugin worker: {error}',
                    plugin=item.command.plugin,
                    environment=item.command.environment_id,
                    command=item.command.command_id,
                    phase=PluginTaskPhase.STARTING,
                )
            )
        self._log(
            f'Worker startup failed: {error}',
            plugin=reservation.command.plugin,
            environment_id=reservation.command.environment_id,
            level='error',
        )
        self._changed()

    def _fatal_pool(
        self, runtime: _RuntimeEntry, reservation: _Reservation
    ) -> None:
        with self._lock:
            queued = tuple(runtime.queue)
            runtime.queue.clear()
            runtime.current = None
            runtime.dispatcher_active = False
            pool = runtime.pool
            runtime.state = _WorkerState.STOPPING
            future = (
                None
                if pool is None
                else self._start_pool_close_locked(
                    reservation.command.environment_id, runtime, pool
                )
            )
        for item in queued:
            item.task._set_error(
                PluginWorkerError(
                    'The worker pool failed before this command could run; '
                    'the command was not replayed',
                    plugin=item.command.plugin,
                    environment=item.command.environment_id,
                    command=item.command.command_id,
                )
            )
        if future is None:
            with self._lock:
                runtime.state = _WorkerState.STOPPED
        self._changed()

    def _start_pool_close_locked(
        self,
        environment_id: str,
        runtime: _RuntimeEntry,
        pool: BackendPool,
    ) -> Future[None]:
        existing = runtime.stop_future
        if existing is not None and not existing.done():
            return existing
        runtime.state = _WorkerState.STOPPING
        future = self._control_executor.submit(
            self._close_runtime_pool, environment_id, runtime, pool
        )
        runtime.stop_future = future
        return future

    def stop_worker(self, environment_id: str) -> Future[None]:
        with self._lock:
            runtime = self._runtime.get(environment_id)
            if runtime is None:
                raise KeyError(environment_id)
            if (
                runtime.stop_future is not None
                and not runtime.stop_future.done()
            ):
                return runtime.stop_future
            if (
                runtime.state
                not in {
                    _WorkerState.IDLE,
                    _WorkerState.CLEANUP_FAILED,
                }
                or runtime.current is not None
                or runtime.queue
            ):
                raise RuntimeError('Only an idle plugin worker can be stopped')
            if runtime.pool is None:
                raise RuntimeError('The plugin worker is already stopped')
            future = self._start_pool_close_locked(
                environment_id, runtime, runtime.pool
            )
        self._log(
            'Stopping worker',
            plugin=runtime.setup.declaration.plugin,
            environment_id=environment_id,
        )
        self._changed()
        return future

    def _close_runtime_pool(
        self,
        environment_id: str,
        runtime: _RuntimeEntry,
        pool: BackendPool,
    ) -> None:
        try:
            pool.close()
        except Exception as error:
            with self._lock:
                if runtime.pool is pool:
                    runtime.state = _WorkerState.CLEANUP_FAILED
                    runtime.diagnostic = str(error)
            self._log(
                f'Worker cleanup failed: {error}',
                plugin=runtime.setup.declaration.plugin,
                environment_id=environment_id,
                level='error',
            )
            self._changed()
            raise
        with self._lock:
            if runtime.pool is pool:
                runtime.pool = None
                runtime.state = _WorkerState.STOPPED
                runtime.diagnostic = None
        self._log(
            'Worker stopped',
            plugin=runtime.setup.declaration.plugin,
            environment_id=environment_id,
        )
        self._changed()

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - monotonic())

    def _wait_for_close_attempt(
        self, attempt: Future[None], deadline: float
    ) -> None:
        try:
            attempt.result(timeout=self._remaining(deadline))
        except FutureTimeoutError as error:
            raise PluginEnvironmentError(
                'Plugin worker shutdown is still in progress'
            ) from error

    def close(self, timeout: float = _SHUTDOWN_TIMEOUT) -> None:
        """Cancel work and close workers within one retryable deadline."""

        if timeout < 0:
            raise ValueError('timeout must be non-negative')
        deadline = monotonic() + timeout
        with self._lock:
            if self._closed:
                return
            if self._closing:
                attempt = self._close_attempt
                assert attempt is not None
                wait_for_attempt = True
            else:
                self._closing = True
                self._close_failed = False
                attempt = Future()
                self._close_attempt = attempt
                wait_for_attempt = False
            self._setup_cancel_requested = True
            setup_cancel = self._setup_cancel
            reservations = [
                item
                for runtime in self._runtime.values()
                for item in (
                    *(
                        (runtime.current,)
                        if runtime.current is not None
                        else ()
                    ),
                    *runtime.queue,
                )
            ]
        if wait_for_attempt:
            self._wait_for_close_attempt(attempt, deadline)
            return

        errors: list[BaseException] = []
        if setup_cancel is not None:
            try:
                setup_cancel()
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        for reservation in reservations:
            reservation.task.cancel('Napari is shutting down')

        setup_future = self._setup_future
        setup_timed_out = False
        if setup_future is not None:
            try:
                setup_future.result(timeout=self._remaining(deadline))
            except FutureTimeoutError:
                setup_timed_out = True
                errors.append(
                    TimeoutError(
                        'Plugin environment setup did not stop before the '
                        'shutdown deadline'
                    )
                )
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        unfinished_tasks = 0
        if not setup_timed_out:
            for reservation in reservations:
                if reservation.task.done:
                    continue
                if not reservation.task._done.wait(self._remaining(deadline)):
                    unfinished_tasks += 1
            if unfinished_tasks:
                errors.append(
                    TimeoutError(
                        f'{unfinished_tasks} plugin worker tasks did not stop '
                        'before the shutdown deadline'
                    )
                )

        pool_futures: list[Future[None]] = []
        if not setup_timed_out and not unfinished_tasks:
            with self._lock:
                for environment_id, runtime in self._runtime.items():
                    pool = runtime.pool
                    if pool is not None:
                        pool_futures.append(
                            self._start_pool_close_locked(
                                environment_id, runtime, pool
                            )
                        )
            done, unfinished = wait(
                pool_futures, timeout=self._remaining(deadline)
            )
            errors.extend(
                error
                for future in done
                if (error := future.exception()) is not None
            )
            if unfinished:
                errors.append(
                    TimeoutError(
                        f'{len(unfinished)} plugin worker pools did not close '
                        'before the shutdown deadline'
                    )
                )

        with self._lock:
            pools_remain = any(
                runtime.pool is not None for runtime in self._runtime.values()
            )
        if not setup_timed_out and not unfinished_tasks and not pools_remain:
            backend = self._backend
            if backend is not None:
                with self._lock:
                    backend_future = self._backend_close_future
                    if backend_future is None or (
                        backend_future.done()
                        and backend_future.exception() is not None
                    ):
                        remaining = self._remaining(deadline)
                        backend_future = self._control_executor.submit(
                            backend.close, timeout=remaining
                        )
                        self._backend_close_future = backend_future
                try:
                    backend_future.result(timeout=self._remaining(deadline))
                except FutureTimeoutError:
                    errors.append(
                        TimeoutError(
                            'Plugin environment backend did not close before '
                            'the shutdown deadline'
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    errors.append(error)

        if errors:
            details = '\n'.join(str(error) for error in errors)
            close_error = PluginEnvironmentError(
                'Could not clean up every plugin worker', details=details
            )
            with self._lock:
                self._closing = False
                self._close_failed = True
            attempt.set_exception(close_error)
            self._log(
                f'Plugin worker cleanup failures:\n{details}', level='error'
            )
            raise close_error

        self._execution_executor.shutdown(wait=False, cancel_futures=True)
        self._setup_executor.shutdown(wait=False, cancel_futures=True)
        self._control_executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._closed = True
            self._closing = False
            self._close_failed = False
        attempt.set_result(None)


_manager: PluginEnvironmentManager | None = None
_manager_lock = threading.RLock()
_automatic_shutdown_started = False


def get_plugin_environment_manager() -> PluginEnvironmentManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PluginEnvironmentManager()
        return _manager


def _set_plugin_environment_manager(
    manager: PluginEnvironmentManager | None,
) -> None:
    global _manager
    with _manager_lock:
        _manager = manager


def shutdown_plugin_environments() -> None:
    with _manager_lock:
        manager = _manager
    if manager is not None:
        manager.close()


def _shutdown_plugin_environments_once() -> None:
    """Make at most one bounded automatic process-exit close attempt."""

    global _automatic_shutdown_started
    with _manager_lock:
        if _automatic_shutdown_started:
            return
        _automatic_shutdown_started = True
        manager = _manager
    if manager is not None:
        manager.close(timeout=_SHUTDOWN_TIMEOUT)


def _shutdown_plugin_environments_at_exit() -> None:
    try:
        _shutdown_plugin_environments_once()
    except Exception:
        logger.exception('Could not shut down managed plugin workers')


atexit.register(_shutdown_plugin_environments_at_exit)


__all__ = ('PluginEnvironmentManager',)

"""Private backend-neutral types for managed plugin workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class BackendPhase(Enum):
    """Private lifecycle phase reported by an environment backend."""

    PROVISIONING = 'provisioning'
    STARTING = 'starting'
    EXECUTING = 'executing'
    CLEANING_UP = 'cleaning_up'


@dataclass(frozen=True)
class EnvironmentRecipe:
    plugin: str
    plugin_version: str
    environment_id: str
    python: str
    conda: tuple[str, ...]
    pypi: tuple[str, ...]
    channels: tuple[str, ...]
    worker_package: Path | None
    worker_content_identity: str | None


@dataclass(frozen=True)
class WorkerCommand:
    plugin: str
    environment_id: str
    command_id: str
    target: str
    accepts_context: bool


@dataclass(frozen=True)
class BackendProgress:
    phase: BackendPhase
    message: str
    current: int | None = None
    total: int | None = None


ProgressCallback = Callable[[BackendProgress], None]
CancelCallbackSetter = Callable[[Callable[[], Any]], None]


class BackendCanceled(RuntimeError):
    """A private backend operation was canceled."""


class BackendFailure(RuntimeError):
    """A normalized failure from a managed environment backend."""

    def __init__(
        self,
        message: str,
        *,
        details: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.diagnostics = diagnostics


class BackendUnavailable(BackendFailure):
    """The configured backend cannot be imported or initialized."""


class BackendPool(Protocol):
    def execute(
        self,
        target: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        accepts_context: bool,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any: ...

    def close(self) -> None: ...


class EnvironmentBackend(Protocol):
    def worker_content_identity(self, source: Path) -> str: ...

    def provision_environment(
        self,
        name: str,
        recipe: EnvironmentRecipe,
        *,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
        on_mutation_started: Callable[[], None] | None = None,
    ) -> Any: ...

    def start_pool(
        self,
        environment: Any,
        *,
        progress: ProgressCallback,
    ) -> BackendPool: ...

    def remove_environment(
        self,
        name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None: ...

    def environment_names(self) -> tuple[str, ...]: ...

    def close(self, *, timeout: float | None = None) -> None: ...

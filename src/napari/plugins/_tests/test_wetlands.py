from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from napari.plugins._environment_types import BackendUnavailable
from napari.plugins._tests.test_environments import _recipe
from napari.plugins._wetlands import WetlandsBackend

if TYPE_CHECKING:
    from pathlib import Path


class _OperationCanceled(RuntimeError):
    pass


@dataclass
class _LocalPackage:
    source: Path
    editable: bool = False
    content_identity: str | None = None


class _EnvironmentSpec:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


class _Operation:
    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.listeners = []

    def cancel(self) -> bool:
        return True

    def listen(self, callback) -> None:
        self.listeners.append(callback)

    def wait_for(self) -> Any:
        return self.result


class _EnvironmentManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.provision_call = None
        self.close_timeout: float | None = None
        self.running_workers = ()

    def provision(self, name, spec, **kwargs):
        self.provision_call = (name, spec, kwargs)
        return _Operation(SimpleNamespace(name=name))

    def managed_environments(self):
        return ()

    def remove(self, name):
        return _Operation()

    def close(self, *, timeout: float | None = None) -> None:
        self.close_timeout = timeout


def _wetlands_module(version: str = '2.3.3') -> SimpleNamespace:
    return SimpleNamespace(
        __version__=version,
        EnvironmentManager=_EnvironmentManager,
        EnvironmentSpec=_EnvironmentSpec,
        LocalPackage=_LocalPackage,
        OperationCanceled=_OperationCanceled,
        local_package_content_identity=lambda path: f'sha256:{"a" * 64}',
    )


@pytest.mark.parametrize('version', ['2.3.2', '2.4.0', '3.0.0', 'invalid'])
def test_backend_rejects_unsupported_wetlands_version(
    tmp_path: Path, monkeypatch, version: str
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module(version))
    with pytest.raises(BackendUnavailable, match=r'Wetlands >=2\.3\.3,<2\.4'):
        WetlandsBackend(tmp_path)


def test_backend_rejects_missing_module_api_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    wetlands = _wetlands_module()
    del wetlands.local_package_content_identity
    monkeypatch.setitem(sys.modules, 'wetlands', wetlands)

    with pytest.raises(BackendUnavailable, match='missing required') as error:
        WetlandsBackend(tmp_path)

    assert error.value.details is not None
    assert 'local_package_content_identity' in error.value.details


def test_backend_rejects_missing_manager_api_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    class IncompleteEnvironmentManager(_EnvironmentManager):
        remove = None

    wetlands = _wetlands_module()
    wetlands.EnvironmentManager = IncompleteEnvironmentManager
    monkeypatch.setitem(sys.modules, 'wetlands', wetlands)

    with pytest.raises(BackendUnavailable, match='missing required') as error:
        WetlandsBackend(tmp_path)

    assert error.value.details is not None
    assert 'EnvironmentManager.remove' in error.value.details


def test_spec_uses_immutable_worker_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module())
    backend = WetlandsBackend(tmp_path)
    recipe = _recipe().__class__(
        **{
            **_recipe().__dict__,
            'worker_package': tmp_path,
            'worker_content_identity': f'sha256:{"b" * 64}',
        }
    )

    spec = backend._spec(recipe)

    assert spec.values['local'] == (
        _LocalPackage(
            tmp_path,
            editable=False,
            content_identity=f'sha256:{"b" * 64}',
        ),
    )


def test_provision_is_single_replace_aware_operation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module())
    backend = WetlandsBackend(tmp_path)

    def mutation() -> None:
        pass

    environment = backend.provision_environment(
        'example-plugin.worker',
        _recipe(),
        progress=lambda update: None,
        set_cancel_callback=lambda callback: None,
        on_mutation_started=mutation,
    )

    assert environment.name == 'example-plugin.worker'
    name, _spec, kwargs = backend._manager.provision_call
    assert name == 'example-plugin.worker'
    assert kwargs == {
        'replace_existing': True,
        'on_mutation_started': mutation,
    }


def test_worker_content_identity_uses_wetlands_helper(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module())
    assert WetlandsBackend(tmp_path).worker_content_identity(tmp_path) == (
        f'sha256:{"a" * 64}'
    )


def test_backend_forwards_shutdown_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, 'wetlands', _wetlands_module())
    backend = WetlandsBackend(tmp_path)

    backend.close(timeout=1.25)

    assert backend._manager.close_timeout == 1.25

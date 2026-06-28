"""Unit tests for HACS update helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_hacs_api():
    mod_name = "ha_agent.api.hacs"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.api" not in sys.modules:
        api_pkg = types.ModuleType("ha_agent.api")
        api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.api"] = api_pkg

    spec = importlib.util.spec_from_file_location(
        mod_name, COMPONENT / "api" / "hacs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRegistry:
    def __init__(self, entities: dict) -> None:
        self.entities = entities

    def async_get(self, entity_id: str):
        return self.entities.get(entity_id)


class _FakeHass:
    def __init__(self, states=None, data=None, registry_entities=None) -> None:
        self.states = states or {}
        self.data = data or {}
        self._registry_entities = registry_entities or {}

    class states:
        @staticmethod
        def get(entity_id):
            return None


def test_update_available_compares_versions():
    hacs_api = _load_hacs_api()
    assert hacs_api._update_available("1.9.0", "1.9.1") is True
    assert hacs_api._update_available("1.9.1", "1.9.1") is False
    assert hacs_api._update_available(None, "1.9.1") is True


def test_get_update_status_without_hacs():
    hacs_api = _load_hacs_api()
    hass = SimpleNamespace(data={}, states=SimpleNamespace(get=lambda _eid: None))
    status = hacs_api.get_update_status(hass)  # type: ignore[arg-type]
    assert status["hacs_available"] is False
    assert status["repository_found"] is False
    assert status["update_available"] is False

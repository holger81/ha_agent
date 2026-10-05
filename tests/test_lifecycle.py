"""Lifecycle helpers: entry task tracking and unload cleanup."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_helpers():
    if "ha_agent.api.helpers" in sys.modules:
        return sys.modules["ha_agent.api.helpers"]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.api" not in sys.modules:
        api_pkg = types.ModuleType("ha_agent.api")
        api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.api"] = api_pkg

    # Minimal stubs for helpers imports.
    if "homeassistant.core" not in sys.modules:
        ha = types.ModuleType("homeassistant.core")
        ha.HomeAssistant = object
        sys.modules["homeassistant.core"] = ha
    if "homeassistant.config_entries" not in sys.modules:
        ce = types.ModuleType("homeassistant.config_entries")
        ce.ConfigEntry = object
        sys.modules["homeassistant.config_entries"] = ce
    if "homeassistant.exceptions" not in sys.modules:
        hexc = types.ModuleType("homeassistant.exceptions")

        class HomeAssistantError(Exception):
            pass

        hexc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant.exceptions"] = hexc

    for name in ("const", "thinking", "config_helpers", "role_registry"):
        mod_name = f"ha_agent.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / f"{name}.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # role_registry may pull optional deps; helpers only needs const.
            if name != "const":
                sys.modules.pop(mod_name, None)

    # Stub config_helpers / role_registry if load failed.
    if "ha_agent.config_helpers" not in sys.modules:
        ch = types.ModuleType("ha_agent.config_helpers")
        ch.get_agent_config = lambda entry: None
        ch.get_llm_backend = lambda entry: None
        ch.get_router_config = lambda entry: None
        ch.get_skills_config = lambda entry: None
        sys.modules["ha_agent.config_helpers"] = ch
    if "ha_agent.role_registry" not in sys.modules:
        rr = types.ModuleType("ha_agent.role_registry")

        class ModelRole:
            pass

        rr.ModelRole = ModelRole
        rr.build_role_registry = lambda *a, **k: None
        rr.friendly_role_label = lambda *a, **k: ""
        sys.modules["ha_agent.role_registry"] = rr

    path = COMPONENT / "api" / "helpers.py"
    spec = importlib.util.spec_from_file_location("ha_agent.api.helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["ha_agent.api.helpers"] = module
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()
track_entry_task = helpers.track_entry_task
cancel_entry_tasks = helpers.cancel_entry_tasks
ENTRY_TASKS_KEY = helpers.ENTRY_TASKS_KEY
DATA_KEY = sys.modules["ha_agent.const"].DATA_KEY


@pytest.mark.asyncio
async def test_track_entry_task_registers_and_clears() -> None:
    hass = MagicMock()
    hass.data = {}

    def _create(coro, name=None):
        return asyncio.get_running_loop().create_task(coro, name=name)

    hass.async_create_task = _create

    async def _work():
        return 42

    task = track_entry_task(hass, "entry1", _work(), name="test")
    assert task in hass.data[DATA_KEY][ENTRY_TASKS_KEY]["entry1"]
    assert await task == 42
    await asyncio.sleep(0)
    assert "entry1" not in hass.data[DATA_KEY].get(ENTRY_TASKS_KEY, {})


@pytest.mark.asyncio
async def test_cancel_entry_tasks_cancels_pending() -> None:
    hass = MagicMock()
    hass.data = {}

    def _create(coro, name=None):
        return asyncio.get_running_loop().create_task(coro, name=name)

    hass.async_create_task = _create

    async def _hang():
        await asyncio.sleep(60)

    task = track_entry_task(hass, "entry1", _hang())
    cancel_entry_tasks(hass, "entry1")
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()

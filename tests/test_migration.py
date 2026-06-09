"""Unit tests for config entry migration."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_stubs() -> None:
    ha_pkg = sys.modules.get("homeassistant")
    if ha_pkg is None or not hasattr(ha_pkg, "__path__"):
        ha_pkg = types.ModuleType("homeassistant")
        ha_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["homeassistant"] = ha_pkg

    if "homeassistant.config_entries" not in sys.modules:
        ha_entries = types.ModuleType("homeassistant.config_entries")

        class ConfigEntry:
            def __init__(self, data: dict, version: int = 1) -> None:
                self.data = data
                self.version = version
                self.entry_id = "test-entry"

        ha_entries.ConfigEntry = ConfigEntry
        sys.modules["homeassistant.config_entries"] = ha_entries

    if "homeassistant.const" not in sys.modules:
        ha_const = types.ModuleType("homeassistant.const")
        ha_const.Platform = types.SimpleNamespace(
            CONVERSATION="conversation",
            SELECT="select",
            SWITCH="switch",
            SENSOR="sensor",
        )
        sys.modules["homeassistant.const"] = ha_const

    if "homeassistant.helpers" not in sys.modules:
        sys.modules["homeassistant.helpers"] = types.ModuleType("homeassistant.helpers")

    if "homeassistant.helpers.device_registry" not in sys.modules:
        ha_dr = types.ModuleType("homeassistant.helpers.device_registry")

        class DeviceRegistry:
            def async_get(self, _hass):
                return self

            def async_get_or_create(self, **_kwargs):
                return MagicMock()

        ha_dr.async_get = lambda _hass: DeviceRegistry()
        sys.modules["homeassistant.helpers.device_registry"] = ha_dr

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object
        sys.modules["homeassistant.core"] = ha_core


def _load_init_module():
    _ensure_ha_stubs()

    for mod in list(sys.modules):
        if mod == "ha_agent" or mod.startswith("ha_agent."):
            del sys.modules[mod]

    for name in ("const",):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    package = types.ModuleType("ha_agent")
    package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
    sys.modules["ha_agent"] = package

    path = COMPONENT / "__init__.py"
    spec = importlib.util.spec_from_file_location("ha_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_migrate_entry_resets_legacy_tool_instructions() -> None:
    """Legacy mcp_call_tool prompt text is replaced with MCP-compliant defaults."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 1
    entry.data = {
        const.CONF_TOOL_INSTRUCTIONS: "MCP tools via mcp_call_tool only.",
        const.CONF_AGENT_SYSTEM_PROMPT: ha_agent._LEGACY_AGENT_SYSTEM_PROMPT,
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    args, kwargs = hass.config_entries.async_update_entry.call_args
    assert args[0] is entry
    data = kwargs["data"]
    assert data[const.CONF_TOOL_INSTRUCTIONS] == const.DEFAULT_TOOL_INSTRUCTIONS
    assert data[const.CONF_AGENT_SYSTEM_PROMPT] == const.DEFAULT_AGENT_SYSTEM_PROMPT
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migrate_entry_adds_action_model_defaults() -> None:
    """Version 2 entries gain action-model routing defaults."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 2
    entry.data = {
        const.CONF_LLM_MODEL: "test-model",
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    data = kwargs["data"]
    assert data[const.CONF_ACTION_MODEL_ENABLED] is False
    assert data[const.CONF_ACTION_LLM_MODEL] == ""
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION

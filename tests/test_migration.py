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

        class ServiceCall:
            def __init__(self, data: dict | None = None) -> None:
                self.data = data or {}

        class SupportsResponse:
            ONLY = "only"

        ha_core.ServiceCall = ServiceCall
        ha_core.SupportsResponse = SupportsResponse

        def callback(func):
            return func

        ha_core.callback = callback
        sys.modules["homeassistant.core"] = ha_core

    if "homeassistant.components" not in sys.modules:
        components = types.ModuleType("homeassistant.components")
        components.__path__ = []  # type: ignore[attr-defined]
        sys.modules["homeassistant.components"] = components

    if "homeassistant.components.panel_custom" not in sys.modules:
        panel_custom = types.ModuleType("homeassistant.components.panel_custom")

        async def _async_register_panel(*_args, **_kwargs) -> None:
            return None

        panel_custom.async_register_panel = _async_register_panel
        sys.modules["homeassistant.components.panel_custom"] = panel_custom

    if "homeassistant.components.http" not in sys.modules:
        http = types.ModuleType("homeassistant.components.http")
        http.StaticPathConfig = object
        sys.modules["homeassistant.components.http"] = http

    if "homeassistant.components.websocket_api" not in sys.modules:
        ws_api = types.ModuleType("homeassistant.components.websocket_api")
        ws_api.async_register_command = lambda *_args, **_kwargs: None
        sys.modules["homeassistant.components.websocket_api"] = ws_api

    if "homeassistant.helpers.storage" not in sys.modules:
        storage = types.ModuleType("homeassistant.helpers.storage")

        class Store:
            def __init__(self, *_args, **_kwargs) -> None:
                self.data = {}

            async def async_load(self) -> dict:
                return self.data

            async def async_save(self) -> None:
                return None

        storage.Store = Store
        sys.modules["homeassistant.helpers.storage"] = storage

    if "homeassistant.helpers.config_validation" not in sys.modules:
        ha_cv = types.ModuleType("homeassistant.helpers.config_validation")

        def config_entry_only_config_schema(_domain: str):
            return lambda _: {}

        ha_cv.config_entry_only_config_schema = config_entry_only_config_schema
        sys.modules["homeassistant.helpers.config_validation"] = ha_cv


def _load_init_module():
    _ensure_ha_stubs()

    ha_core = sys.modules.get("homeassistant.core")
    if ha_core is not None and not hasattr(ha_core, "SupportsResponse"):

        class SupportsResponse:
            ONLY = "only"

        ha_core.SupportsResponse = SupportsResponse

    ha_core = sys.modules.get("homeassistant.core")
    if ha_core is not None and not hasattr(ha_core, "ServiceCall"):

        class ServiceCall:
            def __init__(self, data: dict | None = None) -> None:
                self.data = data or {}

        ha_core.ServiceCall = ServiceCall

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

    panel_stub = types.ModuleType("ha_agent.panel")

    async def _register_panel(_hass) -> None:
        return None

    panel_stub.async_register_panel = _register_panel
    sys.modules["ha_agent.panel"] = panel_stub

    ws_stub = types.ModuleType("ha_agent.websocket_api")
    ws_stub.async_register_handlers = lambda _hass: None
    sys.modules["ha_agent.websocket_api"] = ws_stub

    for name in ("thinking",):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

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
    assert data[const.CONF_SKILLS_LEARNING_ENABLED] is False
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migrate_entry_adds_skills_defaults() -> None:
    """Version 3 entries gain skills defaults."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 3
    entry.data = {
        const.CONF_LLM_MODEL: "test-model",
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    data = kwargs["data"]
    assert data[const.CONF_SKILLS_LEARNING_ENABLED] is False
    assert data[const.CONF_SKILLS_USE_ENABLED] is True
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migrate_entry_converts_thinking_level() -> None:
    """Version 4 entries convert legacy enable_thinking to thinking_level."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 4
    entry.data = {
        const.CONF_LLM_MODEL: "test-model",
        const.CONF_LLM_ENABLE_THINKING: True,
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    data = kwargs["data"]
    assert data[const.CONF_LLM_THINKING_LEVEL] == "medium"
    assert const.CONF_LLM_ENABLE_THINKING not in data
    assert data[const.CONF_CONVERSATION_SHOW_REASONING] is True
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migrate_entry_adds_show_reasoning_default() -> None:
    """Version 5 entries gain show-reasoning default."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 5
    entry.data = {
        const.CONF_LLM_MODEL: "test-model",
        const.CONF_LLM_THINKING_LEVEL: "medium",
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    data = kwargs["data"]
    assert data[const.CONF_CONVERSATION_SHOW_REASONING] is True
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


@pytest.mark.asyncio
async def test_migrate_entry_adds_memory_persist_default() -> None:
    """Version 6 entries gain conversation memory persistence default."""
    ha_agent = _load_init_module()
    const = sys.modules["ha_agent.const"]

    entry = MagicMock()
    entry.version = 6
    entry.data = {
        const.CONF_LLM_MODEL: "test-model",
        const.CONF_CONVERSATION_SHOW_REASONING: True,
    }

    hass = MagicMock()
    await ha_agent.async_migrate_entry(hass, entry)

    hass.config_entries.async_update_entry.assert_called_once()
    _args, kwargs = hass.config_entries.async_update_entry.call_args
    data = kwargs["data"]
    assert data[const.CONF_CONVERSATION_MEMORY_PERSIST] is False
    assert kwargs["version"] == const.CONFIG_ENTRY_VERSION


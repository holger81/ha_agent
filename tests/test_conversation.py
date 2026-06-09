"""Unit tests for conversation platform helpers."""

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


def _load_conversation_module():
    module_name = "ha_agent.conversation"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    ha_pkg = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")
    ha_exc = types.ModuleType("homeassistant.exceptions")
    ha_helpers_intent = types.ModuleType("homeassistant.helpers.intent")
    ha_helpers_er = types.ModuleType("homeassistant.helpers.entity_registry")
    ha_helpers_ar = types.ModuleType("homeassistant.helpers.area_registry")
    ha_helpers_aiohttp = types.ModuleType("homeassistant.helpers.aiohttp_client")
    ha_helpers_entity_platform = types.ModuleType(
        "homeassistant.helpers.entity_platform"
    )
    ha_helpers_entity = types.ModuleType("homeassistant.helpers.entity")

    class DeviceInfo(dict):
        pass

    ha_helpers_entity.DeviceInfo = DeviceInfo
    ha_components = types.ModuleType("homeassistant.components")
    ha_conversation = types.ModuleType("homeassistant.components.conversation")
    ha_exposed = types.ModuleType(
        "homeassistant.components.homeassistant.exposed_entities"
    )
    ha_config_entries = types.ModuleType("homeassistant.config_entries")

    class HomeAssistantError(Exception):
        pass

    class IntentResponse:
        def __init__(self, *, language: str) -> None:
            self.language = language
            self.speech: dict = {}

        def async_set_speech(self, text: str) -> None:
            self.speech = {"plain": {"speech": text}}

    def callback(func):
        return func

    ha_core.HomeAssistant = object
    ha_core.callback = callback
    ha_exc.HomeAssistantError = HomeAssistantError
    ha_helpers_intent.IntentResponse = IntentResponse
    ha_helpers_er.async_get = MagicMock()
    ha_helpers_ar.async_get = MagicMock()
    ha_helpers_aiohttp.async_get_clientsession = MagicMock()
    ha_helpers_entity_platform.AddEntitiesCallback = object
    class AbstractConversationAgent:
        pass

    class ConversationInput:
        pass

    class ChatLog:
        pass

    class ConversationResult:
        pass

    class ConversationEntityFeature:
        CONTROL = 1

    ha_conversation.ConversationEntityFeature = ConversationEntityFeature
    class _ConversationEntityBase:
        async def async_process(self, user_input):
            raise NotImplementedError

    class ConversationEntity(_ConversationEntityBase):
        pass

    ha_conversation.ConversationEntity = ConversationEntity
    ha_conversation.AbstractConversationAgent = AbstractConversationAgent
    ha_conversation.ConversationInput = ConversationInput
    ha_conversation.ChatLog = ChatLog
    ha_conversation.ConversationResult = ConversationResult
    ha_conversation.async_get_result_from_chat_log = MagicMock()
    ha_exposed.async_should_expose = MagicMock(return_value=True)
    ha_config_entries.ConfigEntry = object
    ha_chat_log = types.ModuleType("homeassistant.components.conversation.chat_log")
    ha_chat_log.current_chat_log = types.SimpleNamespace(get=lambda: None)

    sys.modules["homeassistant"] = ha_pkg
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.exceptions"] = ha_exc
    sys.modules["homeassistant.helpers"] = types.ModuleType("homeassistant.helpers")
    sys.modules["homeassistant.helpers.intent"] = ha_helpers_intent
    sys.modules["homeassistant.helpers.entity_registry"] = ha_helpers_er
    sys.modules["homeassistant.helpers.area_registry"] = ha_helpers_ar
    sys.modules["homeassistant.helpers.aiohttp_client"] = ha_helpers_aiohttp
    sys.modules["homeassistant.helpers.entity_platform"] = ha_helpers_entity_platform
    sys.modules["homeassistant.helpers.entity"] = ha_helpers_entity
    sys.modules["homeassistant.components"] = ha_components
    sys.modules["homeassistant.components.conversation"] = ha_conversation
    sys.modules["homeassistant.components.conversation.chat_log"] = ha_chat_log
    sys.modules["homeassistant.components.homeassistant"] = types.ModuleType(
        "homeassistant.components.homeassistant"
    )
    sys.modules["homeassistant.components.homeassistant.exposed_entities"] = (
        ha_exposed
    )
    sys.modules["homeassistant.config_entries"] = ha_config_entries

    for dep in (
        "const",
        "config_helpers",
        "llm_client",
        "mcp_client",
        "context",
        "tools",
        "memory",
        "agent",
    ):
        dep_name = f"ha_agent.{dep}"
        if dep_name not in sys.modules:
            path = COMPONENT / f"{dep}.py"
            spec = importlib.util.spec_from_file_location(dep_name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[dep_name] = module
            spec.loader.exec_module(module)

    path = COMPONENT / "conversation.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


conversation_mod = _load_conversation_module()


def test_supports_streaming_follows_agent_config() -> None:
    """Assist pipeline reads supports_streaming from the conversation entity."""
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "HA Agent"
    entry.data = {"llm_model": "test-model", "conversation_enable_streaming": True}

    entity = conversation_mod.HaAgentConversationEntity(MagicMock(), entry)
    assert entity.supports_streaming is True

    entry.data["conversation_enable_streaming"] = False
    assert entity.supports_streaming is False


@pytest.mark.asyncio
async def test_collect_exposed_entities_uses_entity_registry_get() -> None:
    """Exposed entities are read via er.async_get(hass).entities.values()."""
    entry = MagicMock()
    entry.entity_id = "light.kitchen"
    entry.name = "Kitchen"
    entry.area_id = None
    entry.aliases = set()

    entity_registry = MagicMock()
    entity_registry.entities.values.return_value = [entry]

    state = MagicMock()
    state.name = "Kitchen"
    state.state = "on"

    hass = MagicMock()
    hass.states.get.return_value = state

    import homeassistant.helpers.entity_registry as er

    er.async_get.return_value = entity_registry

    exposed = await conversation_mod.collect_exposed_entities(hass)

    er.async_get.assert_called_once_with(hass)
    assert exposed == [
        {
            "entity_id": "light.kitchen",
            "name": "Kitchen",
            "state": "on",
            "area_name": None,
            "aliases": [],
        }
    ]

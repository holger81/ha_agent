"""HA Agent conversation entity for Home Assistant Assist."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .agent import run_agent
from .config_helpers import get_agent_config, get_llm_backend, get_mcp_config
from .const import DOMAIN, LOGGER, SUPPORTED_LANGUAGES
from .context import user_text_from_input
from .llm_client import LlmClient
from .mcp_client import McpProxyClient


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Agent conversation entity from a config entry."""
    async_add_entities([HaAgentConversationEntity(hass, config_entry)])


def collect_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Collect entities exposed to Assist for tool hints."""
    area_registry = ar.async_get(hass)
    exposed: list[dict[str, Any]] = []

    for entry in er.async_entries(hass):
        if not async_should_expose(hass, conversation.DOMAIN, entry.entity_id):
            continue
        state = hass.states.get(entry.entity_id)
        area_name = None
        if entry.area_id and (area := area_registry.async_get_area(entry.area_id)):
            area_name = area.name
        exposed.append(
            {
                "entity_id": entry.entity_id,
                "name": entry.name or (state.name if state else entry.entity_id),
                "state": state.state if state else None,
                "area_name": area_name,
                "aliases": list(entry.aliases) if entry.aliases else [],
            }
        )

    return exposed


class HaAgentConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """HA Agent voice conversation entity."""

    _attr_has_entity_name = True
    _attr_name = "HA Agent"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the conversation entity."""
        self._entry = entry
        session = async_get_clientsession(hass)
        self._llm = LlmClient(session)
        self._mcp = McpProxyClient(session, get_mcp_config(entry))
        self._attr_unique_id = f"{entry.entry_id}_conversation"
        agent_config = get_agent_config(entry)
        self._attr_supports_streaming = agent_config.enable_streaming

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return list(SUPPORTED_LANGUAGES)

    async def async_added_to_hass(self) -> None:
        """Register as the conversation agent for this config entry."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister the conversation agent."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process user input through the agent loop."""
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                None,
                None,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        user_text = user_text_from_input(user_input)
        intent_response = intent.IntentResponse(language=user_input.language)

        if not user_text:
            intent_response.async_set_speech("I didn't catch that.")
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
            )

        backend = get_llm_backend(self._entry)
        agent_config = get_agent_config(self._entry)
        exposed = collect_exposed_entities(self.hass)

        async def delta_stream() -> AsyncGenerator[dict[str, Any], None]:
            yield {"role": "assistant"}
            async for chunk in run_agent(
                self.hass,
                llm=self._llm,
                mcp_client=self._mcp,
                backend=backend,
                agent_config=agent_config,
                conversation_id=chat_log.conversation_id,
                user_text=user_text,
                exposed_entities=exposed,
                extra_system_prompt=user_input.extra_system_prompt,
            ):
                yield {"content": chunk}

        try:
            async for _content in chat_log.async_add_delta_content_stream(
                user_input.agent_id,
                delta_stream(),
            ):
                pass
        except Exception as err:
            LOGGER.exception("HA Agent failed: %s", err)
            intent_response.async_set_speech(
                "Sorry, something went wrong while processing your request."
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
            )

        speech = ""
        if chat_log.content:
            speech = chat_log.content[-1].content or ""

        intent_response.async_set_speech(speech or "Done.")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

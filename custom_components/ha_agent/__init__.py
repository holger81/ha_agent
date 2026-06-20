"""HA Agent integration for Home Assistant."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr

from .const import (
    CONF_ACTION_LLM_BASE_URL,
    CONF_ACTION_LLM_MAX_TOKENS,
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_LLM_TEMPERATURE,
    CONF_ACTION_MODEL_ENABLED,
    CONF_AGENT_SYSTEM_PROMPT,
    CONF_CONVERSATION_MEMORY_PERSIST,
    CONF_CONVERSATION_SHOW_REASONING,
    CONF_LLM_ENABLE_THINKING,
    CONF_LLM_MODEL,
    CONF_LLM_THINKING_LEVEL,
    CONF_SKILLS_AUTO_SAVE,
    CONF_SKILLS_LEARNING_ENABLED,
    CONF_SKILLS_MAX_INJECT,
    CONF_SKILLS_USE_ENABLED,
    CONF_TOOL_INSTRUCTIONS,
    CONFIG_ENTRY_VERSION,
    DEFAULT_ACTION_LLM_MAX_TOKENS,
    DEFAULT_ACTION_LLM_TEMPERATURE,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_SKILLS_MAX_INJECT,
    DEFAULT_TOOL_INSTRUCTIONS,
    DOMAIN,
    LEGACY_TOOL_INSTRUCTION_MARKERS,
)
from .memory import async_load_memory
from .panel import async_register_panel
from .playbooks import close_playbook_store, get_playbook_store
from .skills.commands import async_setup_services
from .skills.store import close_skill_store, get_skill_store
from .thinking import normalize_thinking_level
from .threads import async_load_threads
from .websocket_api import async_register_handlers

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS: list[Platform] = [
    Platform.CONVERSATION,
    Platform.SELECT,
    Platform.SWITCH,
    Platform.SENSOR,
]

_LEGACY_AGENT_SYSTEM_PROMPT = (
    "You are a voice assistant for Home Assistant.\n"
    "Answer questions truthfully in plain text. Keep replies concise for speech.\n"
    "When the user asks you to perform an action, ALWAYS use a tool.\n"
    "When asked for news, use the MCP news tool — never invent headlines."
)


def _is_legacy_tool_instructions(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LEGACY_TOOL_INSTRUCTION_MARKERS)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate config entries to current schema."""
    version = config_entry.version
    data = dict(config_entry.data)

    if version == 1:
        if _is_legacy_tool_instructions(data.get(CONF_TOOL_INSTRUCTIONS, "")):
            data[CONF_TOOL_INSTRUCTIONS] = DEFAULT_TOOL_INSTRUCTIONS
        if data.get(CONF_AGENT_SYSTEM_PROMPT) == _LEGACY_AGENT_SYSTEM_PROMPT:
            data[CONF_AGENT_SYSTEM_PROMPT] = DEFAULT_AGENT_SYSTEM_PROMPT
        version = 2

    if version == 2:
        data.setdefault(CONF_ACTION_MODEL_ENABLED, False)
        data.setdefault(CONF_ACTION_LLM_BASE_URL, "")
        data.setdefault(CONF_ACTION_LLM_MODEL, "")
        data.setdefault(
            CONF_ACTION_LLM_TEMPERATURE,
            DEFAULT_ACTION_LLM_TEMPERATURE,
        )
        data.setdefault(CONF_ACTION_LLM_MAX_TOKENS, DEFAULT_ACTION_LLM_MAX_TOKENS)
        version = 3

    if version == 3:
        data.setdefault(CONF_SKILLS_LEARNING_ENABLED, False)
        data.setdefault(CONF_SKILLS_AUTO_SAVE, False)
        data.setdefault(CONF_SKILLS_USE_ENABLED, True)
        data.setdefault(CONF_SKILLS_MAX_INJECT, DEFAULT_SKILLS_MAX_INJECT)
        version = 4

    if version == 4:
        if CONF_LLM_THINKING_LEVEL not in data:
            data[CONF_LLM_THINKING_LEVEL] = normalize_thinking_level(
                data.get(CONF_LLM_ENABLE_THINKING)
            )
        data.pop(CONF_LLM_ENABLE_THINKING, None)
        version = 5

    if version == 5:
        data.setdefault(CONF_CONVERSATION_SHOW_REASONING, True)
        version = 6

    if version == 6:
        data.setdefault(CONF_CONVERSATION_MEMORY_PERSIST, False)
        version = CONFIG_ENTRY_VERSION

    if version != config_entry.version:
        hass.config_entries.async_update_entry(
            config_entry,
            data=data,
            version=version,
        )

    return True


async def async_setup(hass: HomeAssistant, _config: dict[str, Any]) -> bool:
    """Set up HA Agent global handlers."""
    async_register_handlers(hass)
    await async_register_panel(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Agent from the config entry."""
    device_registry = dr.async_get(hass)
    device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={(DOMAIN, entry.entry_id)},
        manufacturer="HA Agent",
        name=entry.title,
        model=entry.data.get(CONF_LLM_MODEL),
    )
    get_skill_store(hass, entry.entry_id)
    get_playbook_store(hass, entry.entry_id)
    await async_setup_services(hass)
    await async_load_memory(hass, entry.entry_id)
    await async_load_threads(hass, entry.entry_id)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    close_skill_store(hass, entry.entry_id)
    close_playbook_store(hass, entry.entry_id)
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload integration when config changes."""
    await hass.config_entries.async_reload(entry.entry_id)

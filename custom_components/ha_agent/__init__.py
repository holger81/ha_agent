"""HA Agent integration for Home Assistant."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from .const import (
    CONF_ACTION_LLM_BASE_URL,
    CONF_ACTION_LLM_MAX_TOKENS,
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_LLM_TEMPERATURE,
    CONF_ACTION_MODEL_ENABLED,
    CONF_AGENT_SYSTEM_PROMPT,
    CONF_LLM_MODEL,
    CONF_TOOL_INSTRUCTIONS,
    CONFIG_ENTRY_VERSION,
    DEFAULT_ACTION_LLM_MAX_TOKENS,
    DEFAULT_ACTION_LLM_TEMPERATURE,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_TOOL_INSTRUCTIONS,
    DOMAIN,
    LEGACY_TOOL_INSTRUCTION_MARKERS,
)

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
        version = CONFIG_ENTRY_VERSION

    if version != config_entry.version:
        hass.config_entries.async_update_entry(
            config_entry,
            data=data,
            version=version,
        )

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
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload integration when config changes."""
    await hass.config_entries.async_reload(entry.entry_id)

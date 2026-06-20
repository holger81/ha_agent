"""Shared helpers for the HA Agent console API."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..config_helpers import (
    get_agent_config,
    get_llm_backend,
    get_router_config,
    get_skills_config,
)
from ..const import DOMAIN


def require_admin(connection) -> None:
    """Raise if the connection user is not an admin."""
    if not connection.user.is_admin:
        raise HomeAssistantError("Admin access required")


def list_entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Return all HA Agent config entries."""
    return list(hass.config_entries.async_entries(DOMAIN))


def entry_summaries(hass: HomeAssistant) -> list[dict[str, str]]:
    """Return id/title pairs for all config entries."""
    return [
        {"entry_id": entry.entry_id, "title": entry.title}
        for entry in list_entries(hass)
    ]


def get_entry(hass: HomeAssistant, entry_id: str) -> ConfigEntry:
    """Return a config entry or raise."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise HomeAssistantError(f"Unknown HA Agent entry: {entry_id}")
    return entry


def config_snapshot(hass: HomeAssistant, entry: ConfigEntry) -> dict:
    """Return a JSON-safe config snapshot for the console."""
    data = entry.data
    agent = get_agent_config(entry)
    skills = get_skills_config(entry)
    router = get_router_config(entry)
    llm = get_llm_backend(entry)
    return {
        "entry_id": entry.entry_id,
        "title": entry.title,
        "llm_model": llm.model,
        "llm_base_url": llm.base_url,
        "llm_timeout": llm.timeout,
        "thinking_level": llm.thinking_level,
        "action_model_enabled": router.action_enabled,
        "action_model": (
            router.action_backend.model if router.action_backend else None
        ),
        "classifier_model_enabled": router.classifier_backend is not None,
        "classifier_model": (
            router.classifier_backend.model if router.classifier_backend else None
        ),
        "classifier_llm_base_url": data.get("classifier_llm_base_url", ""),
        "max_iterations": agent.max_iterations,
        "history_turns": agent.history_turns,
        "enable_streaming": agent.enable_streaming,
        "show_reasoning_in_chat": agent.show_reasoning_in_chat,
        "skills_learning_enabled": skills.learning_enabled,
        "skills_auto_save": skills.auto_save,
        "skills_use_enabled": skills.use_enabled,
        "skills_max_inject": skills.max_inject,
        "memory_persist": bool(data.get("conversation_memory_persist", False)),
    }

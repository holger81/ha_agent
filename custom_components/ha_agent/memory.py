"""Per-conversation history for multi-turn Assist and console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, callback

from .const import CONF_CONVERSATION_MEMORY_PERSIST, DATA_KEY, DOMAIN, LOGGER

MEMORY_KEY = "conversation_memory"


def _memory_file(hass: HomeAssistant, entry_id: str) -> Path:
    return Path(hass.config.path(".storage")) / f"{DOMAIN}_memory_{entry_id}.json"


@callback
def _memory_store(hass: HomeAssistant) -> dict[str, list[dict[str, Any]]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(MEMORY_KEY, {})


@callback
def get_history(
    hass: HomeAssistant,
    conversation_id: str | None,
    *,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Return stored history for a conversation."""
    if not conversation_id or max_turns <= 0:
        return []
    store = _memory_store(hass)
    history = store.get(conversation_id, [])
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return list(history)


@callback
def conversation_history_for_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    *,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Return prior turns only, excluding the in-flight user message for this turn."""
    history = get_history(
        hass,
        conversation_id,
        max_turns=max_turns,
    )
    if not history:
        return []
    last = history[-1]
    if (
        last.get("role") == "user"
        and str(last.get("content", "")).strip() == user_text.strip()
    ):
        return list(history[:-1])
    return history


@callback
def append_user_message(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    *,
    max_turns: int,
    entry_id: str | None = None,
) -> None:
    """Append only the user side of a turn (for immediate history visibility)."""
    if not conversation_id or max_turns <= 0 or not user_text.strip():
        return

    store = _memory_store(hass)
    history = store.setdefault(conversation_id, [])
    last = history[-1] if history else None
    if last and last.get("role") == "user" and last.get("content") == user_text.strip():
        return
    history.append({"role": "user", "content": user_text.strip()})

    max_messages = max_turns * 2
    if len(history) > max_messages:
        store[conversation_id] = history[-max_messages:]

    if entry_id:
        _maybe_persist(hass, entry_id)


@callback
def append_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    assistant_text: str,
    *,
    max_turns: int,
    entry_id: str | None = None,
    turn_meta: dict[str, Any] | None = None,
) -> None:
    """Append a user/assistant turn to memory."""
    if not conversation_id or max_turns <= 0:
        return
    if not user_text.strip() and not assistant_text.strip():
        return

    store = _memory_store(hass)
    history = store.setdefault(conversation_id, [])
    if user_text.strip():
        last = history[-1] if history else None
        if not (
            last
            and last.get("role") == "user"
            and last.get("content") == user_text.strip()
        ):
            history.append({"role": "user", "content": user_text.strip()})
    if assistant_text.strip():
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text.strip(),
        }
        if turn_meta:
            assistant_entry["turn_meta"] = turn_meta
        history.append(assistant_entry)

    max_messages = max_turns * 2
    if len(history) > max_messages:
        store[conversation_id] = history[-max_messages:]

    if entry_id:
        _maybe_persist(hass, entry_id)


@callback
def clear_conversation(hass: HomeAssistant, conversation_id: str | None) -> None:
    """Clear stored history for a conversation."""
    if not conversation_id:
        return
    store = _memory_store(hass)
    store.pop(conversation_id, None)


def _entry_wants_persist(hass: HomeAssistant, entry_id: str) -> bool:
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None:
        return False
    return bool(entry.data.get(CONF_CONVERSATION_MEMORY_PERSIST, False))


def _maybe_persist(hass: HomeAssistant, entry_id: str) -> None:
    if _entry_wants_persist(hass, entry_id):
        hass.async_create_task(async_save_memory(hass, entry_id))


async def async_load_memory(hass: HomeAssistant, entry_id: str) -> None:
    """Load persisted conversation memory from disk."""
    path = _memory_file(hass, entry_id)
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        LOGGER.warning("Failed to load HA Agent memory for %s: %s", entry_id, err)
        return
    if not isinstance(data, dict):
        return
    store = _memory_store(hass)
    for conversation_id, messages in data.items():
        if isinstance(messages, list):
            store[conversation_id] = messages


async def async_save_memory(hass: HomeAssistant, entry_id: str) -> None:
    """Persist in-memory conversations to disk."""
    store = _memory_store(hass)
    path = _memory_file(hass, entry_id)
    try:
        path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    except OSError as err:
        LOGGER.warning("Failed to save HA Agent memory for %s: %s", entry_id, err)

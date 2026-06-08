"""Per-conversation history for multi-turn Assist."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, callback

from .const import DATA_KEY

MEMORY_KEY = "conversation_memory"


@callback
def _memory_store(hass: HomeAssistant) -> dict[str, list[dict[str, str]]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(MEMORY_KEY, {})


@callback
def get_history(
    hass: HomeAssistant,
    conversation_id: str | None,
    *,
    max_turns: int,
) -> list[dict[str, str]]:
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
def append_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    assistant_text: str,
    *,
    max_turns: int,
) -> None:
    """Append a user/assistant turn to memory."""
    if not conversation_id or max_turns <= 0:
        return
    if not user_text.strip() and not assistant_text.strip():
        return

    store = _memory_store(hass)
    history = store.setdefault(conversation_id, [])
    if user_text.strip():
        history.append({"role": "user", "content": user_text.strip()})
    if assistant_text.strip():
        history.append({"role": "assistant", "content": assistant_text.strip()})

    max_messages = max_turns * 2
    if len(history) > max_messages:
        store[conversation_id] = history[-max_messages:]


@callback
def clear_conversation(hass: HomeAssistant, conversation_id: str | None) -> None:
    """Clear stored history for a conversation."""
    if not conversation_id:
        return
    store = _memory_store(hass)
    store.pop(conversation_id, None)

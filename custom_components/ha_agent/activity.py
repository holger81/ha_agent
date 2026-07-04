"""Turn activity log for the HA Agent console."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from homeassistant.core import HomeAssistant, callback

from .api.serialize import turn_trace_to_dict
from .const import DATA_KEY
from .skills.models import TurnTrace

ACTIVITY_KEY = "activity"
DEFAULT_MAX_TURNS = 100


@callback
def _activity_store(hass: HomeAssistant) -> dict[str, deque[dict[str, Any]]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(ACTIVITY_KEY, {})


@callback
def record_turn(
    hass: HomeAssistant,
    entry_id: str,
    trace: TurnTrace,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    """Append a turn trace to the per-entry activity ring buffer."""
    store = _activity_store(hass)
    buffer = store.get(entry_id)
    if buffer is None:
        buffer = deque(maxlen=max_turns)
        store[entry_id] = buffer
    item = turn_trace_to_dict(trace, timestamp=time.time())
    buffer.append(item)
    hass.bus.async_fire(
        "ha_agent_turn_recorded",
        {
            "entry_id": entry_id,
            "conversation_id": trace.conversation_id,
            "timestamp": item.get("timestamp"),
            "turn": item,
        },
    )
    return item


@callback
def get_turn(
    hass: HomeAssistant,
    entry_id: str,
    *,
    timestamp: float | None = None,
    conversation_id: str | None = None,
    latest: bool = False,
) -> dict[str, Any] | None:
    """Return one activity turn by timestamp or latest for a conversation."""
    turns, _total = list_turns(hass, entry_id, limit=DEFAULT_MAX_TURNS)
    if latest and conversation_id:
        for turn in turns:
            if turn.get("conversation_id") == conversation_id:
                return turn
        return None
    if timestamp is not None:
        target = float(timestamp)
        for turn in turns:
            raw = turn.get("timestamp")
            if raw is not None and abs(float(raw) - target) < 0.001:
                return turn
        return None
    if conversation_id:
        for turn in turns:
            if turn.get("conversation_id") == conversation_id:
                return turn
    return turns[0] if turns and latest else None


@callback
def update_turn_identity(
    hass: HomeAssistant,
    entry_id: str,
    timestamp: float,
    *,
    agent_user_id: str,
    agent_user_display_name: str,
    agent_user_kind: str,
    corrected_by_ha_user_id: str | None,
    original_user_id: str,
    original_display_name: str,
) -> dict[str, Any] | None:
    """Patch identity fields on one activity turn."""
    store = _activity_store(hass)
    buffer = store.get(entry_id)
    if not buffer:
        return None

    target = float(timestamp)
    for turn in buffer:
        raw = turn.get("timestamp")
        if raw is None or abs(float(raw) - target) >= 0.001:
            continue
        if not turn.get("identity_original_user_id"):
            turn["identity_original_user_id"] = original_user_id
            turn["identity_original_display_name"] = original_display_name
        turn["agent_user_id"] = agent_user_id
        turn["agent_user_display_name"] = agent_user_display_name
        turn["agent_user_kind"] = agent_user_kind
        turn["identity_source"] = "corrected"
        turn["identity_corrected_by_ha_user_id"] = corrected_by_ha_user_id
        hass.bus.async_fire(
            "ha_agent_turn_updated",
            {
                "entry_id": entry_id,
                "timestamp": turn.get("timestamp"),
                "turn": dict(turn),
            },
        )
        return turn
    return None


@callback
def list_turns(
    hass: HomeAssistant,
    entry_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return paginated activity turns newest-first."""
    store = _activity_store(hass)
    buffer = store.get(entry_id)
    if not buffer:
        return [], 0
    items = list(reversed(buffer))
    total = len(items)
    return items[offset : offset + limit], total

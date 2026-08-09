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


def activity_turn_to_trace(data: dict[str, Any]) -> TurnTrace:
    """Rebuild a TurnTrace from an activity log row."""
    followed = data.get("skill_followed")
    return TurnTrace(
        user_text=str(data.get("user_text") or ""),
        history_len=int(data.get("history_len") or 0),
        tool_calls=list(data.get("tool_calls") or []),
        tool_errors=int(data.get("tool_errors") or 0),
        iterations=int(data.get("iterations") or 0),
        fallback=bool(data.get("fallback")),
        assistant_text=str(data.get("assistant_text") or ""),
        matched_skill_ids=list(data.get("matched_skill_ids") or []),
        controlled_entity_ids=list(data.get("controlled_entity_ids") or []),
        conversation_id=data.get("conversation_id"),
        outcome=str(data.get("outcome") or ""),
        verification_notes=list(data.get("verification_notes") or []),
        route=str(data.get("route") or ""),
        exposed_entities=list(data.get("exposed_entities") or []),
        complexity=str(data.get("complexity") or "simple"),
        verifier_verdict=str(data.get("verifier_verdict") or ""),
        verifier_detail=str(data.get("verifier_detail") or ""),
        matched_learned_skill_ids=list(data.get("matched_learned_skill_ids") or []),
        skill_followed=followed if isinstance(followed, bool) else None,
        skill_plan_override=bool(data.get("skill_plan_override")),
        skill_plan_override_reason=str(data.get("skill_plan_override_reason") or ""),
        recovery_hints=list(data.get("recovery_hints") or []),
        llm_calls=list(data.get("llm_calls") or []),
        plan_progress=list(data.get("plan_progress") or []),
    )


@callback
def find_prior_workflow_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str | None,
    *,
    skip_user_texts: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Return the newest prior turn in this conversation with tool activity.

    Skips turns whose user text is in ``skip_user_texts`` (e.g. the current
    "save this as a skill" request when it was already recorded).
    """
    turns, _total = list_turns(hass, entry_id, limit=DEFAULT_MAX_TURNS)
    skip = {text.strip().lower() for text in (skip_user_texts or frozenset()) if text}
    for turn in turns:
        if conversation_id and turn.get("conversation_id") != conversation_id:
            continue
        user = str(turn.get("user_text") or "").strip()
        if user.lower() in skip:
            continue
        tools = turn.get("tool_calls") or []
        if not isinstance(tools, list) or not tools:
            continue
        if any(
            isinstance(call, dict) and call.get("succeeded") is not False
            for call in tools
        ):
            return turn
    return None


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

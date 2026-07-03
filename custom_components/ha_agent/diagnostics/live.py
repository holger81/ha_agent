"""In-memory live chat observation buffers for external agents."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from homeassistant.core import HomeAssistant, callback

from ..const import DATA_KEY

LIVE_KEY = "diagnostics_live"
MAX_DELTAS_PER_TURN = 200


@callback
def _live_store(
    hass: HomeAssistant,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(LIVE_KEY, {})


@callback
def begin_live_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    user_text: str,
    source: str,
) -> None:
    """Mark a conversation turn as active for live observation."""
    store = _live_store(hass)
    per_entry = store.setdefault(entry_id, {})
    per_entry[(entry_id, conversation_id)] = {
        "entry_id": entry_id,
        "conversation_id": conversation_id,
        "user_text": user_text,
        "source": source,
        "started_at": time.time(),
        "deltas": deque(maxlen=MAX_DELTAS_PER_TURN),
    }


@callback
def record_live_delta(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    payload: dict[str, Any],
) -> None:
    """Append one streamed delta to the live observation buffer."""
    store = _live_store(hass)
    session = store.get(entry_id, {}).get((entry_id, conversation_id))
    if session is None:
        return
    session["deltas"].append(
        {
            "at": time.time(),
            **{key: value for key, value in payload.items() if value is not None},
        }
    )


@callback
def end_live_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> None:
    """Clear the live observation session when a turn completes."""
    store = _live_store(hass)
    per_entry = store.get(entry_id)
    if not per_entry:
        return
    per_entry.pop((entry_id, conversation_id), None)


@callback
def live_snapshot(
    hass: HomeAssistant,
    entry_id: str,
    *,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Return active turns and recent deltas for external observation."""
    store = _live_store(hass)
    per_entry = store.get(entry_id, {})
    active: list[dict[str, Any]] = []
    for _key, session in per_entry.items():
        if conversation_id and session.get("conversation_id") != conversation_id:
            continue
        active.append(
            {
                "entry_id": session["entry_id"],
                "conversation_id": session["conversation_id"],
                "user_text": session.get("user_text", ""),
                "source": session.get("source", "console"),
                "started_at": session.get("started_at"),
                "deltas": list(session.get("deltas") or []),
            }
        )
    active.sort(key=lambda item: -(item.get("started_at") or 0))
    return {"active": active}

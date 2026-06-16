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
) -> None:
    """Append a turn trace to the per-entry activity ring buffer."""
    store = _activity_store(hass)
    buffer = store.get(entry_id)
    if buffer is None:
        buffer = deque(maxlen=max_turns)
        store[entry_id] = buffer
    buffer.append(
        turn_trace_to_dict(trace, timestamp=time.time()),
    )


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

"""Diagnostics API: inject console turns and return analysis."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from homeassistant.core import HomeAssistant, callback

from ..activity import get_turn
from ..diagnostics.analyze import analyze_turn_dict
from ..diagnostics.live import live_snapshot
from .chat import _chat_turn_timeout_seconds, start_chat
from .helpers import get_entry


async def inject_console_turn(
    hass: HomeAssistant,
    *,
    entry_id: str,
    text: str,
    conversation_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Send a console chat message and wait for completion plus analysis."""
    entry = get_entry(hass, entry_id)
    conv_id = conversation_id or (f"inject-{int(time.time())}-{uuid.uuid4().hex[:6]}")
    wait_timeout = (
        timeout if timeout is not None else (_chat_turn_timeout_seconds(entry) + 30.0)
    )

    loop = asyncio.get_running_loop()
    done_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    recorded_future: asyncio.Future[dict[str, Any]] = loop.create_future()

    @callback
    def _on_done(event) -> None:
        data = event.data
        if (
            data.get("entry_id") == entry_id
            and data.get("conversation_id") == conv_id
            and not done_future.done()
        ):
            done_future.set_result(dict(data))

    @callback
    def _on_recorded(event) -> None:
        data = event.data
        if (
            data.get("entry_id") == entry_id
            and data.get("conversation_id") == conv_id
            and not recorded_future.done()
        ):
            recorded_future.set_result(dict(data))

    unsub_done = hass.bus.async_listen("ha_agent_chat_done", _on_done)
    unsub_recorded = hass.bus.async_listen("ha_agent_turn_recorded", _on_recorded)
    try:
        start_chat(
            hass,
            entry_id=entry_id,
            conversation_id=conv_id,
            text=text,
        )
        done_data = await asyncio.wait_for(done_future, timeout=wait_timeout)
        recorded_data: dict[str, Any] | None = None
        try:
            recorded_data = await asyncio.wait_for(recorded_future, timeout=5.0)
        except TimeoutError:
            recorded_data = None
    finally:
        unsub_done()
        unsub_recorded()

    turn = None
    if recorded_data and recorded_data.get("turn"):
        turn = recorded_data["turn"]
    if turn is None:
        turn = get_turn(hass, entry_id, conversation_id=conv_id, latest=True)

    analysis = (
        analyze_turn_dict(turn)
        if turn
        else {
            "severity": "error",
            "summary": "Turn finished but no activity trace was recorded.",
            "issues": [],
            "suggested_actions": [],
        }
    )

    return {
        "conversation_id": conv_id,
        "done": done_data,
        "recorded": recorded_data,
        "turn": turn,
        "analysis": analysis,
        "live": live_snapshot(hass, entry_id, conversation_id=conv_id),
    }

"""Shared chat bus events and live observation hooks."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant, callback

from .diagnostics.live import begin_live_turn, end_live_turn, record_live_delta


@callback
def begin_chat_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    user_text: str,
    source: str,
) -> None:
    """Start live observation for a console or Assist turn."""
    begin_live_turn(
        hass,
        entry_id,
        conversation_id,
        user_text=user_text,
        source=source,
    )


@callback
def publish_chat_delta(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    content: str | None = None,
    thinking: str | None = None,
    thinking_clear: bool | None = None,
    content_clear: bool | None = None,
    tool: dict[str, Any] | None = None,
    skill: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    subagent: dict[str, Any] | None = None,
) -> None:
    """Emit a chat delta on the HA bus and record it for live observation."""
    payload: dict[str, Any] = {
        "entry_id": entry_id,
        "conversation_id": conversation_id,
        "content": content,
        "thinking": thinking,
        "thinking_clear": thinking_clear,
        "content_clear": content_clear,
        "tool": tool,
        "skill": skill,
        "meta": meta,
        "subagent": subagent,
    }
    record_live_delta(hass, entry_id, conversation_id, payload)
    hass.bus.async_fire("ha_agent_chat_delta", payload)


@callback
def finish_chat_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    done_payload: dict[str, Any] | None = None,
) -> None:
    """Emit chat completion and clear the live observation session."""
    payload = {
        "entry_id": entry_id,
        "conversation_id": conversation_id,
        **(done_payload or {}),
    }
    hass.bus.async_fire("ha_agent_chat_done", payload)
    end_live_turn(hass, entry_id, conversation_id)

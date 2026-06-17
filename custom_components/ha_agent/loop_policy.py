"""Agent loop policies: verification, error recovery, and stuck detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .llm_client import ToolCall


class TurnOutcome(StrEnum):
    """Terminal status for one agent turn."""

    SUCCESS = "success"
    NEEDS_USER = "needs_user"
    PARTIAL = "partial"
    FAILED = "failed"
    STUCK = "stuck"


@dataclass
class LoopState:
    """Mutable per-turn loop state."""

    tool_signatures: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    stuck: bool = False
    stuck_message: str = ""


def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a stable signature for duplicate tool-call detection."""
    try:
        args_blob = json.dumps(arguments, sort_keys=True, ensure_ascii=True)
    except TypeError:
        args_blob = str(arguments)
    return f"{tool_name}:{args_blob}"


def check_stuck(loop_state: LoopState, tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return an escalation message when the same tool call repeats."""
    signature = tool_call_signature(tool_name, arguments)
    if signature in loop_state.tool_signatures:
        loop_state.stuck = True
        loop_state.stuck_message = (
            "I tried the same tool with the same arguments twice without progress. "
            "Please narrow the request or tell me what to do differently."
        )
        return (
            f"Blocked repeated identical call to {tool_name}. "
            "Use a different tool, different arguments, or ask the user for help."
        )
    loop_state.tool_signatures.append(signature)
    return None


_EMAIL_LARGE_INBOX = re.compile(
    r"\b(too many|very large|large number|limit|timeout|overflow)\b",
    re.IGNORECASE,
)
_MCP_DOWN = re.compile(
    r"\b(unreachable|connection refused|timed out|timeout|502|503|504)\b",
    re.IGNORECASE,
)


def enrich_tool_output(
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
) -> str:
    """Append recovery hints to failed tool output."""
    if not output.startswith("Tool error:"):
        return output

    lowered = output.lower()
    name_lower = tool_name.lower()
    hints: list[str] = []

    if "mail" in name_lower or "imap" in name_lower or "email" in lowered:
        if _EMAIL_LARGE_INBOX.search(lowered):
            hints.append(
                "Search unread messages only with a small limit (e.g. 10) via "
                "mail_mcp_imap_search_messages instead of listing the full inbox."
            )
        hints.append(
            "Prefer mailbox_status for unseen count, then search_messages with "
            "unread_only=true before fetching individual messages."
        )

    if "news" in name_lower and "curate" not in name_lower:
        hints.append(
            "For headlines, call mcp_news__news_curate directly with "
            '{"limit": 5} before trying other news tools.'
        )

    if _MCP_DOWN.search(lowered):
        hints.append(
            "MCP may be offline. Tell the user to check MCP proxy connectivity "
            "in HA Agent Settings."
        )

    if "ha_call_service" in name_lower and "domain" in lowered:
        hints.append(
            "Include domain, service, and entity_id in ha_call_service arguments. "
            "Derive domain from the entity_id prefix (light.example -> light)."
        )

    if not hints:
        return output

    unique = list(dict.fromkeys(hints))
    return output + "\n\nRECOVERY HINTS:\n" + "\n".join(f"- {hint}" for hint in unique)


def _expected_states_for_service(service: str) -> set[str] | None:
    """Return acceptable HA states after a service call."""
    key = service.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"turn_on", "open_cover", "unlock", "media_play"}:
        return {"on", "open", "unlocked", "playing", "idle", "paused"}
    if key in {"turn_off", "close_cover", "lock", "media_pause", "media_stop"}:
        return {"off", "closed", "locked", "idle", "standby"}
    if key == "toggle":
        return None
    return None


def verify_ha_service(
    hass: HomeAssistant,
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
) -> str | None:
    """Verify entity state after a successful ha_call_service call."""
    if output.startswith("Tool error:"):
        return None
    if not tool_name.endswith("ha_call_service"):
        return None

    entity_id = arguments.get("entity_id")
    service = arguments.get("service")
    if not isinstance(entity_id, str) or not isinstance(service, str):
        return None

    state = hass.states.get(entity_id)
    if state is None:
        return f"VERIFICATION: {entity_id} was not found in Home Assistant."

    expected = _expected_states_for_service(service)
    if expected is None:
        return (
            f"VERIFICATION: {entity_id} is '{state.state}' after {service}."
        )

    if state.state in expected:
        return (
            f"VERIFICATION: {entity_id} is '{state.state}' after {service}."
        )
    return (
        f"VERIFICATION FAILED: {entity_id} is '{state.state}' after {service} "
        f"(expected one of {', '.join(sorted(expected))}). "
        "Do not tell the user the action succeeded."
    )


def finalize_output(
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
    *,
    hass: HomeAssistant | None = None,
    loop_state: LoopState | None = None,
) -> str:
    """Apply error enrichment and optional HA verification to tool output."""
    enriched = enrich_tool_output(tool_name, arguments, output)
    if hass is None:
        return enriched

    if note := verify_ha_service(hass, tool_name, arguments, enriched):
        if loop_state is not None:
            loop_state.verification_notes.append(note)
        return f"{enriched}\n\n{note}"
    return enriched

"""Promote production activity turns into custom eval benchmark cases."""

from __future__ import annotations

import re
import time
from typing import Any

from homeassistant.exceptions import HomeAssistantError

from ..skills.models import TurnTrace
from .models import EVAL_TASKS, EvalCase

_DISCOVERY_TOOL_MARKERS = (
    "searchtoolsfordomain",
    "searchtool",
    "tools/list",
    "list_tools",
)

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "your",
        "you",
        "i",
        "is",
        "are",
        "was",
        "to",
        "for",
        "of",
        "in",
        "it",
        "that",
        "this",
        "with",
        "have",
        "has",
        "been",
        "here",
        "there",
    }
)

_HA_SERVICE_ARG_KEYS = ("domain", "service", "entity_id")


def turn_dict_to_trace(data: dict[str, Any]) -> TurnTrace:
    """Rebuild a TurnTrace from an activity log row."""
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
    )


def find_activity_turn(
    turns: list[dict[str, Any]],
    timestamp: float,
) -> dict[str, Any] | None:
    """Find one activity row by its timestamp (float tolerance)."""
    target = float(timestamp)
    for turn in turns:
        raw = turn.get("timestamp")
        if raw is None:
            continue
        if abs(float(raw) - target) < 0.001:
            return turn
    return None


def _tool_payload(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if not isinstance(args, dict):
        return {}
    inner = args.get("arguments")
    return inner if isinstance(inner, dict) else args


def _primary_tool_call(
    tool_calls: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    """Return the last non-discovery tool call from a turn."""
    selected_name: str | None = None
    selected_args: dict[str, Any] | None = None
    for call in tool_calls:
        name = str(
            call.get("toolName") or call.get("name") or call.get("tool_name") or ""
        )
        lowered = name.lower()
        if not name or any(marker in lowered for marker in _DISCOVERY_TOOL_MARKERS):
            continue
        payload = _tool_payload(call)
        selected_name = name
        selected_args = payload or None
    return selected_name, selected_args


def _expected_tool_args(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    if "domain" in payload or "service" in payload:
        return {
            key: payload[key]
            for key in _HA_SERVICE_ARG_KEYS
            if key in payload and payload[key] is not None
        } or None
    return {key: value for key, value in payload.items() if value is not None} or None


def _text_tokens(text: str, *, limit: int = 3) -> list[str]:
    words = re.findall(r"[a-z0-9']{3,}", text.lower())
    picked: list[str] = []
    for word in words:
        if word in _STOP_WORDS:
            continue
        if word not in picked:
            picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def _resolve_task(trace: TurnTrace, *, task_override: str | None = None) -> str:
    if task_override:
        task = task_override.strip().lower()
        if task not in EVAL_TASKS:
            raise HomeAssistantError(
                f"Unsupported eval task {task!r}. "
                f"Use one of: {', '.join(EVAL_TASKS)}."
            )
        return task
    route = (trace.route or "").strip().lower()
    if route in EVAL_TASKS and route != "classifier":
        return route
    raise HomeAssistantError(
        "Turn has no route metadata. Re-run the request after upgrading, "
        "or pass an explicit task when promoting."
    )


def validate_turn_for_promotion(trace: TurnTrace) -> None:
    """Raise when a turn should not become an eval case."""
    if trace.fallback:
        raise HomeAssistantError("Cannot promote a fallback response.")
    if trace.tool_errors:
        raise HomeAssistantError(
            f"Cannot promote a turn with {trace.tool_errors} tool error(s)."
        )
    if trace.outcome and trace.outcome not in {"success", "partial"}:
        raise HomeAssistantError(
            f"Cannot promote a turn with outcome {trace.outcome!r}."
        )
    if not trace.user_text.strip():
        raise HomeAssistantError("Turn has no user text.")
    if not trace.assistant_text.strip() and not trace.tool_calls:
        raise HomeAssistantError("Turn has no assistant reply or tool calls.")


def build_case_from_turn(
    trace: TurnTrace,
    *,
    source_timestamp: float | None = None,
    case_id: str | None = None,
    task_override: str | None = None,
) -> EvalCase:
    """Build a promoted eval case from a production turn trace."""
    validate_turn_for_promotion(trace)
    task = _resolve_task(trace, task_override=task_override)
    expected_tool, tool_payload = _primary_tool_call(trace.tool_calls)
    expected_tool_args = _expected_tool_args(tool_payload)
    text_source = trace.assistant_text or trace.user_text
    expected_text = _text_tokens(text_source)
    if not expected_text and expected_tool:
        expected_text = _text_tokens(trace.user_text, limit=2)
    mock_responses = ['{"success": true}' for _ in trace.tool_calls] or [
        '{"success": true}'
    ]
    promoted_at = time.time()
    resolved_id = case_id or f"promoted-{int((source_timestamp or promoted_at) * 1000)}"
    max_iterations = max(trace.iterations + 1, 6)
    return EvalCase(
        id=resolved_id,
        task=task,
        user_text=trace.user_text.strip(),
        exposed_entities=list(trace.exposed_entities),
        expected_tool=expected_tool,
        expected_tool_args=expected_tool_args,
        expected_text_contains=expected_text,
        mock_mcp_responses=mock_responses,
        max_iterations=max_iterations,
        source="promoted",
        promoted_at=promoted_at,
        source_timestamp=source_timestamp,
        source_conversation_id=trace.conversation_id,
    )

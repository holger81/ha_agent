"""Verifier/critic pass before finalizing assistant answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config_helpers import LlmBackend
from .const import LOGGER
from .llm_client import LlmClient
from .llm_telemetry import record_llm_call
from .skills.models import Skill
from .structured_output import VERIFIER_SCHEMA, json_schema_format, strip_json_fence

_VERIFY_PROMPT = (
    "You verify whether an assistant turn satisfied the user goal.\n"
    'Return ONLY JSON: {"pass": true|false, "reason": "...", '
    '"skill_followed": true|false, "retry_hint": "..."}.\n'
    "Rules:\n"
    "- pass=false when the goal is not met or critical tools failed.\n"
    "- skill_followed=false when a workflow skill was provided but ignored "
    "without justified adaptation.\n"
    "- retry_hint: one sentence for the worker if pass=false."
)

_TRACE_FIELD_MAX_CHARS = 2_000


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Outcome of the verifier critic pass."""

    passed: bool
    reason: str
    skill_followed: bool = True
    retry_hint: str = ""


def _coerce_bool(value: Any) -> bool:
    """Parse JSON booleans, including string forms like ``\"false\"``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _cap_text(value: Any, *, max_chars: int = _TRACE_FIELD_MAX_CHARS) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max_chars - 16] + "…[truncated]"
    return value


def _serialize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Cap argument/error fields before sending tool calls to the verifier."""
    serialized = dict(call)
    arguments = serialized.get("arguments")
    if isinstance(arguments, dict):
        serialized["arguments"] = {
            key: _cap_text(val) if isinstance(val, str) else val
            for key, val in arguments.items()
        }
    elif isinstance(arguments, str):
        serialized["arguments"] = _cap_text(arguments)
    if "error" in serialized:
        serialized["error"] = _cap_text(serialized.get("error"))
    return serialized


def parse_verifier_response(content: str) -> VerifierResult | None:
    """Parse verifier JSON."""
    try:
        data = json.loads(strip_json_fence(content))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "pass" not in data:
        return None
    return VerifierResult(
        passed=_coerce_bool(data.get("pass")),
        reason=str(data.get("reason", "")).strip(),
        skill_followed=_coerce_bool(data.get("skill_followed", True)),
        retry_hint=str(data.get("retry_hint", "")).strip(),
    )


async def verify_turn(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    assistant_text: str,
    tool_calls: list[dict[str, Any]],
    tool_errors: int,
    skill: Skill | None = None,
    slot_bindings: dict[str, str] | None = None,
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> VerifierResult:
    """Run the verifier critic on a completed worker turn."""
    payload: dict[str, Any] = {
        "user_goal": user_text,
        "assistant_reply": assistant_text[:2000],
        "tool_calls": [_serialize_tool_call(call) for call in tool_calls[-12:]],
        "tool_errors": tool_errors,
    }
    if skill:
        payload["skill"] = {
            "title": skill.title,
            "body": skill.body[:1500],
            "tool_steps": skill.tool_steps,
            "slot_bindings": slot_bindings or {},
        }
    messages = [
        {"role": "system", "content": _VERIFY_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    response_format = (
        json_schema_format("verifier", VERIFIER_SCHEMA, strict=False)
        if structured_output_enabled
        else None
    )
    try:
        result = await llm.chat(
            messages,
            backend,
            tools=[],
            response_format=response_format,
        )
        record_llm_call(trace, role="verifier", backend=backend, result=result)
    except Exception as err:
        LOGGER.warning("Verifier LLM call failed: %s", err)
        record_llm_call(trace, role="verifier", backend=backend, error=str(err))
        return VerifierResult(passed=True, reason="verifier unavailable")
    parsed = parse_verifier_response(result.content or "")
    if parsed is None:
        return VerifierResult(passed=True, reason="verifier parse failed")
    return parsed


def build_verifier_retry_guidance(result: VerifierResult) -> str:
    """Return internal guidance injected when verifier rejects a turn."""
    lines = [
        "VERIFIER REJECTED PREVIOUS ATTEMPT (internal — not from the user):",
        result.reason or "Goal not satisfied.",
    ]
    if not result.skill_followed:
        lines.append(
            "Adapt the active skill workflow — change slot values (mailbox, "
            "folder, date range) but keep the same tool sequence."
        )
    if result.retry_hint:
        lines.append(f"Retry: {result.retry_hint}")
    return "\n".join(lines)

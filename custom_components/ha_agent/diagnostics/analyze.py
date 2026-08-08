"""Deterministic turn analysis for external debugging agents."""

from __future__ import annotations

import re
from typing import Any

_DISCOVERY_MARKERS = (
    "searchtoolsfordomain",
    "searchtool",
    "tools/list",
    "list_tools",
)


def _severity(issues: list[dict[str, Any]]) -> str:
    kinds = {issue.get("kind") for issue in issues}
    if {
        "tool_error",
        "fallback",
        "verifier_fail",
        "outcome_failed",
        "false_action_success",
    } & kinds:
        return "error"
    if kinds - {"ok"}:
        return "warning"
    return "ok"


def _is_discovery_tool(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _DISCOVERY_MARKERS)


def analyze_turn_dict(turn: dict[str, Any]) -> dict[str, Any]:
    """Return structured issues and suggested follow-ups for one activity turn."""
    issues: list[dict[str, Any]] = []
    suggested_actions: list[dict[str, str]] = []

    if turn.get("fallback"):
        issues.append(
            {
                "kind": "fallback",
                "detail": (
                    "Agent returned a fallback response instead of a full answer."
                ),
                "suggestion": (
                    "Check MCP reachability, loop limits, and repeated-tool guards."
                ),
            }
        )

    outcome = str(turn.get("outcome") or "")
    if outcome and outcome not in {"success", "partial"}:
        issues.append(
            {
                "kind": "outcome_failed",
                "detail": f"Turn outcome was {outcome!r}.",
                "suggestion": "Inspect tool errors and verifier notes in the trace.",
            }
        )

    if turn.get("verifier_verdict") == "fail":
        detail = str(turn.get("verifier_detail") or "Verifier rejected the answer.")
        issues.append(
            {
                "kind": "verifier_fail",
                "detail": detail,
                "suggestion": (
                    "Compare assistant text with tool results and active skills."
                ),
            }
        )

    tool_errors = int(turn.get("tool_errors") or 0)
    if tool_errors:
        issues.append(
            {
                "kind": "tool_error",
                "detail": f"{tool_errors} tool call(s) failed.",
                "suggestion": "Review per-call error, error_kind, and missing_fields.",
            }
        )

    for call in turn.get("tool_calls") or []:
        if call.get("succeeded") is not False:
            continue
        name = str(call.get("toolName") or call.get("name") or "tool")
        error = str(call.get("error") or "unknown error")
        missing = call.get("missing_fields") or []
        detail = f"{name}: {error}"
        if missing:
            detail += f" (missing: {', '.join(str(item) for item in missing)})"
        issues.append(
            {
                "kind": "tool_call_failed",
                "detail": detail,
                "suggestion": (
                    "Retry with required arguments or update the matched skill "
                    "template."
                    if missing
                    else "Check MCP tool schema and arguments."
                ),
            }
        )

    if turn.get("skill_followed") is False:
        issues.append(
            {
                "kind": "skill_not_followed",
                "detail": "Matched skill workflow was not followed.",
                "suggestion": (
                    "Open skill revisions or run auto-repair from a similar turn."
                ),
            }
        )

    discovery_calls = [
        str(call.get("toolName") or call.get("name") or "")
        for call in turn.get("tool_calls") or []
        if _is_discovery_tool(str(call.get("toolName") or call.get("name") or ""))
    ]
    if discovery_calls and turn.get("matched_skill_ids"):
        issues.append(
            {
                "kind": "discovery_with_skill",
                "detail": (
                    "Discovery tools were called although skill(s) were matched: "
                    + ", ".join(discovery_calls)
                ),
                "suggestion": (
                    "Ensure the skill lists concrete tool_steps and blocks discovery."
                ),
            }
        )

    for note in turn.get("verification_notes") or []:
        if "VERIFICATION FAILED" in str(note):
            issues.append(
                {
                    "kind": "verification_failed",
                    "detail": str(note),
                    "suggestion": "Check entity state after the service call.",
                }
            )

    route = str(
        turn.get("route") or (turn.get("turn_meta") or {}).get("route") or ""
    ).lower()
    assistant = str(turn.get("assistant_text") or "")
    tool_calls = turn.get("tool_calls") or []
    controlled = turn.get("controlled_entity_ids") or []
    success_claim = bool(
        re.search(
            r"\b(successfully|turned on|turned off|completed|all set|"
            r"i'?ve turned|i have turned)\b|"
            r"\bcontrolled:\s*\w+\.",
            assistant,
            re.IGNORECASE,
        )
    ) and not re.search(
        r"\b(couldn'?t|could not|unable|failed|error|didn'?t work|not able|"
        r"haven'?t confirmed)\b",
        assistant,
        re.IGNORECASE,
    )
    control_ok = False
    for call in tool_calls:
        if call.get("succeeded") is False:
            continue
        name = str(call.get("toolName") or call.get("name") or "")
        if re.search(
            r"(ha_call_service|hassturnon|hassturnoff|hasstoggle)\b",
            name,
            re.IGNORECASE,
        ):
            control_ok = True
            break
    if controlled:
        control_ok = True
    if route == "action" and success_claim and not control_ok:
        issues.append(
            {
                "kind": "false_action_success",
                "detail": (
                    "Action route claimed device success without a successful "
                    "control tool call."
                ),
                "suggestion": (
                    "Retry the request; ensure the action model calls "
                    "ha_call_service / HassTurnOn / HassTurnOff before confirming."
                ),
            }
        )

    if not issues:
        issues.append(
            {
                "kind": "ok",
                "detail": "No obvious issues detected in the turn trace.",
                "suggestion": "",
            }
        )

    timestamp = turn.get("timestamp")
    if (
        timestamp is not None
        and not turn.get("fallback")
        and tool_errors == 0
        and not any(issue.get("kind") == "false_action_success" for issue in issues)
    ):
        suggested_actions.append(
            {
                "action": "promote_eval_case",
                "detail": (
                    "Promote this turn via ha_agent/eval/cases/promote "
                    f"with timestamp {timestamp} for regression coverage."
                ),
            }
        )

    if any(issue.get("kind") == "tool_call_failed" for issue in issues):
        suggested_actions.append(
            {
                "action": "inspect_skill_repair",
                "detail": (
                    "Check whether auto-repair updated a matched skill "
                    "(turn_meta.skill_update) or run ha_agent/diagnostics/analyze_turn "
                    "after the next retry."
                ),
            }
        )

    if turn.get("conversation_id"):
        suggested_actions.append(
            {
                "action": "open_chat_history",
                "detail": (
                    "Load full conversation with ha_agent/chat/history/list "
                    f"for conversation_id {turn['conversation_id']}."
                ),
            }
        )

    severity = _severity(issues)
    summary = issues[0]["detail"] if issues else "Turn recorded."
    if severity == "ok":
        summary = "Turn completed without detected issues."

    return {
        "severity": severity,
        "summary": summary,
        "issues": issues,
        "suggested_actions": suggested_actions,
    }

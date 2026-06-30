"""Agent loop policies: verification, error recovery, and stuck detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


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
    duplicate_blocks: dict[str, int] = field(default_factory=dict)
    verification_notes: list[str] = field(default_factory=list)
    stuck: bool = False
    stuck_message: str = ""
    unproductive_iterations: int = 0
    iteration_had_successful_tool: bool = False
    iteration_had_duplicate_block: bool = False
    iteration_failures: list[str] = field(default_factory=list)
    pending_failure_summary: str | None = None
    plan_goal: str = ""
    plan_route: str = ""
    plan_skill_title: str = ""
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    plan_step_statuses: list[str] = field(default_factory=list)
    plan_current_step_index: int | None = None
    plan_completed_tools: list[str] = field(default_factory=list)
    skill_plan_override: bool = False
    skill_plan_override_reason: str = ""
    empty_responses: int = 0
    mcp_guidance: list[str] = field(default_factory=list)
    include_full_tool_catalog: bool = False


# Role used for internal/system-injected guidance (plan progress, failure
# summaries, MCP guidance, empty-response nudges). These are NOT user input.
# The backend is OpenAI-compatible (llama.cpp / local servers) and forwards
# messages verbatim, so a mid-conversation ``system`` message is accepted and
# rendered as instruction content by standard chat templates. ``system`` is
# more widely supported than the newer ``developer`` role and is the role models
# most reliably treat as instructions rather than user input.
INTERNAL_GUIDANCE_ROLE = "system"

_MAX_REASONING_CHARS = 8000
_MAX_EMPTY_RESPONSES = 2
_MAX_MCP_GUIDANCE_CHARS = 600
_ROUTE_PLAN_STEPS: dict[str, list[dict[str, Any]]] = {
    "email": [
        {"toolName": "mail_mcp__imap_mailbox_status"},
        {"toolName": "mail_mcp__imap_search_messages"},
        {"toolName": "mail_mcp__imap_get_message"},
    ],
    "news": [
        {"toolName": "news_curate"},
    ],
    "action": [
        {"toolName": "ha_call_service"},
    ],
}
_ROUTE_NEXT_HINTS: dict[str, str] = {
    "email": (
        "Complete the email workflow: check mailbox, search messages, "
        "fetch bodies if needed, then answer."
    ),
    "news": "Run news_curate (or equivalent), then summarize headlines.",
    "action": (
        "Prefer exposed-entity shortcuts when they match; otherwise discover "
        "entities in domain smart-home, then call ha_call_service with domain, "
        "service, and entity_id. Do not call ha_search_entities."
    ),
    "general": "Use tools to gather evidence, then answer from results.",
}
_REASONING_REPEAT_MARKER = 60
_MAX_UNPRODUCTIVE_ITERATIONS = 2
_REASONING_WILL_CALL = re.compile(
    r"\b(?:will|should|i'?ll|going to)\s+call\s+`?([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)`?",
    re.IGNORECASE,
)
_REASONING_TOOL_BACKTICK = re.compile(
    r"`([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)`",
    re.IGNORECASE,
)
_SKILL_OVERRIDE_MARKER = re.compile(
    r"SKILL_OVERRIDE:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_USER_SKILL_OVERRIDE = re.compile(
    r"\b(?:"
    r"ignore (?:the )?skill|"
    r"without (?:the )?skill|"
    r"don'?t use (?:the )?skill|"
    r"override (?:the )?skill|"
    r"forget (?:the )?skill|"
    r"skip (?:the )?skill|"
    r"not using (?:the )?skill"
    r")\b",
    re.IGNORECASE,
)
_REASONING_SKILL_MISMATCH = re.compile(
    r"\b(?:"
    r"skill (?:does not|doesn'?t) (?:include|cover|apply|fit|match)|"
    r"neither includes?|"
    r"not (?:in|part of) (?:the )?(?:active )?skill|"
    r"outside (?:the )?skill(?: workflow)?|"
    r"override (?:the )?skill(?: workflow| plan)?|"
    r"abandon (?:the )?skill|"
    r"skill workflow (?:does not|doesn'?t)|"
    r"no (?:matching )?tool step|"
    r"need to (?:run )?discover"
    r")\b",
    re.IGNORECASE,
)


def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a stable signature for duplicate tool-call detection."""
    try:
        args_blob = json.dumps(arguments, sort_keys=True, ensure_ascii=True)
    except TypeError:
        args_blob = str(arguments)
    return f"{tool_name}:{args_blob}"


def normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool args so equivalent calls share one signature."""
    normalized: dict[str, Any] = {}
    for key, value in sorted(arguments.items()):
        if value is None:
            continue
        if isinstance(value, str):
            normalized[key] = value.strip()
        else:
            normalized[key] = value
    return normalized


def reasoning_stream_stuck(buffer: str) -> bool:
    """Return True when streamed reasoning is repeating or too long."""
    if len(buffer) > _MAX_REASONING_CHARS:
        return True
    if len(buffer) < 240:
        return False
    marker = buffer[-_REASONING_REPEAT_MARKER:]
    if len(marker.strip()) < 20:
        return False
    return buffer.count(marker) >= 4


def check_stuck(
    loop_state: LoopState,
    tool_name: str,
    arguments: dict[str, Any],
) -> str | None:
    """Return a block message when the same tool call repeats.

    The first duplicate is a soft block: the model gets the error in context and
    another loop iteration to replan. A second duplicate of the same signature
    ends the turn as stuck.
    """
    signature = tool_call_signature(
        tool_name,
        normalize_tool_arguments(arguments),
    )
    if signature not in loop_state.tool_signatures:
        loop_state.tool_signatures.append(signature)
        return None

    blocks = loop_state.duplicate_blocks.get(signature, 0) + 1
    loop_state.duplicate_blocks[signature] = blocks

    if blocks >= 2:
        loop_state.stuck = True
        loop_state.stuck_message = (
            "I tried the same tool with the same arguments twice without progress. "
            "Please narrow the request or tell me what to do differently."
        )
        return (
            f"Blocked repeated identical call to {tool_name}. "
            "Use a different tool, different arguments, or ask the user for help."
        )

    return (
        f"Blocked repeated identical call to {tool_name}. "
        "You already used this tool with the same arguments. "
        "STOP retrying this call this turn. Review the previous tool result, "
        "answer from it if sufficient, or use a different tool (for example "
        "mail_mcp__imap_get_message with message_id from search results)."
    )


def reset_iteration_flags(loop_state: LoopState) -> None:
    """Clear per-iteration progress markers."""
    loop_state.iteration_had_successful_tool = False
    loop_state.iteration_had_duplicate_block = False
    loop_state.iteration_failures = []


def _compact_tool_detail(detail: str, *, limit: int = 200) -> str:
    text = detail.removeprefix("Tool error:").strip()
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _compact_arguments(arguments: dict[str, Any], *, limit: int = 120) -> str:
    try:
        preview = json.dumps(
            normalize_tool_arguments(arguments),
            ensure_ascii=True,
            sort_keys=True,
        )
    except TypeError:
        preview = str(arguments)
    if len(preview) > limit:
        return f"{preview[: limit - 3]}..."
    return preview


def record_iteration_failure(
    loop_state: LoopState,
    tool_name: str,
    arguments: dict[str, Any],
    detail: str,
) -> None:
    """Remember a failed or blocked tool call for the next loop iteration."""
    line = (
        f"- {tool_name}({_compact_arguments(arguments)}): "
        f"{_compact_tool_detail(detail)}"
    )
    loop_state.iteration_failures.append(line)


def build_pending_failure_summary(loop_state: LoopState) -> None:
    """Compile this iteration's failures for injection before the next step."""
    if not loop_state.iteration_failures:
        loop_state.pending_failure_summary = None
        return
    unique = list(dict.fromkeys(loop_state.iteration_failures))
    body = "\n".join(unique)
    loop_state.pending_failure_summary = (
        "TURN PROGRESS SUMMARY (internal — not from the user):\n"
        "The previous step failed or was blocked. Do not retry these "
        "approaches unchanged.\n"
        f"{body}\n"
        "Use prior successful results, different arguments, or a different tool."
    )
    loop_state.iteration_failures = []


def inject_pending_failure_summary(
    messages: list[dict[str, Any]],
    loop_state: LoopState,
) -> None:
    """Insert the compiled failure summary into the next agent loop step."""
    inject_loop_context(messages, loop_state)


def _tool_names_match(plan_tool: str, actual_tool: str) -> bool:
    if plan_tool == actual_tool:
        return True
    plan_tail = plan_tool.split("__")[-1]
    actual_tail = actual_tool.split("__")[-1]
    return plan_tail == actual_tail or actual_tool.endswith(plan_tool)


def extract_intended_tools_from_reasoning(reasoning: str) -> list[str]:
    """Return tool names the model committed to in streamed reasoning."""
    text = reasoning.strip()
    if not text:
        return []
    tail = text[-1200:]
    will_calls = [match.group(1) for match in _REASONING_WILL_CALL.finditer(tail)]
    if will_calls:
        return list(dict.fromkeys(will_calls))
    backticks = [match.group(1) for match in _REASONING_TOOL_BACKTICK.finditer(tail)]
    if backticks:
        return list(dict.fromkeys(backticks))
    return []


def reasoning_execution_mismatch(
    reasoning: str,
    execution_tools: list[str],
) -> str | None:
    """Return guidance when reasoning names different tools than execution."""
    from .tools import is_discovery_tool_name

    intended = extract_intended_tools_from_reasoning(reasoning)
    if not intended or not execution_tools:
        return None

    actionable = [
        name for name in execution_tools if not is_discovery_tool_name(name)
    ]
    if not actionable:
        return None

    for actual in actionable:
        if any(_tool_names_match(intent, actual) for intent in intended):
            return None

    primary = intended[-1]
    actual = actionable[0]
    return (
        "REASONING / EXECUTION MISMATCH (internal — not from the user):\n"
        f"Your reasoning selected `{primary}` but the tool payload used "
        f"`{actual}`. Do NOT call `{actual}`. "
        f"Call `{primary}` with the arguments from your reasoning instead."
    )


def _next_incomplete_plan_step(loop_state: LoopState) -> int | None:
    for index, status in enumerate(loop_state.plan_step_statuses):
        if status != "done":
            return index
    return None


def _match_plan_step_index(loop_state: LoopState, tool_name: str) -> int | None:
    for index, step in enumerate(loop_state.plan_steps):
        plan_tool = str(step.get("toolName", ""))
        if not plan_tool:
            continue
        if not _tool_names_match(plan_tool, tool_name):
            continue
        status = loop_state.plan_step_statuses[index]
        if status in {"pending", "needs_work"}:
            return index
    for index, step in enumerate(loop_state.plan_steps):
        plan_tool = str(step.get("toolName", ""))
        if plan_tool and _tool_names_match(plan_tool, tool_name):
            return index
    return None


def user_requests_skill_override(user_text: str) -> bool:
    """Return True when the user explicitly asks to bypass the active skill."""
    return bool(_USER_SKILL_OVERRIDE.search(user_text.strip()))


def reasoning_declares_skill_mismatch(reasoning: str) -> bool:
    """Return True when model reasoning states the active skill does not fit."""
    text = reasoning.strip()
    if not text:
        return False
    if _SKILL_OVERRIDE_MARKER.search(text):
        return True
    tail = text[-2400:]
    if not _REASONING_SKILL_MISMATCH.search(tail):
        return False
    return "skill" in tail.lower() or "workflow" in tail.lower()


def extract_skill_override_reason(reasoning: str) -> str | None:
    """Return an override reason from explicit markers or mismatch reasoning."""
    text = reasoning.strip()
    if not text:
        return None
    marker = _SKILL_OVERRIDE_MARKER.search(text)
    if marker:
        reason = marker.group(1).strip()
        return reason[:400] if reason else "Declared in reasoning."
    if reasoning_declares_skill_mismatch(text):
        tail = text[-400:].strip()
        return tail[:400] if tail else "Active skill does not fit the user's goal."
    return None


def suspend_skill_plan(loop_state: LoopState, reason: str) -> None:
    """Stop enforcing the active skill's concrete tool-step plan for this turn."""
    loop_state.skill_plan_override = True
    loop_state.skill_plan_override_reason = reason.strip()[:400]
    loop_state.plan_steps = []
    loop_state.plan_step_statuses = []
    loop_state.plan_current_step_index = None


def maybe_suspend_skill_plan_from_reasoning(
    loop_state: LoopState,
    reasoning: str,
) -> bool:
    """Suspend the skill plan when reasoning explicitly declares a mismatch."""
    if loop_state.skill_plan_override:
        return False
    reason = extract_skill_override_reason(reasoning)
    if not reason:
        return False
    suspend_skill_plan(loop_state, reason)
    return True


def skill_plan_blocks_discovery(loop_state: LoopState) -> bool:
    """Return True when discovery tools should stay blocked for the skill plan."""
    return (
        not loop_state.skill_plan_override
        and bool(loop_state.plan_steps)
        and len(loop_state.plan_steps) >= 2
        and bool(loop_state.plan_skill_title)
    )


def initialize_loop_plan(
    loop_state: LoopState,
    *,
    goal: str,
    route: str,
    tool_steps: list[dict[str, Any]] | None = None,
    skill_title: str = "",
    slot_bindings: dict[str, str] | None = None,
) -> None:
    """Seed per-turn plan state from the user goal, route, and optional skill."""
    loop_state.plan_goal = goal.strip()
    loop_state.plan_route = route
    loop_state.plan_skill_title = skill_title
    steps = list(tool_steps or _ROUTE_PLAN_STEPS.get(route, []))
    loop_state.plan_steps = steps
    loop_state.plan_step_statuses = ["pending"] * len(steps)
    loop_state.plan_current_step_index = 0 if steps else None
    loop_state.plan_completed_tools = []
    if slot_bindings:
        bound = ", ".join(
            f"{key}={value}" for key, value in slot_bindings.items() if value
        )
        if bound:
            loop_state.mcp_guidance.insert(
                0,
                (
                    "ADAPT skill workflow — bound slots: "
                    f"{bound}. Change slot values for this goal; "
                    "keep the same tool sequence."
                ),
            )


def record_plan_tool_result(
    loop_state: LoopState,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    succeeded: bool,
    verification_failed: bool = False,
) -> None:
    """Update plan step progress after a tool call attempt."""
    if not loop_state.plan_goal:
        return

    if (
        succeeded
        and not verification_failed
        and tool_name not in loop_state.plan_completed_tools
    ):
        loop_state.plan_completed_tools.append(tool_name)

    step_index = _match_plan_step_index(loop_state, tool_name)
    if step_index is None:
        return

    if succeeded and not verification_failed:
        loop_state.plan_step_statuses[step_index] = "done"
        loop_state.plan_current_step_index = _next_incomplete_plan_step(loop_state)
        return

    loop_state.plan_step_statuses[step_index] = "needs_work"
    loop_state.plan_current_step_index = step_index


def describe_plan_next_action(loop_state: LoopState) -> str:
    """Return a short directive for what the model should do next."""
    if loop_state.plan_steps and loop_state.plan_current_step_index is not None:
        index = loop_state.plan_current_step_index
        if index < len(loop_state.plan_steps):
            step = loop_state.plan_steps[index]
            name = str(step.get("toolName", "tool"))
            status = loop_state.plan_step_statuses[index]
            if status == "needs_work":
                return (
                    f"Fix step {index + 1} ({name}) — use different arguments "
                    "or prior tool output."
                )
            if status == "pending":
                return f"Execute step {index + 1}: {name}"
    if (
        loop_state.plan_steps
        and loop_state.plan_step_statuses
        and all(status == "done" for status in loop_state.plan_step_statuses)
    ):
        return (
            "All planned steps are done. STOP calling tools and write the final "
            "answer to the user now using the tool results above."
        )

    hint = _ROUTE_NEXT_HINTS.get(
        loop_state.plan_route,
        _ROUTE_NEXT_HINTS["general"],
    )
    if loop_state.plan_completed_tools:
        return f"{hint} Do not repeat tools that already succeeded."
    return hint


def build_plan_progress_summary(loop_state: LoopState) -> str | None:
    """Compile plan progress for injection at the start of a loop step."""
    if not loop_state.plan_goal:
        return None

    if loop_state.skill_plan_override:
        reason = loop_state.skill_plan_override_reason or (
            "Active skill does not fit the user's goal."
        )
        return (
            "AGENT PLAN (internal — not from the user): Active skill workflow "
            f"suspended — {reason} Use discovery and other tools as needed."
        )

    lines = [
        "AGENT PLAN PROGRESS (internal — not from the user):",
        f"Goal: {loop_state.plan_goal}",
    ]
    if loop_state.plan_skill_title:
        lines.append(f"Workflow skill: {loop_state.plan_skill_title}")

    if loop_state.plan_steps:
        lines.append("Plan steps:")
        for index, step in enumerate(loop_state.plan_steps):
            name = str(step.get("toolName", "step"))
            status = (
                loop_state.plan_step_statuses[index]
                if index < len(loop_state.plan_step_statuses)
                else "pending"
            )
            marker = {"pending": "[ ]", "done": "[x]", "needs_work": "[!]"}[status]
            focus = ""
            if loop_state.plan_current_step_index == index and status != "done":
                focus = "  <-- focus here"
            lines.append(f"{index + 1}. {marker} {name}{focus}")
    elif loop_state.plan_completed_tools:
        lines.append("Tools completed this turn:")
        for name in loop_state.plan_completed_tools[-6:]:
            lines.append(f"- {name}")

    lines.append(f"Next action: {describe_plan_next_action(loop_state)}")

    if (
        loop_state.plan_current_step_index is not None
        and loop_state.plan_step_statuses
        and loop_state.plan_current_step_index < len(loop_state.plan_step_statuses)
        and loop_state.plan_step_statuses[loop_state.plan_current_step_index]
        == "needs_work"
    ):
        lines.append("The current plan step still needs work before advancing.")

    return "\n".join(lines)


def inject_loop_context(
    messages: list[dict[str, Any]],
    loop_state: LoopState,
) -> None:
    """Insert plan progress, MCP guidance, and failures before a loop step."""
    parts: list[str] = []
    plan = build_plan_progress_summary(loop_state)
    if plan:
        parts.append(plan)
    if loop_state.mcp_guidance:
        guidance = "\n".join(f"- {hint}" for hint in loop_state.mcp_guidance)
        parts.append(
            "MCP SERVER GUIDANCE (from tool discovery — follow it):\n" + guidance
        )
        loop_state.mcp_guidance = []
    if loop_state.pending_failure_summary:
        parts.append(loop_state.pending_failure_summary)
        loop_state.pending_failure_summary = None
    if not parts:
        return
    messages.append(
        {"role": INTERNAL_GUIDANCE_ROLE, "content": "\n\n".join(parts)}
    )


def extract_mcp_guidance(tool_name: str, output: str) -> list[str]:
    """Pull serverLlmContext guidance from a discovery tool result."""
    if output.startswith("Tool error:"):
        return []
    if "searchtool" not in tool_name.lower():
        return []
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []

    entries: list[dict[str, Any]] = []
    if isinstance(data, list):
        entries = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        for key in ("tools", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                entries = [item for item in value if isinstance(item, dict)]
                break
        if not entries:
            entries = [data]

    guidance: list[str] = []
    for entry in entries:
        context = entry.get("serverLlmContext")
        if isinstance(context, str) and context.strip():
            guidance.append(context.strip()[:_MAX_MCP_GUIDANCE_CHARS])
    return list(dict.fromkeys(guidance))


def record_mcp_guidance(
    loop_state: LoopState,
    tool_name: str,
    output: str,
) -> None:
    """Stash discovery guidance for injection into the next loop step."""
    for hint in extract_mcp_guidance(tool_name, output):
        if hint not in loop_state.mcp_guidance:
            loop_state.mcp_guidance.append(hint)


def build_empty_response_nudge(loop_state: LoopState) -> str:
    """Return a directive when the model produced no answer and no tool call."""
    return (
        "SYSTEM (internal — not from the user): Your previous reply was empty. "
        "Either call exactly one tool to make progress, or write the final "
        "answer to the user in plain text now. Do not send an empty message "
        f"again. {describe_plan_next_action(loop_state)}"
    )


def should_retry_empty_response(
    loop_state: LoopState,
    iteration: int,
    max_iterations: int,
) -> bool:
    """Return True when an empty model reply should trigger a guided retry."""
    if iteration >= max_iterations - 1:
        return False
    loop_state.empty_responses += 1
    return loop_state.empty_responses <= _MAX_EMPTY_RESPONSES


def mark_iteration_outcome(loop_state: LoopState) -> None:
    """Track iterations that repeat blocked calls without progress."""
    if loop_state.iteration_had_successful_tool:
        loop_state.unproductive_iterations = 0
        return
    if loop_state.iteration_had_duplicate_block:
        loop_state.unproductive_iterations += 1
        if loop_state.unproductive_iterations >= _MAX_UNPRODUCTIVE_ITERATIONS:
            loop_state.stuck = True
            loop_state.stuck_message = (
                "I kept retrying the same approach without making progress. "
                "Please narrow the request or tell me what to do differently."
            )


_EMAIL_LARGE_INBOX = re.compile(
    r"\b(too many|very large|large number|limit|timeout|overflow)\b",
    re.IGNORECASE,
)
_MCP_DOWN = re.compile(
    r"\b(unreachable|connection refused|timed out|timeout|502|503|504)\b",
    re.IGNORECASE,
)


def _default_recovery_hints(name_lower: str, lowered: str) -> list[str]:
    """Return the shipped, hardcoded recovery hints for a failed tool result."""
    hints: list[str] = []

    if "mail" in name_lower or "imap" in name_lower or "email" in lowered:
        if _EMAIL_LARGE_INBOX.search(lowered):
            hints.append(
                "Search unread messages only with a small limit (e.g. 10) via "
                "`mail_mcp__imap_search_messages` instead of listing the full inbox."
            )
        hints.append(
            "Prefer `mail_mcp__imap_mailbox_status` for unseen count, then "
            "`mail_mcp__imap_search_messages` with mailbox INBOX and "
            "unread_only=true before fetching individual messages."
        )

    if "news" in name_lower and "curate" not in name_lower:
        hints.append(
            "For headlines, call mcp_news__news_curate directly with no "
            "arguments ({}) before trying other news tools."
        )

    if _MCP_DOWN.search(lowered):
        hints.append(
            "MCP may be offline. Tell the user to check MCP proxy connectivity "
            "in HA Agent Settings."
        )

    if "search_entities" in name_lower and re.search(
        r"unknown tool|not found|unavailable",
        lowered,
    ):
        hints.append(
            "home_assistant__ha_search_entities is unavailable. Skip entity "
            "search. Use an EXPOSED ENTITIES shortcut with "
            "home_assistant__ha_call_service (domain, service, entity_id) "
            "instead."
        )

    if "ha_call_service" in name_lower and "domain" in lowered:
        hints.append(
            "Include domain, service, and entity_id in ha_call_service arguments. "
            "Derive domain from the entity_id prefix (light.example -> light)."
        )

    missing_field = re.search(r"missing field ['\"]?(\w+)", lowered)
    if missing_field:
        field_name = missing_field.group(1)
        hints.append(
            f"Re-call with required argument `{field_name}`. "
            "For email IMAP tools use mailbox INBOX unless the user specified "
            "Junk or another folder."
        )

    return hints


def _rule_recovery_hints(
    rules: list[Any],
    name_lower: str,
    lowered: str,
) -> list[str]:
    """Return hint bodies from editable rules that match a failed result.

    A rule matches when its (optional) tool-name substring is contained in the
    tool name and its (optional) error pattern is found in the error text. An
    empty substring/pattern is treated as a wildcard. Rules are duck-typed and
    expose ``enabled``, ``tool_substring``, ``error_pattern``, and ``body``.
    """
    hints: list[str] = []
    for rule in rules:
        if not getattr(rule, "enabled", True):
            continue
        substring = (getattr(rule, "tool_substring", "") or "").strip().lower()
        if substring and substring not in name_lower:
            continue
        pattern = (getattr(rule, "error_pattern", "") or "").strip()
        if pattern:
            try:
                if not re.search(pattern, lowered, re.IGNORECASE):
                    continue
            except re.error:
                if pattern.lower() not in lowered:
                    continue
        body = (getattr(rule, "body", "") or "").strip()
        if body:
            hints.append(body)
    return hints


def enrich_tool_output(
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
    *,
    rules: list[Any] | None = None,
) -> str:
    """Append recovery hints to failed tool output.

    When ``rules`` is supplied (UI-editable recovery hints), they replace the
    shipped hardcoded logic. When ``rules`` is ``None`` (store unavailable),
    the deterministic shipped defaults are used.
    """
    if not output.startswith("Tool error:"):
        return output

    lowered = output.lower()
    name_lower = tool_name.lower()
    if rules is None:
        hints = _default_recovery_hints(name_lower, lowered)
    else:
        hints = _rule_recovery_hints(rules, name_lower, lowered)

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
    hint_rules: list[Any] | None = None,
) -> str:
    """Apply error enrichment and optional HA verification to tool output."""
    from .tools import compact_tool_output

    output = compact_tool_output(tool_name, output)
    enriched = enrich_tool_output(tool_name, arguments, output, rules=hint_rules)
    if hass is None:
        return enriched

    if note := verify_ha_service(hass, tool_name, arguments, enriched):
        if loop_state is not None:
            loop_state.verification_notes.append(note)
        return f"{enriched}\n\n{note}"
    return enriched

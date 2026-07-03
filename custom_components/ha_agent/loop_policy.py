"""Agent loop policies: verification, error recovery, and stuck detection."""

from __future__ import annotations

import json
import re
from contextlib import suppress
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
    plan_step_notes: list[str] = field(default_factory=list)
    plan_current_step_index: int | None = None
    plan_completed_tools: list[str] = field(default_factory=list)
    skill_plan_override: bool = False
    skill_plan_override_reason: str = ""
    override_intent: str = ""
    empty_responses: int = 0
    mcp_guidance: list[str] = field(default_factory=list)
    include_full_tool_catalog: bool = False
    pagination_pending: dict[str, Any] = field(default_factory=dict)
    preserve_stream_ui: bool = False
    last_draft_answer: str = ""
    override_block_count: int = 0
    mcp_tool_catalog: dict[str, dict[str, str]] = field(default_factory=dict)


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
_MARK_READ_INTENT = re.compile(
    r"\b(?:"
    r"mark(?:\s+\w+){0,12}\s+(?:as\s+)?read|"
    r"flag(?:\s+\w+){0,12}\s+(?:as\s+)?read|"
    r"set(?:\s+\w+){0,12}\s+(?:as\s+)?read"
    r")\b",
    re.IGNORECASE,
)
_MARK_READ_TOOL_MARKERS = (
    "mark_read",
    "bulk_update",
    "update_flags",
    "set_flags",
    "store_flags",
)
_PLAN_TERMINAL_STATUSES = frozenset({"done", "omitted"})
_OMIT_TOOL_MARKER = re.compile(
    r"OMIT(?:TED)?(?::|\s+step\s+\d+\s*[:\-])?\s*"
    r"`?([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)`?"
    r"(?:\s*[—\-:]\s*(.+))?",
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
        if status not in _PLAN_TERMINAL_STATUSES:
            return index
    return None


def _plan_status_label(status: str) -> str:
    return {
        "pending": "[ ]",
        "done": "[x]",
        "needs_work": "[!]",
        "omitted": "[~]",
    }.get(status, "[ ]")


def omit_plan_step(loop_state: LoopState, index: int, reason: str) -> None:
    """Mark a plan step deliberately omitted with a short reason."""
    if index < 0 or index >= len(loop_state.plan_step_statuses):
        return
    if loop_state.plan_step_statuses[index] in _PLAN_TERMINAL_STATUSES:
        return
    loop_state.plan_step_statuses[index] = "omitted"
    if index < len(loop_state.plan_step_notes):
        loop_state.plan_step_notes[index] = reason.strip()[:200]
    else:
        loop_state.plan_step_notes.extend(
            [""] * (index - len(loop_state.plan_step_notes) + 1)
        )
        loop_state.plan_step_notes[index] = reason.strip()[:200]
    loop_state.plan_current_step_index = _next_incomplete_plan_step(loop_state)


def maybe_omit_plan_steps_from_reasoning(
    loop_state: LoopState,
    reasoning: str,
) -> None:
    """Apply explicit OMIT markers from model reasoning to the active plan."""
    if loop_state.skill_plan_override or not loop_state.plan_steps:
        return
    text = reasoning.strip()
    if not text:
        return
    for match in _OMIT_TOOL_MARKER.finditer(text):
        tool_name = match.group(1)
        reason = (match.group(2) or "Declared omitted in reasoning.").strip()
        for index, step in enumerate(loop_state.plan_steps):
            plan_tool = str(step.get("toolName", ""))
            if plan_tool and _tool_names_match(plan_tool, tool_name):
                omit_plan_step(loop_state, index, reason)


def reconcile_plan_after_tools(loop_state: LoopState) -> None:
    """Mark earlier pending steps omitted when a later planned step succeeded."""
    if loop_state.skill_plan_override or not loop_state.plan_steps:
        return
    for index, status in enumerate(loop_state.plan_step_statuses):
        if status != "done":
            continue
        step_name = str(loop_state.plan_steps[index].get("toolName", "step"))
        for prior in range(index):
            if loop_state.plan_step_statuses[prior] == "pending":
                omit_plan_step(
                    loop_state,
                    prior,
                    f"Superseded by successful {step_name} (step {index + 1}).",
                )


def reconcile_plan_before_answer(loop_state: LoopState) -> None:
    """Mark remaining pending steps omitted when the model is answering."""
    if loop_state.skill_plan_override or not loop_state.plan_steps:
        return
    for index, status in enumerate(loop_state.plan_step_statuses):
        if status != "pending":
            continue
        step_name = str(loop_state.plan_steps[index].get("toolName", "step"))
        omit_plan_step(
            loop_state,
            index,
            f"Not required to answer user goal (step {index + 1}: {step_name}).",
        )


def _mark_read_tools_equivalent(plan_tool: str, actual_tool: str) -> bool:
    plan_lower = plan_tool.lower()
    actual_lower = actual_tool.lower()
    markers = ("bulk_update", "mark_read", "update_flags")
    return any(marker in plan_lower for marker in markers) and any(
        marker in actual_lower for marker in markers
    )


def _match_plan_step_index(loop_state: LoopState, tool_name: str) -> int | None:
    for index, step in enumerate(loop_state.plan_steps):
        plan_tool = str(step.get("toolName", ""))
        if not plan_tool:
            continue
        matches = _tool_names_match(plan_tool, tool_name)
        matches = matches or _mark_read_tools_equivalent(plan_tool, tool_name)
        if not matches:
            continue
        status = loop_state.plan_step_statuses[index]
        if status in {"pending", "needs_work"}:
            return index
    for index, step in enumerate(loop_state.plan_steps):
        plan_tool = str(step.get("toolName", ""))
        if plan_tool and (
            _tool_names_match(plan_tool, tool_name)
            or _mark_read_tools_equivalent(plan_tool, tool_name)
        ):
            return index
    return None


def user_requests_skill_override(user_text: str) -> bool:
    """Return True when the user explicitly asks to bypass the active skill."""
    return bool(_USER_SKILL_OVERRIDE.search(user_text.strip()))


def _skill_supports_mark_read(skill: Any) -> bool:
    """Return True when a skill's workflow includes mark-as-read tooling."""
    body = str(getattr(skill, "body", "") or "").lower()
    tool_steps = getattr(skill, "tool_steps", None) or []
    blob = body
    for step in tool_steps:
        if isinstance(step, dict):
            blob += " " + str(step.get("toolName", "")).lower()
    return any(marker in blob for marker in _MARK_READ_TOOL_MARKERS)


def skill_goal_mismatch_reason(user_text: str, skill: Any) -> str | None:
    """Return a suspend reason when the user goal exceeds the matched skill."""
    if not _MARK_READ_INTENT.search(user_text.strip()):
        return None
    if _skill_supports_mark_read(skill):
        return None
    title = str(getattr(skill, "title", "") or "Active skill").strip()
    return (
        f"User asked to mark mail read; {title} only checks and reads unread mail."
    )


def _plan_supports_mark_read(plan_steps: list[dict[str, Any]]) -> bool:
    blob = " ".join(str(step.get("toolName", "")) for step in plan_steps).lower()
    return any(marker in blob for marker in _MARK_READ_TOOL_MARKERS)


def active_plan_goal_mismatch(
    user_text: str,
    plan_steps: list[dict[str, Any]],
) -> str | None:
    """Return suspend reason when the seeded route/skill plan cannot reach the goal."""
    if not plan_steps:
        return None
    if not _MARK_READ_INTENT.search(user_text.strip()):
        return None
    if _plan_supports_mark_read(plan_steps):
        return None
    return (
        "User asked to mark mail read; active plan only covers checking or "
        "reading unread mail."
    )


def resolve_skill_plan_conflict(
    user_text: str,
    *,
    matched_skills: list[Any],
    plan_steps: list[dict[str, Any]],
) -> str | None:
    """Return a suspend reason when the user goal exceeds matched skills or the plan."""
    for skill in matched_skills:
        if getattr(skill, "is_builtin", False):
            continue
        reason = skill_goal_mismatch_reason(user_text, skill)
        if reason:
            return reason
    return active_plan_goal_mismatch(user_text, plan_steps)


_EXPLORATION_GUIDANCE = (
    "Discover MCP tools for the user's goal (searchToolsForDomain or searchTool), "
    "then call the best match using arguments from discovery output and earlier "
    "tool results. Do not repeat a mismatched workflow."
)
_ROUTE_EXPLORATION_HINTS: dict[str, str] = {
    "email": (
        "Discover MCP tools for the goal, then adhere strictly to each tool's "
        "MCP description and serverLlmContext when calling it."
    ),
}
_OVERRIDE_EXPLORATION_PLANS: dict[str, list[dict[str, Any]]] = {
    "mark_read": [
        {"toolName": "mail_mcp__imap_search_messages"},
        {"toolName": "mail_mcp__imap_bulk_update_flags"},
    ],
}


def override_intent_key(user_text: str) -> str | None:
    """Return a coarse override intent label for exploration plan seeding."""
    if _MARK_READ_INTENT.search(user_text.strip()):
        return "mark_read"
    return None


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
    loop_state.plan_step_notes = []
    loop_state.plan_current_step_index = None
    loop_state.override_intent = override_intent_key(loop_state.plan_goal) or ""
    loop_state.mcp_guidance.insert(
        0,
        f"SKILL PLAN SUSPENDED — {reason.strip()[:200]}. {_EXPLORATION_GUIDANCE}",
    )
    route_hint = _ROUTE_EXPLORATION_HINTS.get(loop_state.plan_route)
    if route_hint and route_hint not in loop_state.mcp_guidance:
        loop_state.mcp_guidance.insert(1, route_hint)
    exploration_steps = _OVERRIDE_EXPLORATION_PLANS.get(loop_state.override_intent)
    if exploration_steps:
        loop_state.plan_steps = list(exploration_steps)
        loop_state.plan_step_statuses = ["pending"] * len(exploration_steps)
        loop_state.plan_step_notes = [""] * len(exploration_steps)
        loop_state.plan_current_step_index = 0
        loop_state.mcp_guidance.insert(
            2,
            (
                "Override exploration plan seeded — follow plan steps in order. "
                "Use discovery only if a step tool is missing from the catalog."
            ),
        )


def should_block_reasoning_execution_mismatch(loop_state: LoopState) -> bool:
    """Return True when reasoning/tool mismatch checks should block execution."""
    return not loop_state.skill_plan_override and skill_plan_blocks_discovery(
        loop_state
    )


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
    if loop_state.skill_plan_override:
        return bool(loop_state.plan_steps) and any(
            status == "done" for status in loop_state.plan_step_statuses
        )
    return (
        bool(loop_state.plan_steps)
        and len(loop_state.plan_steps) >= 2
        and bool(loop_state.plan_skill_title)
    )


def redundant_override_tool_block(
    loop_state: LoopState,
    tool_name: str,
) -> str | None:
    """Block repeat discovery/search when an override exploration plan advanced."""
    if not loop_state.skill_plan_override or not loop_state.plan_steps:
        return None
    if _pagination_allows_repeat(loop_state, tool_name):
        return None
    from .tools import is_discovery_tool_name

    if is_discovery_tool_name(tool_name) and any(
        status == "done" for status in loop_state.plan_step_statuses
    ):
        next_index = _next_incomplete_plan_step(loop_state)
        if next_index is not None:
            next_tool = str(
                loop_state.plan_steps[next_index].get("toolName", "tool")
            )
            return (
                "Tool error: Discovery already completed the information-gathering "
                f"step. Call `{next_tool}` next using prior tool output. "
                "Do not repeat discovery."
            )
    for index, step in enumerate(loop_state.plan_steps):
        plan_tool = str(step.get("toolName", "")).lower()
        if not plan_tool or not _tool_names_match(plan_tool, tool_name):
            continue
        if (
            index < len(loop_state.plan_step_statuses)
            and loop_state.plan_step_statuses[index] == "done"
        ):
            if _pagination_allows_repeat(loop_state, tool_name):
                continue
            if all(
                status in _PLAN_TERMINAL_STATUSES
                for status in loop_state.plan_step_statuses
            ):
                return (
                    "Tool error: All override plan steps are complete. "
                    "STOP calling tools and write the final answer to the user "
                    "using the prior tool results."
                )
            next_index = _next_incomplete_plan_step(loop_state)
            if next_index is not None and next_index != index:
                next_tool = str(
                    loop_state.plan_steps[next_index].get("toolName", "tool")
                )
                return (
                    f"Tool error: `{step.get('toolName', tool_name)}` already "
                    f"succeeded. Call `{next_tool}` next or answer the user if "
                    "all steps are done."
                )
            return (
                "Tool error: This plan step already succeeded. STOP calling "
                "tools and answer the user from prior results."
            )
    return None


def _catalog_tool_key(tool_name: str) -> str:
    return tool_name.strip()


def _lookup_catalog_entry(
    loop_state: LoopState,
    tool_name: str,
) -> dict[str, str]:
    key = _catalog_tool_key(tool_name)
    if key in loop_state.mcp_tool_catalog:
        return loop_state.mcp_tool_catalog[key]
    for stored, entry in loop_state.mcp_tool_catalog.items():
        if _tool_names_match(stored, tool_name):
            return entry
    return {}


def _parameters_summary(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return ""
    lines: list[str] = []
    required = schema.get("required")
    props = schema.get("properties")
    if isinstance(required, list) and required:
        lines.append("Required: " + ", ".join(str(name) for name in required))
    if isinstance(props, dict):
        keys = list(required) if isinstance(required, list) else list(props.keys())
        for name in keys[:10]:
            prop = props.get(name)
            if not isinstance(prop, dict):
                continue
            desc = prop.get("description")
            if isinstance(desc, str) and desc.strip():
                lines.append(f"- {name}: {desc.strip()[:160]}")
    return "\n".join(lines)


def cache_mcp_tool_catalog_entry(
    loop_state: LoopState,
    tool_name: str,
    *,
    description: str = "",
    server_llm_context: str = "",
    parameters: str = "",
) -> None:
    """Store MCP metadata for one tool so later guidance can cite it."""
    key = _catalog_tool_key(tool_name)
    if not key:
        return
    entry = dict(loop_state.mcp_tool_catalog.get(key, {}))
    if description.strip():
        entry["description"] = description.strip()[:800]
    if server_llm_context.strip():
        entry["serverLlmContext"] = server_llm_context.strip()[:_MAX_MCP_GUIDANCE_CHARS]
    if parameters.strip():
        entry["parameters"] = parameters.strip()[:800]
    loop_state.mcp_tool_catalog[key] = entry


def cache_mcp_tools_from_schemas(
    loop_state: LoopState,
    llm_tools: list[dict[str, Any]],
) -> None:
    """Seed the MCP tool catalog from session tools passed to the LLM."""
    for tool in llm_tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        params = function.get("parameters")
        cache_mcp_tool_catalog_entry(
            loop_state,
            name,
            description=str(function.get("description") or ""),
            parameters=_parameters_summary(params if isinstance(params, dict) else {}),
        )


def _discovery_tool_entries(output: str) -> list[dict[str, Any]]:
    if output.startswith("Tool error:"):
        return []
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("tools", "results", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if data.get("toolName") or data.get("name"):
            return [data]
    return []


def cache_discovery_tool_catalog(loop_state: LoopState, output: str) -> None:
    """Cache description and serverLlmContext from discovery tool output."""
    for entry in _discovery_tool_entries(output):
        name = entry.get("toolName") or entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        schema = entry.get("inputSchema")
        cache_mcp_tool_catalog_entry(
            loop_state,
            name,
            description=str(entry.get("description") or ""),
            server_llm_context=str(entry.get("serverLlmContext") or ""),
            parameters=_parameters_summary(schema if isinstance(schema, dict) else {}),
        )


def build_mcp_tool_adherence_hint(
    loop_state: LoopState,
    tool_name: str,
    *,
    lead_in: str = "",
) -> str:
    """Build guidance that cites MCP metadata instead of hard-coded arguments."""
    entry = _lookup_catalog_entry(loop_state, tool_name)
    parts: list[str] = []
    if lead_in.strip():
        parts.append(lead_in.strip())
    parts.append(f"Adhere strictly to the MCP tool definition for `{tool_name}`:")
    if entry.get("description"):
        parts.append(entry["description"])
    if entry.get("serverLlmContext"):
        parts.append(entry["serverLlmContext"])
    if entry.get("parameters"):
        parts.append(entry["parameters"])
    if len(parts) == 1:
        parts.append(
            "Discover this tool with searchTool or searchToolsForDomain to load "
            "its MCP description, required parameters, and serverLlmContext "
            "before calling it."
        )
    return "\n".join(parts)


def _next_plan_tool_name(loop_state: LoopState) -> str | None:
    next_index = _next_incomplete_plan_step(loop_state)
    if next_index is None or next_index >= len(loop_state.plan_steps):
        return None
    name = loop_state.plan_steps[next_index].get("toolName")
    return str(name).strip() if name else None


def _inject_next_tool_adherence(loop_state: LoopState, *, lead_in: str) -> None:
    next_tool = _next_plan_tool_name(loop_state)
    if not next_tool:
        hint = f"{lead_in} If the goal is already satisfied, answer the user now."
        loop_state.mcp_guidance.insert(0, hint)
        return
    hint = build_mcp_tool_adherence_hint(loop_state, next_tool, lead_in=lead_in)
    if hint not in loop_state.mcp_guidance:
        loop_state.mcp_guidance.insert(0, hint)


def record_override_block_guidance(
    loop_state: LoopState,
    tool_name: str,
    block_message: str,
) -> None:
    """Inject actionable guidance when a repeat tool call is blocked by the plan."""
    loop_state.iteration_had_duplicate_block = True
    loop_state.override_block_count += 1
    cleaned = block_message.removeprefix("Tool error:").strip()
    hint = (
        "BLOCKED REPEAT TOOL — the plan already advanced past this step:\n"
        f"{cleaned}\n"
        f"Do not call `{tool_name}` again. {describe_plan_next_action(loop_state)}"
    )
    if loop_state.override_block_count >= 3:
        hint += (
            "\nYou have retried blocked tools multiple times. STOP calling tools "
            "and write the final answer to the user using prior tool results."
        )
    if hint not in loop_state.mcp_guidance:
        loop_state.mcp_guidance.insert(0, hint)


    if hint not in loop_state.mcp_guidance:
        loop_state.mcp_guidance.insert(0, hint)
    next_tool = _next_plan_tool_name(loop_state)
    if next_tool:
        adherence = build_mcp_tool_adherence_hint(
            loop_state,
            next_tool,
            lead_in="Required next plan tool:",
        )
        if adherence not in loop_state.mcp_guidance:
            loop_state.mcp_guidance.insert(0, adherence)


def analyze_search_tool_result(
    loop_state: LoopState,
    tool_name: str,
    output: str,
    arguments: dict[str, Any],
) -> None:
    """Summarize list/search tool output and point at MCP metadata for the next step."""
    if output.startswith("Tool error:"):
        return
    data = _parse_tool_result_json(output)
    if not data:
        return

    entries = data.get("messages") or data.get("items") or data.get("results") or []
    if not isinstance(entries, list):
        return

    filtered = _coerce_bool(
        arguments.get("unread_only") or arguments.get("unreadOnly")
    )
    has_more = _coerce_bool(data.get("hasMore") or data.get("has_more"))

    if not entries:
        summary = "SEARCH RESULT: the query returned no items."
        if filtered:
            summary += " Filters still apply to any follow-up search or update call."
        _inject_next_tool_adherence(loop_state, lead_in=summary)
        return

    already_handled = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        flags = entry.get("flags") or entry.get("Flags") or []
        if isinstance(flags, str):
            flag_text = flags
        elif isinstance(flags, list):
            flag_text = " ".join(str(flag) for flag in flags)
        else:
            flag_text = ""
        if "\\Seen" in flag_text or r"\Seen" in flag_text:
            already_handled += 1

    pending = len(entries) - already_handled
    summary = (
        f"SEARCH RESULT: returned {len(entries)} item(s)"
        f"{f' with {pending} not yet marked read' if pending else ''}."
    )
    if filtered and not pending and already_handled == len(entries):
        if has_more:
            summary += (
                " This page only contains already-processed items; paginate using "
                "metadata from the tool result if more pages exist."
            )
        else:
            summary += " No remaining items match the active filters on this page."
    elif has_more:
        summary += " More pages are available per the tool result metadata."

    _inject_next_tool_adherence(loop_state, lead_in=summary)


def analyze_imap_search_result(
    loop_state: LoopState,
    tool_name: str,
    output: str,
    arguments: dict[str, Any],
) -> None:
    """Backward-compatible alias for search/list tool analysis."""
    if "search" not in tool_name.lower() and "list" not in tool_name.lower():
        return
    analyze_search_tool_result(loop_state, tool_name, output, arguments)


def guide_after_override_tool_result(
    loop_state: LoopState,
    tool_name: str,
    *,
    succeeded: bool,
) -> None:
    """Inject next-step hints after successful override-plan tool calls."""
    if not loop_state.skill_plan_override or not succeeded:
        return
    if loop_state.plan_steps and loop_state.plan_step_statuses and all(
        status in _PLAN_TERMINAL_STATUSES for status in loop_state.plan_step_statuses
    ):
        loop_state.mcp_guidance.insert(
            0,
            (
                "Plan steps are complete. STOP calling tools and answer the user "
                "using prior tool results."
            ),
        )
        return
    next_tool = _next_plan_tool_name(loop_state)
    if next_tool:
        hint = build_mcp_tool_adherence_hint(
            loop_state,
            next_tool,
            lead_in="Previous plan step succeeded.",
        )
        if hint not in loop_state.mcp_guidance:
            loop_state.mcp_guidance.insert(0, hint)


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
    loop_state.plan_step_notes = [""] * len(steps)
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
        if step_index < len(loop_state.plan_step_notes):
            loop_state.plan_step_notes[step_index] = ""
        loop_state.plan_current_step_index = _next_incomplete_plan_step(loop_state)
        reconcile_plan_after_tools(loop_state)
        guide_after_override_tool_result(loop_state, tool_name, succeeded=True)
        return

    if loop_state.plan_step_statuses[step_index] == "done":
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
    if loop_state.plan_steps and loop_state.plan_step_statuses and all(
        status in _PLAN_TERMINAL_STATUSES for status in loop_state.plan_step_statuses
    ):
        return (
            "All planned steps are done or deliberately omitted. STOP calling "
            "tools and write the final answer to the user now using the tool "
            "results above."
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

    lines = [
        "AGENT PLAN PROGRESS (internal — not from the user):",
        f"Goal: {loop_state.plan_goal}",
    ]
    if loop_state.skill_plan_override:
        reason = loop_state.skill_plan_override_reason or (
            "Active skill does not fit the user's goal."
        )
        lines.append(f"Skill workflow suspended — {reason}")
        if loop_state.plan_steps:
            lines.append("Override exploration plan:")
        else:
            lines.append(
                "No concrete override steps seeded — use discovery and tools "
                "as needed."
            )
    elif loop_state.plan_skill_title:
        lines.append(f"Workflow skill: {loop_state.plan_skill_title}")

    if loop_state.last_draft_answer:
        lines.append(
            "Previous answer attempt (verifier rejected — build on prior work, "
            "do not restart from scratch):"
        )
        lines.append(loop_state.last_draft_answer[:800])

    if loop_state.plan_steps:
        lines.append("Plan steps:")
        for index, step in enumerate(loop_state.plan_steps):
            name = str(step.get("toolName", "step"))
            status = (
                loop_state.plan_step_statuses[index]
                if index < len(loop_state.plan_step_statuses)
                else "pending"
            )
            marker = _plan_status_label(status)
            focus = ""
            if (
                loop_state.plan_current_step_index == index
                and status not in _PLAN_TERMINAL_STATUSES
            ):
                focus = "  <-- focus here"
            note = (
                loop_state.plan_step_notes[index].strip()
                if index < len(loop_state.plan_step_notes)
                else ""
            )
            suffix = f" — omitted: {note}" if status == "omitted" and note else ""
            lines.append(f"{index + 1}. {marker} {name}{suffix}{focus}")
        lines.append(
            "Follow steps in order. To skip a remaining step deliberately, "
            "include OMIT: <toolName> — <reason> in your reasoning before "
            "continuing or answering."
        )
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


def mark_iteration_after_tools(loop_state: LoopState) -> None:
    """Prepare the next loop iteration after tool execution."""
    loop_state.preserve_stream_ui = False


def mark_iteration_preserve_stream(
    loop_state: LoopState,
    *,
    draft_answer: str = "",
) -> None:
    """Prepare the next loop iteration without clearing streamed UI content."""
    loop_state.preserve_stream_ui = True
    if draft_answer.strip():
        loop_state.last_draft_answer = draft_answer.strip()[:2000]


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
    entry = {
        "role": INTERNAL_GUIDANCE_ROLE,
        "content": "\n\n".join(parts),
    }
    if messages and messages[-1].get("role") == "user":
        messages.insert(len(messages) - 1, entry)
    else:
        messages.append(entry)


def extract_mcp_guidance(tool_name: str, output: str) -> list[str]:
    """Pull serverLlmContext guidance from a discovery tool result."""
    if output.startswith("Tool error:"):
        return []
    lowered = tool_name.lower()
    if "searchtool" not in lowered:
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
    tool_names: list[str] = []
    for entry in entries:
        context = entry.get("serverLlmContext")
        if isinstance(context, str) and context.strip():
            guidance.append(context.strip()[:_MAX_MCP_GUIDANCE_CHARS])
        name = entry.get("toolName") or entry.get("name")
        if isinstance(name, str) and name.strip():
            tool_names.append(name.strip())
    if tool_names:
        preview = ", ".join(list(dict.fromkeys(tool_names))[:8])
        guidance.append(f"Discovered tools: {preview}")
    return list(dict.fromkeys(guidance))


def record_mcp_guidance(
    loop_state: LoopState,
    tool_name: str,
    output: str,
) -> None:
    """Stash discovery guidance for injection into the next loop step."""
    cache_discovery_tool_catalog(loop_state, output)
    for hint in extract_mcp_guidance(tool_name, output):
        if hint not in loop_state.mcp_guidance:
            loop_state.mcp_guidance.append(hint)


def _parse_tool_result_json(output: str) -> dict[str, Any] | None:
    if output.startswith("Tool error:"):
        return None
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _pagination_fields(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    nested = data.get("pagination")
    if isinstance(nested, dict):
        merged.update(nested)
    return merged


def extract_pagination_meta(
    output: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return continuation metadata when a tool result indicates more pages."""
    data = _parse_tool_result_json(output)
    if not data:
        return None

    fields = _pagination_fields(data)
    args = arguments or {}

    next_cursor = fields.get("nextCursor") or fields.get("next_cursor")
    if isinstance(next_cursor, str) and next_cursor.strip():
        return {"kind": "cursor", "cursor": next_cursor.strip()}

    next_page_token = (
        fields.get("nextPageToken")
        or fields.get("next_page_token")
        or fields.get("pageToken")
    )
    if isinstance(next_page_token, str) and next_page_token.strip():
        return {"kind": "page_token", "page_token": next_page_token.strip()}

    has_more = _coerce_bool(fields.get("hasMore") or fields.get("has_more"))
    if not has_more:
        return None

    offset = fields.get("offset", args.get("offset", 0))
    limit = fields.get("limit", args.get("limit"))
    try:
        next_offset = int(offset) + int(limit)
    except (TypeError, ValueError):
        try:
            next_offset = int(offset) + 1
        except (TypeError, ValueError):
            next_offset = 0
    meta: dict[str, Any] = {"kind": "offset", "offset": next_offset}
    if limit is not None:
        with suppress(TypeError, ValueError):
            meta["limit"] = int(limit)
    return meta


def build_pagination_hint(
    tool_name: str,
    meta: dict[str, Any],
) -> str:
    """Format one directive for fetching the next page of a paginated tool."""
    from .tools import format_pagination_hint

    return format_pagination_hint(tool_name, meta)


def _pagination_allows_repeat(loop_state: LoopState, tool_name: str) -> bool:
    pending = loop_state.pagination_pending
    pending_tool = str(pending.get("tool_name") or "")
    if not pending_tool:
        return False
    return _tool_names_match(pending_tool, tool_name)


def record_pagination_state(
    loop_state: LoopState,
    tool_name: str,
    output: str,
    arguments: dict[str, Any],
) -> None:
    """Track paginated tool results and inject next-page guidance."""
    meta = extract_pagination_meta(output, arguments)
    if meta:
        loop_state.pagination_pending = {"tool_name": tool_name, **meta}
        hint = build_pagination_hint(tool_name, meta)
        if hint and hint not in loop_state.mcp_guidance:
            loop_state.mcp_guidance.insert(0, hint)
        return

    pending_tool = str(loop_state.pagination_pending.get("tool_name") or "")
    if pending_tool and _tool_names_match(pending_tool, tool_name):
        loop_state.pagination_pending = {}


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

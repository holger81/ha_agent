"""Generic policies for skill learning: merge vs fork, draft enrichment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .body import derive_tool_steps_from_body
from .models import Skill, SkillDraft, TurnTrace

SaveMode = Literal["update", "fork"]

_DISCOVERY_TOOL = re.compile(
    r"(searchToolsForDomain|searchTool|tools/list|tools_list)",
    re.IGNORECASE,
)

_READ_EFFECT = re.compile(
    r"(?:^|_)(?:get|read|search|list|find|fetch|status|curate|describe|show|lookup|"
    r"query|summarize|check)(?:_|$)",
    re.IGNORECASE,
)
_MUTATE_EFFECT = re.compile(
    r"(?:^|_)(?:set|update|mark|flag|bulk|delete|remove|create|send|call|turn_on|"
    r"turn_off|write|apply|move|copy|add|clear|toggle|enable|disable)(?:_|$)",
    re.IGNORECASE,
)


def is_discovery_tool(tool_name: str) -> bool:
    """Return True for MCP discovery/list tools."""
    return bool(_DISCOVERY_TOOL.search(tool_name or ""))


@dataclass(frozen=True, slots=True)
class WorkflowDelta:
    """Difference between a parent skill workflow and a completed turn."""

    parent_tools: tuple[str, ...]
    executed_tools: tuple[str, ...]
    new_tools: tuple[str, ...]
    parent_effects: frozenset[str]
    executed_effects: frozenset[str]
    added_effects: frozenset[str]
    recommendation: SaveMode
    reason: str


def _tool_name(raw: str) -> str:
    return str(raw or "").strip()


def concrete_tool_names_from_steps(
    tool_steps: list[dict[str, Any]] | None,
) -> list[str]:
    """Return non-discovery tool names from skill tool_steps."""
    names: list[str] = []
    seen: set[str] = set()
    for step in tool_steps or []:
        if not isinstance(step, dict):
            continue
        name = _tool_name(step.get("toolName") or step.get("name"))
        if not name or is_discovery_tool(name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def successful_workflow_tools(trace: TurnTrace) -> list[str]:
    """Return ordered, de-duplicated successful non-discovery tools from a trace."""
    names: list[str] = []
    seen: set[str] = set()
    for call in trace.tool_calls:
        if not call.get("succeeded"):
            continue
        name = _tool_name(call.get("toolName") or call.get("name"))
        if not name or is_discovery_tool(name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def tool_effect_kind(tool_name: str) -> str:
    """Classify a tool as read, mutate, or other from its name."""
    tail = tool_name.split("__")[-1] if "__" in tool_name else tool_name
    if _MUTATE_EFFECT.search(tail):
        return "mutate"
    if _READ_EFFECT.search(tail):
        return "read"
    return "other"


def _effect_set(tool_names: list[str]) -> frozenset[str]:
    return frozenset(tool_effect_kind(name) for name in tool_names)


def trigger_overlap(left: list[str], right: list[str]) -> float:
    """Return Jaccard similarity of normalized trigger word sets."""
    def tokens(items: list[str]) -> set[str]:
        words: set[str] = set()
        for item in items:
            for word in re.findall(r"[a-z0-9]+", item.lower()):
                if len(word) > 2:
                    words.add(word)
        return words

    a = tokens(left)
    b = tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def draft_would_regress_parent(parent: Skill, draft: SkillDraft) -> bool:
    """Return True when applying draft would wipe a concrete parent workflow."""
    parent_tools = concrete_tool_names_from_steps(parent.tool_steps)
    if not parent_tools:
        return False
    draft_tools = list(draft.tool_steps)
    if not draft_tools:
        draft_tools = derive_tool_steps_from_body(draft.body)
    return not draft_tools


def analyze_workflow_delta(parent: Skill, trace: TurnTrace) -> WorkflowDelta:
    """Compare parent skill tools/effects to a successful override turn."""
    parent_tools = concrete_tool_names_from_steps(parent.tool_steps)
    executed_tools = successful_workflow_tools(trace)
    parent_set = set(parent_tools)
    executed_set = set(executed_tools)
    new_tools = tuple(name for name in executed_tools if name not in parent_set)
    parent_effects = _effect_set(parent_tools)
    executed_effects = _effect_set(executed_tools)
    added_effects = executed_effects - parent_effects

    recommendation: SaveMode = "fork"
    reason = "Default to a child skill after plan override."

    if not parent_tools:
        recommendation = "update"
        reason = "Parent skill has no concrete workflow yet; extend in place."
    elif not new_tools and executed_set <= parent_set:
        recommendation = "update"
        reason = "Turn reused only tools already in the parent skill."
    elif new_tools and parent_tools:
        recommendation = "fork"
        reason = (
            "Turn introduced new workflow tools not present in the parent skill: "
            + ", ".join(new_tools[:4])
            + ("…" if len(new_tools) > 4 else "")
        )
    if (
        "mutate" in added_effects
        and parent_effects
        and "mutate" not in parent_effects
    ):
        recommendation = "fork"
        reason = (
            "Turn adds mutating tools while the parent workflow is read/query-only."
        )

    return WorkflowDelta(
        parent_tools=tuple(parent_tools),
        executed_tools=tuple(executed_tools),
        new_tools=new_tools,
        parent_effects=parent_effects,
        executed_effects=executed_effects,
        added_effects=added_effects,
        recommendation=recommendation,
        reason=reason,
    )


def recommend_override_save_mode(
    parent: Skill,
    trace: TurnTrace,
    draft: SkillDraft,
) -> SaveMode:
    """Recommend update vs fork from workflow structure, not domain keywords."""
    delta = analyze_workflow_delta(parent, trace)
    if draft_would_regress_parent(parent, draft):
        return "fork"
    overlap = trigger_overlap(parent.triggers, draft.triggers)
    if (
        delta.recommendation == "update"
        and overlap < 0.15
        and delta.new_tools
    ):
        return "fork"
    return delta.recommendation


def tool_steps_from_trace(trace: TurnTrace) -> list[dict[str, Any]]:
    """Build tool_steps from the first successful call per non-discovery tool."""
    by_tool: dict[str, dict[str, Any]] = {}
    for call in trace.tool_calls:
        if not call.get("succeeded"):
            continue
        name = _tool_name(call.get("toolName") or call.get("name"))
        if not name or is_discovery_tool(name):
            continue
        args = call.get("arguments")
        if name not in by_tool and isinstance(args, dict):
            by_tool[name] = dict(args)
    return [
        {"toolName": name, "arguments": args}
        for name, args in by_tool.items()
    ]


def enrich_draft_from_trace(draft: SkillDraft, trace: TurnTrace) -> SkillDraft:
    """Fill missing tool_steps from successful trace tools."""
    steps = list(draft.tool_steps)
    if not steps:
        steps = derive_tool_steps_from_body(draft.body)
    if steps:
        return SkillDraft(
            title=draft.title,
            description=draft.description,
            triggers=draft.triggers,
            body=draft.body,
            tool_steps=steps,
            slots=draft.slots,
            preconditions=draft.preconditions,
            parent_id=draft.parent_id,
            route_scope=draft.route_scope,
        )
    trace_steps = tool_steps_from_trace(trace)
    if not trace_steps:
        return draft
    return SkillDraft(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=draft.body,
        tool_steps=trace_steps,
        slots=draft.slots,
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope,
    )


def merge_parent_skill_draft(
    parent: Skill,
    draft: SkillDraft,
    trace: TurnTrace,
) -> SkillDraft:
    """Extend a parent skill in place instead of replacing it wholesale."""
    enriched = enrich_draft_from_trace(draft, trace)
    parent_steps = list(parent.tool_steps)
    parent_tool_names = {
        _tool_name(step.get("toolName") or step.get("name"))
        for step in parent_steps
        if isinstance(step, dict)
    }
    merged_steps = list(parent_steps)
    for step in enriched.tool_steps:
        name = _tool_name(step.get("toolName") or step.get("name"))
        if name and name not in parent_tool_names:
            merged_steps.append(step)
            parent_tool_names.add(name)

    merged_triggers: list[str] = []
    seen_triggers: set[str] = set()
    for trigger in [*parent.triggers, *enriched.triggers]:
        key = trigger.strip().lower()
        if key and key not in seen_triggers:
            seen_triggers.add(key)
            merged_triggers.append(trigger.strip())

    body = parent.body.rstrip()
    addition = enriched.body.strip()
    if addition and addition not in body:
        body = f"{body}\n\n## Additional workflow\n\n{addition}"

    return SkillDraft(
        title=parent.title,
        description=parent.description or enriched.description,
        triggers=merged_triggers or enriched.triggers,
        body=body,
        tool_steps=merged_steps,
        slots=parent.slots or enriched.slots,
        preconditions=parent.preconditions or enriched.preconditions,
        parent_id=parent.parent_id,
        route_scope=parent.route_scope or enriched.route_scope,
    )


def resolve_override_observer_result(
    parent: Skill,
    trace: TurnTrace,
    result: Any,
) -> Any:
    """Apply generic merge/fork policy to an override observer result."""
    from .observer import SkillObserverResult

    if not result.learn or result.draft is None:
        return result

    mode = recommend_override_save_mode(parent, trace, result.draft)
    draft = result.draft

    if mode == "fork":
        if not draft.parent_id:
            draft = SkillDraft(
                title=draft.title,
                description=draft.description,
                triggers=draft.triggers,
                body=draft.body,
                tool_steps=draft.tool_steps,
                slots=draft.slots,
                preconditions=draft.preconditions,
                parent_id=parent.id,
                route_scope=draft.route_scope or parent.route_scope,
            )
        draft = enrich_draft_from_trace(draft, trace)
        return SkillObserverResult(
            learn=True,
            reason=f"{result.reason} ({analyze_workflow_delta(parent, trace).reason})",
            draft=draft,
            update_parent=False,
        )

    draft = merge_parent_skill_draft(parent, draft, trace)
    return SkillObserverResult(
        learn=True,
        reason=f"{result.reason} ({analyze_workflow_delta(parent, trace).reason})",
        draft=draft,
        update_parent=True,
    )

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
    if "mutate" in added_effects and parent_effects and "mutate" not in parent_effects:
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
    if delta.recommendation == "update" and overlap < 0.15 and delta.new_tools:
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
    return [{"toolName": name, "arguments": args} for name, args in by_tool.items()]


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


_MCP_TOOL_NAME = re.compile(
    r"\b([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)\b",
    re.IGNORECASE,
)
_VOLATILE_ARG_KEYS = frozenset(
    {
        "entity_id",
        "message_id",
        "message_ids",
        "uid",
        "uids",
        "id",
        "ids",
    }
)


def draft_has_executable_tools(draft: SkillDraft) -> bool:
    """Return True when draft tool_steps or body name concrete MCP tools."""
    names = concrete_tool_names_from_steps(draft.tool_steps)
    if names:
        return True
    return bool(_MCP_TOOL_NAME.search(draft.body or ""))


def _tool_steps_preferring_filters(trace: TurnTrace) -> list[dict[str, Any]]:
    """Like tool_steps_from_trace, but prefer search calls that used filters."""
    by_tool: dict[str, dict[str, Any]] = {}
    for call in trace.tool_calls:
        if not call.get("succeeded"):
            continue
        name = _tool_name(call.get("toolName") or call.get("name"))
        if not name or is_discovery_tool(name):
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        existing = by_tool.get(name)
        if existing is None:
            by_tool[name] = dict(args)
            continue
        # Prefer later args that include narrowing filters.
        if any(
            args.get(key)
            for key in ("domain_filter", "area_filter", "state_filter", "domain")
        ) and not any(
            existing.get(key)
            for key in ("domain_filter", "area_filter", "state_filter", "domain")
        ):
            by_tool[name] = dict(args)
    return [{"toolName": name, "arguments": args} for name, args in by_tool.items()]


def prepare_learned_draft(draft: SkillDraft, trace: TurnTrace) -> SkillDraft | None:
    """Ground a draft in successful trace tools or reject prose-only skills.

    Returns None when the turn has no reusable non-discovery workflow.
    """
    from .runtime import is_hard_won_workflow

    trace_steps = (
        _tool_steps_preferring_filters(trace)
        if is_hard_won_workflow(trace)
        else tool_steps_from_trace(trace)
    )
    if not trace_steps:
        return None

    grounded = SkillDraft(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=draft.body,
        tool_steps=trace_steps,
        slots=list(draft.slots),
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope or (str(trace.route).strip() or None),
    )
    grounded = _slotify_action_entity_ids(grounded, trace)
    grounded = _slotify_status_search_args(grounded)
    if not draft_has_executable_tools(grounded):
        return None
    return grounded


def _slotify_status_search_args(draft: SkillDraft) -> SkillDraft:
    """Replace concrete ha_search query strings with a {{query}} slot."""
    from .models import SkillSlot

    has_search_query = False
    new_steps: list[dict[str, Any]] = []
    for step in draft.tool_steps:
        if not isinstance(step, dict):
            continue
        step_copy = dict(step)
        name = str(step_copy.get("toolName") or "").lower()
        args = step_copy.get("arguments")
        if (
            isinstance(args, dict)
            and "ha_search" in name
            and isinstance(args.get("query"), str)
            and args.get("query")
            and "{{" not in args["query"]
        ):
            merged = dict(args)
            merged["query"] = "{{query}}"
            step_copy["arguments"] = merged
            has_search_query = True
        new_steps.append(step_copy)

    if not has_search_query:
        return draft

    slots = list(draft.slots)
    if not any(slot.name == "query" for slot in slots):
        slots.append(
            SkillSlot(
                name="query",
                description="Search term for the place, person, or device",
                source="user",
                default=None,
            )
        )
    body = draft.body or ""
    if "{{query}}" not in body and "ha_search" in body.lower():
        body = (
            body.rstrip()
            + "\n\nUse `query={{query}}` (and `domain_filter` when it narrows "
            "results) for the search step."
        )
    return SkillDraft(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=body,
        tool_steps=new_steps,
        slots=slots,
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope,
    )


def build_deterministic_hard_won_result(trace: TurnTrace) -> Any | None:
    """Distill a parameterized status/lookup skill after a costly success.

    Used when the LLM observer rejects a hard-won turn as one-off Q&A.
    """
    from .observer import SkillObserverResult
    from .runtime import is_hard_won_workflow, struggle_event_count

    if not is_hard_won_workflow(trace):
        return None
    steps = _tool_steps_preferring_filters(trace)
    if not steps:
        return None

    goal = (trace.user_text or "").strip()
    title = "Look up sensor or entity status"
    description = (
        "Parameterized status lookup distilled after a hard-won successful "
        f"turn ({struggle_event_count(trace)} struggle events)."
    )
    triggers = [
        phrase
        for phrase in (
            goal,
            "what is the temperature in {{query}}",
            "status of {{query}}",
            "look up {{query}} sensor",
        )
        if phrase
    ]
    body_lines = [
        "# Look up status",
        "",
        "When the user asks for a reading or status for a place/person/device:",
        "",
    ]
    for index, step in enumerate(steps, start=1):
        name = str(step.get("toolName") or "")
        body_lines.append(
            f"{index}. Call `{name}` using slot values and prior tool results."
        )
    body_lines.extend(
        [
            "",
            "Prefer a short `query={{query}}` plus `domain_filter` when searching.",
            "Pick the entity whose device_class / unit_of_measurement matches "
            "the asked reading (temperature ≠ voltage).",
            "Answer from tool results; do not invent values.",
        ]
    )
    draft = SkillDraft(
        title=title,
        description=description[:512],
        triggers=triggers[:8],
        body="\n".join(body_lines),
        tool_steps=steps,
        route_scope=str(trace.route).strip() or "action",
    )
    draft = _slotify_status_search_args(draft)
    draft = _slotify_action_entity_ids(draft, trace)
    return SkillObserverResult(
        learn=True,
        reason=(
            "Deterministic hard-won status distillation "
            f"({struggle_event_count(trace)} struggle events)."
        ),
        draft=draft,
    )


def _slotify_action_entity_ids(draft: SkillDraft, trace: TurnTrace) -> SkillDraft:
    """Prefer {{entity_id}} slots over baking one-off entity ids for action turns."""
    route = (trace.route or draft.route_scope or "").lower()
    if route not in {"action", "ha_action"} and not trace.controlled_entity_ids:
        return draft

    entity_ids = [
        str(eid).strip()
        for eid in trace.controlled_entity_ids
        if isinstance(eid, str) and eid.strip()
    ]
    if not entity_ids:
        for step in draft.tool_steps:
            args = step.get("arguments") if isinstance(step, dict) else None
            if not isinstance(args, dict):
                continue
            value = args.get("entity_id")
            if isinstance(value, str) and "." in value:
                entity_ids.append(value)
            elif isinstance(value, list):
                entity_ids.extend(
                    str(item) for item in value if isinstance(item, str) and "." in item
                )
    if not entity_ids:
        return draft

    slots = list(draft.slots)
    if not any(slot.name == "entity_id" for slot in slots):
        from .models import SkillSlot

        slots.append(
            SkillSlot(
                name="entity_id",
                description="Home Assistant entity to control",
                source="user",
                default=None,
            )
        )

    new_steps: list[dict[str, Any]] = []
    for step in draft.tool_steps:
        if not isinstance(step, dict):
            continue
        step_copy = dict(step)
        args = step_copy.get("arguments")
        if isinstance(args, dict) and "entity_id" in args:
            merged = dict(args)
            merged["entity_id"] = "{{entity_id}}"
            step_copy["arguments"] = merged
        new_steps.append(step_copy)

    return SkillDraft(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=draft.body,
        tool_steps=new_steps,
        slots=slots,
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope,
    )


def sanitize_repair_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop volatile one-off argument values from repair merges."""
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in _VOLATILE_ARG_KEYS:
            continue
        if isinstance(value, str) and re.fullmatch(
            r"[a-z_]+\.[a-z0-9_]+", value, re.IGNORECASE
        ):
            # Likely a concrete entity_id stored under another key.
            continue
        cleaned[key] = value
    return cleaned


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


def build_deterministic_override_result(
    parent: Skill,
    trace: TurnTrace,
) -> Any | None:
    """Build a child/update skill from trace tools when the LLM observer fails."""
    from .observer import SkillObserverResult

    steps = tool_steps_from_trace(trace)
    if not steps:
        return None

    goal = trace.user_text.strip()
    title = (goal[:64] or f"Workflow for {parent.title}").strip()
    description = (
        "Workflow distilled from a successful override turn when the active "
        f"skill did not cover: {goal[:240]}"
    )
    triggers = [goal] if goal else [title]
    body_lines = [
        f"When the user asks: {goal}",
        "",
        "Use this tool sequence (omit discovery unless tools are unknown):",
    ]
    for index, step in enumerate(steps, start=1):
        name = str(step.get("toolName") or "")
        body_lines.append(
            f"{index}. Call `{name}` with arguments from earlier tool results."
        )
    draft = SkillDraft(
        title=title,
        description=description[:512],
        triggers=triggers,
        body="\n".join(body_lines),
        tool_steps=steps,
        slots=list(parent.slots),
        parent_id=parent.id,
        route_scope=parent.route_scope,
    )
    observed = SkillObserverResult(
        learn=True,
        reason="Deterministic override distillation from successful tools.",
        draft=draft,
        update_parent=False,
    )
    return resolve_override_observer_result(parent, trace, observed)

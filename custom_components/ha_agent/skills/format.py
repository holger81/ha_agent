"""Format skills for LLM context injection."""

from __future__ import annotations

import json

from .models import Skill
from .params import bind_slot_value, default_slots_for_skill

_ROUTE_SKILL_PRIORITY = frozenset({"email", "news", "action"})

_ADAPT_GUIDANCE = (
    "ADAPT, DO NOT ABANDON: When the user goal is a variant of this workflow "
    "(different mailbox/folder, date range, entity, or limit), keep the same "
    "tool sequence and tool names. Only change argument values and {{slot}} "
    "placeholders to match the user request. Do not run discovery tools unless "
    "a planned step fails with recovery hints."
)


def format_skills_for_context(
    skills: list[Skill],
    *,
    route: str | None = None,
    slot_bindings: dict[str, str] | None = None,
) -> str:
    """Render matched skills as a compact ACTIVE SKILLS block."""
    if not skills:
        return ""

    bindings = slot_bindings or {}
    if route in _ROUTE_SKILL_PRIORITY:
        header = (
            "ACTIVE SKILLS (PRIORITY — selected for this turn): "
            "Follow the workflow below. "
        )
        header += _ADAPT_GUIDANCE + " "
        if any(skill.tool_steps for skill in skills):
            header += (
                "When tool steps are listed, run them in order. "
                "Fill {{slot}} placeholders from the user goal before calling tools. "
            )
        header += (
            "Do not repeat identical tool calls already shown in this conversation."
        )
    else:
        header = (
            "ACTIVE SKILLS (follow these workflows when applicable; "
            f"{_ADAPT_GUIDANCE})"
        )

    lines = [header]
    for skill in skills:
        lines.append(f"- [{skill.slug}] {skill.title}: {skill.description}")
        slots = default_slots_for_skill(skill)
        if slots:
            slot_desc = ", ".join(
                f"{s.name} ({s.description or s.source})" for s in slots
            )
            lines.append(f"  Slots: {slot_desc}")
        if bindings:
            lines.append(f"  Bound slots: {json.dumps(bindings, ensure_ascii=True)}")
        body_preview = bind_slot_value(skill.body.strip(), bindings)
        if body_preview:
            lines.append(f"  Workflow:\n{body_preview}")
        if skill.tool_steps:
            from .params import bind_tool_steps

            steps = bind_tool_steps(skill.tool_steps, bindings)
            steps_json = json.dumps(steps, ensure_ascii=True)
            lines.append(f"  Tool steps: {steps_json}")
            lines.append(
                "  Run each tool step once per turn. After the last step, confirm "
                "the outcome matches the workflow before answering the user."
            )
        elif body_preview:
            lines.append(
                "  Follow the workflow steps above; call the named tools in order."
            )
    return "\n".join(lines)


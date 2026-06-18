"""Format skills for LLM context injection."""

from __future__ import annotations

import json

from .models import Skill

_ROUTE_SKILL_PRIORITY = frozenset({"email", "news"})


def format_skills_for_context(
    skills: list[Skill],
    *,
    route: str | None = None,
) -> str:
    """Render matched skills as a compact ACTIVE SKILLS block."""
    if not skills:
        return ""

    if route in _ROUTE_SKILL_PRIORITY:
        header = (
            "ACTIVE SKILLS (PRIORITY): A matching skill applies to this turn. "
            "Follow its tool_steps in order before improvising other tools. "
            "Reuse toolName values and arguments exactly unless a step failed "
            "with recovery hints. Do not repeat identical tool calls already "
            "shown in this conversation."
        )
    else:
        header = (
            "ACTIVE SKILLS (follow these workflows when applicable; "
            "reuse entity_id values and tool patterns exactly):"
        )

    lines = [header]
    for skill in skills:
        lines.append(f"- [{skill.slug}] {skill.title}: {skill.description}")
        if skill.tool_steps:
            steps_json = json.dumps(skill.tool_steps, ensure_ascii=True)
            lines.append(f"  Tool steps: {steps_json}")
            lines.append(
                "  Run each tool step once per turn. After the last step, confirm the "
                "outcome matches the workflow before answering the user."
            )
        body_preview = skill.body.strip()
        if body_preview:
            lines.append(f"  Workflow:\n{body_preview}")
    return "\n".join(lines)

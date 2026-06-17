"""Format skills for LLM context injection."""

from __future__ import annotations

import json

from .models import Skill


def format_skills_for_context(skills: list[Skill]) -> str:
    """Render matched skills as a compact ACTIVE SKILLS block."""
    if not skills:
        return ""

    lines = [
        "ACTIVE SKILLS (follow these workflows when applicable; "
        "reuse entity_id values and tool patterns exactly):"
    ]
    for skill in skills:
        lines.append(f"- [{skill.slug}] {skill.title}: {skill.description}")
        if skill.tool_steps:
            steps_json = json.dumps(skill.tool_steps, ensure_ascii=True)
            lines.append(f"  Tool steps: {steps_json}")
            lines.append(
                "  Run tool_steps in order. After the last step, confirm the "
                "outcome matches the workflow before answering the user."
            )
        body_preview = skill.body.strip()
        if body_preview:
            lines.append(f"  Workflow:\n{body_preview}")
    return "\n".join(lines)

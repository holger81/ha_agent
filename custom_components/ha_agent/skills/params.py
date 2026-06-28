"""Skill slot binding and parameterized tool-step rendering."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import Skill, SkillSlot

_SLOT_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_slots_from_text(text: str) -> list[str]:
    """Return unique slot names referenced as {{name}} in text."""
    return list(dict.fromkeys(_SLOT_PATTERN.findall(text)))


def default_slots_for_skill(skill: Skill) -> list[SkillSlot]:
    """Build slot list from skill.slots or infer from body/tool_steps."""
    if skill.slots:
        return list(skill.slots)
    names: list[str] = []
    for name in extract_slots_from_text(skill.body):
        names.append(name)
    for step in skill.tool_steps:
        for value in step.values():
            if isinstance(value, str):
                for name in extract_slots_from_text(value):
                    if name not in names:
                        names.append(name)
    return [
        SkillSlot(
            name=name,
            description=f"User or context value for {name}",
            source="user",
        )
        for name in names
    ]


def bind_slot_value(text: str, bindings: dict[str, str]) -> str:
    """Replace {{slot}} placeholders with bound values."""
    if not bindings or "{{" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return bindings.get(key, match.group(0))

    return _SLOT_PATTERN.sub(repl, text)


def bind_tool_steps(
    steps: list[dict[str, Any]],
    bindings: dict[str, str],
) -> list[dict[str, Any]]:
    """Return tool steps with slot placeholders filled."""
    if not bindings:
        return [dict(step) for step in steps]
    bound: list[dict[str, Any]] = []
    for step in steps:
        new_step: dict[str, Any] = {}
        for key, value in step.items():
            if isinstance(value, str):
                new_step[key] = bind_slot_value(value, bindings)
            elif isinstance(value, dict):
                new_step[key] = json.loads(
                    bind_slot_value(json.dumps(value, ensure_ascii=True), bindings)
                )
            else:
                new_step[key] = value
        bound.append(new_step)
    return bound


def bindings_diverge_from_defaults(
    skill: Skill,
    bindings: dict[str, str],
) -> bool:
    """Return True when slot bindings differ materially from skill defaults."""
    if not bindings:
        return False
    defaults = {
        slot.name: (slot.default or "").strip().lower()
        for slot in default_slots_for_skill(skill)
    }
    for key, value in bindings.items():
        bound = str(value).strip().lower()
        if not bound:
            continue
        default = defaults.get(key, "")
        if default and bound != default:
            return True
        if not default and bound not in skill.body.lower():
            return True
    return False


async def infer_slot_bindings(
    llm: Any,
    backend: Any,
    *,
    user_text: str,
    skill: Skill,
    route: str | None = None,
) -> dict[str, str]:
    """Ask the router/classifier model to fill skill slots from the user goal."""
    slots = default_slots_for_skill(skill)
    if not slots:
        return {}

    from ..llm_client import LlmClient

    if not isinstance(llm, LlmClient):
        return {}

    slot_specs = [
        {"name": s.name, "description": s.description, "default": s.default or ""}
        for s in slots
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "Fill workflow parameter slots from the user's request.\n"
                'Return ONLY JSON: {"bindings": {"slot_name": "value", ...}}.\n'
                "Use empty string when a slot cannot be inferred."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "route": route or "chat",
                    "skill_title": skill.title,
                    "slots": slot_specs,
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception:
        return {}
    text = (result.content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    bindings = data.get("bindings") if isinstance(data, dict) else None
    if not isinstance(bindings, dict):
        return {}
    return {str(k): str(v) for k, v in bindings.items() if v is not None}

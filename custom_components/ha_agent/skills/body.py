"""Markdown-first skill workflow helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .defaults import apply_route_defaults_to_draft
from .models import Skill, SkillDraft
from .tool_names import canonicalize_skill, canonicalize_skill_draft

_TOOL_STEPS_FENCE = re.compile(
    r"```(?:json)?\s*tool[_\s-]*steps?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_JSON_ARRAY_FENCE = re.compile(
    r"```json\s*\n(\[[\s\S]*?\])\s*```",
    re.IGNORECASE,
)
_BACKTICK_TOOL = re.compile(
    r"`([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)`",
    re.IGNORECASE,
)
_BARE_TOOL = re.compile(
    r"\b([a-z][a-z0-9_]*__[a-z0-9_]+)\b",
    re.IGNORECASE,
)
# Tools that are commonly hallucinated or absent from MCP — omit from derivation.
_DEPRECATED_TOOL_TAILS = frozenset({"ha_search_entities"})


def _is_deprecated_tool_name(name: str) -> bool:
    tail = name.split("__")[-1].lower()
    return tail in _DEPRECATED_TOOL_TAILS


def _coerce_tool_steps(raw: Any) -> list[dict[str, Any]]:
    """Parse a JSON value into normalized tool step dicts."""
    if not isinstance(raw, list):
        return []
    steps: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("toolName") or item.get("name") or "").strip()
        if not name or _is_deprecated_tool_name(name):
            continue
        arguments = item.get("arguments")
        steps.append(
            {
                "toolName": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return steps


def extract_tool_steps_block(body: str) -> list[dict[str, Any]]:
    """Return tool steps from an optional ```tool_steps fenced block."""
    if not body.strip():
        return []
    match = _TOOL_STEPS_FENCE.search(body)
    if match:
        try:
            return _coerce_tool_steps(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            return []
    for match in _JSON_ARRAY_FENCE.finditer(body):
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        steps = _coerce_tool_steps(parsed)
        if steps:
            return steps
    return []


def derive_tool_steps_from_body(body: str) -> list[dict[str, Any]]:
    """Derive structured tool steps from markdown workflow text."""
    if not body.strip():
        return []

    if block_steps := extract_tool_steps_block(body):
        return block_steps

    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    for pattern in (_BACKTICK_TOOL, _BARE_TOOL):
        for match in pattern.finditer(body):
            name = match.group(1).strip()
            if not name or name in seen or _is_deprecated_tool_name(name):
                continue
            seen.add(name)
            steps.append({"toolName": name, "arguments": {}})
    return steps


def resolve_tool_steps(
    body: str,
    tool_steps: list[dict[str, Any]] | None,
    *,
    explicit_override: bool,
) -> list[dict[str, Any]]:
    """Choose explicit tool steps or derive them from the workflow body."""
    if explicit_override and tool_steps is not None:
        return _coerce_tool_steps(tool_steps)
    derived = derive_tool_steps_from_body(body)
    if derived:
        return derived
    return _coerce_tool_steps(tool_steps or [])


def normalize_skill_draft(
    draft: SkillDraft,
    *,
    explicit_tool_steps: bool = False,
) -> SkillDraft:
    """Ensure tool_steps reflect the markdown workflow when not overridden."""
    fixed, _ = canonicalize_skill_draft(draft)
    draft.title = fixed.title
    draft.description = fixed.description
    draft.triggers = fixed.triggers
    draft.body = fixed.body
    draft.tool_steps = fixed.tool_steps
    draft.slots = fixed.slots
    draft.preconditions = fixed.preconditions
    draft.parent_id = fixed.parent_id
    draft.route_scope = fixed.route_scope
    steps = resolve_tool_steps(
        draft.body,
        draft.tool_steps,
        explicit_override=explicit_tool_steps,
    )
    apply_route_defaults_to_draft(draft)
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


def normalize_skill(skill: Skill, *, explicit_tool_steps: bool = False) -> Skill:
    """Refresh a skill's tool_steps from its body unless explicitly overridden."""
    fixed, _ = canonicalize_skill(skill)
    skill.body = fixed.body
    skill.tool_steps = list(fixed.tool_steps)
    skill.tool_steps = resolve_tool_steps(
        skill.body,
        skill.tool_steps,
        explicit_override=explicit_tool_steps,
    )
    return skill

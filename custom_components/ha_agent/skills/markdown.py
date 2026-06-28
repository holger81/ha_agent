"""Parse and serialize HA Agent skills as markdown files with YAML frontmatter."""

from __future__ import annotations

import re
from typing import Any

from homeassistant.exceptions import HomeAssistantError

from .body import normalize_skill_draft
from .models import Skill, SkillDraft, SkillSlot

_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)

NEW_SKILL_MARKDOWN = """---
title: My Skill
description: What the agent should do when this skill matches
triggers:
  - example phrase
enabled: true
---

# Workflow

1. Describe steps in markdown.
2. Name tools in backticks with full MCP names, e.g. `mail_mcp__imap_mailbox_status`.
"""


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as err:
        raise HomeAssistantError(
            "YAML support is required to parse skill files"
        ) from err
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise HomeAssistantError("Skill frontmatter must be a YAML mapping")
    return loaded


def _dump_yaml(data: dict[str, Any]) -> str:
    import yaml  # type: ignore[import-untyped]

    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()


def _coerce_triggers(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _coerce_slots(raw: Any) -> list[SkillSlot]:
    if not isinstance(raw, list):
        return []
    slots: list[SkillSlot] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        slots.append(
            SkillSlot(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                source=str(item.get("source", "user")),
                default=(
                    str(item["default"])
                    if item.get("default") is not None
                    else None
                ),
            )
        )
    return slots


def _coerce_tool_steps(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    steps: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("toolName") or item.get("name") or "").strip()
        if not name:
            continue
        arguments = item.get("arguments")
        steps.append(
            {
                "toolName": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return steps


def split_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Split a skill file into frontmatter metadata and markdown body."""
    stripped = str(text or "").lstrip("\ufeff")
    match = _FRONTMATTER.match(stripped)
    if not match:
        return {}, stripped.strip()
    meta = _load_yaml(match.group(1))
    body = stripped[match.end() :].strip()
    return meta, body


def draft_from_markdown(
    text: str,
    *,
    filename_slug: str | None = None,
) -> tuple[SkillDraft, str | None, bool]:
    """Parse markdown into a skill draft. Returns (draft, slug, explicit_tool_steps)."""
    meta, body = split_skill_markdown(text)
    title = str(meta.get("title", "")).strip()
    description = str(meta.get("description", "")).strip()
    if not title:
        raise HomeAssistantError("Skill markdown requires frontmatter title")
    if not description:
        raise HomeAssistantError("Skill markdown requires frontmatter description")
    triggers = _coerce_triggers(meta.get("triggers"))
    if not triggers:
        raise HomeAssistantError("Skill markdown requires at least one trigger")
    if not body.strip():
        raise HomeAssistantError("Skill markdown requires a workflow body")

    explicit_tool_steps = "tool_steps" in meta
    draft = SkillDraft(
        title=title,
        description=description,
        triggers=triggers,
        body=body,
        tool_steps=_coerce_tool_steps(meta.get("tool_steps", [])),
        slots=_coerce_slots(meta.get("slots")),
        preconditions=str(meta.get("preconditions", "") or ""),
        parent_id=(
            str(meta["parent_id"]).strip()
            if meta.get("parent_id")
            else None
        ),
        route_scope=(
            str(meta["route_scope"]).strip()
            if meta.get("route_scope")
            else None
        ),
    )
    draft = normalize_skill_draft(draft, explicit_tool_steps=explicit_tool_steps)
    slug = str(meta.get("slug") or filename_slug or "").strip() or None
    return draft, slug, explicit_tool_steps


def skill_to_markdown(skill: Skill, *, include_tool_steps: bool = False) -> str:
    """Serialize a skill as a markdown file."""
    meta: dict[str, Any] = {
        "title": skill.title,
        "description": skill.description,
        "triggers": list(skill.triggers),
        "enabled": bool(skill.enabled),
    }
    if skill.slug:
        meta["slug"] = skill.slug
    if skill.route_scope:
        meta["route_scope"] = skill.route_scope
    if skill.preconditions:
        meta["preconditions"] = skill.preconditions
    if skill.parent_id:
        meta["parent_id"] = skill.parent_id
    if skill.slots:
        meta["slots"] = [
            {
                "name": slot.name,
                **({"description": slot.description} if slot.description else {}),
                **({"default": slot.default} if slot.default is not None else {}),
                **({"source": slot.source} if slot.source != "user" else {}),
            }
            for slot in skill.slots
        ]
    if include_tool_steps and skill.tool_steps:
        meta["tool_steps"] = skill.tool_steps
    return f"---\n{_dump_yaml(meta)}\n---\n\n{skill.body.strip()}\n"


def apply_draft_to_skill(skill: Skill, draft: SkillDraft) -> None:
    """Copy parsed draft fields onto an existing skill."""
    skill.title = draft.title
    skill.description = draft.description
    skill.triggers = list(draft.triggers)
    skill.body = draft.body
    skill.tool_steps = list(draft.tool_steps)
    skill.slots = list(draft.slots)
    skill.preconditions = draft.preconditions
    skill.parent_id = draft.parent_id
    skill.route_scope = draft.route_scope

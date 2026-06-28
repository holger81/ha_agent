"""Canonical MCP tool names and alias normalization for skills and playbooks."""

from __future__ import annotations

import copy
import re
from typing import Any

from .models import Skill, SkillDraft

IMAP_MAILBOX_STATUS = "mail_mcp__imap_mailbox_status"
IMAP_SEARCH_MESSAGES = "mail_mcp__imap_search_messages"
IMAP_FETCH_MESSAGE = "mail_mcp__imap_fetch_message"

EMAIL_IMAP_TOOLS = (
    IMAP_MAILBOX_STATUS,
    IMAP_SEARCH_MESSAGES,
    IMAP_FETCH_MESSAGE,
)

# Common LLM / legacy abbreviations → proxy callTool names.
_TOOL_ALIASES: dict[str, str] = {
    "mailbox_status": IMAP_MAILBOX_STATUS,
    "search_messages": IMAP_SEARCH_MESSAGES,
    "get_message": IMAP_FETCH_MESSAGE,
    "fetch_message": IMAP_FETCH_MESSAGE,
    "mail_mcp__mailbox_status": IMAP_MAILBOX_STATUS,
    "mail_mcp__search_messages": IMAP_SEARCH_MESSAGES,
    "mail_mcp__get_message": IMAP_FETCH_MESSAGE,
    "mail_mcp__imap_get_message": IMAP_FETCH_MESSAGE,
    "mail_mcp_imap_mailbox_status": IMAP_MAILBOX_STATUS,
    "mail_mcp_imap_search_messages": IMAP_SEARCH_MESSAGES,
    "mail_mcp_imap_fetch_message": IMAP_FETCH_MESSAGE,
    "mail_mcp_imap_get_message": IMAP_FETCH_MESSAGE,
}

_BACKTICK_TOOL = re.compile(
    r"`([a-z][a-z0-9_]*(?:__[a-z0-9_]+)*)`",
    re.IGNORECASE,
)
_BARE_MCP_TOOL = re.compile(
    r"\b([a-z][a-z0-9_]*(?:__[a-z0-9_]+)+)\b",
    re.IGNORECASE,
)
_BARE_ALIAS = re.compile(
    r"\b(mailbox_status|search_messages|get_message|fetch_message)\b",
    re.IGNORECASE,
)


def _normalize_upstream_tool_name(name: str) -> str:
    """Fix common LLM mistakes in MCP proxy tool names."""
    cleaned = name.strip()
    if not cleaned:
        return cleaned

    if "__" not in cleaned:
        return cleaned.replace("-", "_")

    parts = [part.replace("-", "_") for part in cleaned.split("__") if part]
    while len(parts) >= 3 and parts[0] == parts[1]:
        parts = [parts[0], *parts[2:]]
    return "__".join(parts)


def canonicalize_tool_name(name: str) -> str:
    """Return the MCP proxy tool name for a shorthand or mistyped name."""
    cleaned = name.strip()
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()
    if lowered in _TOOL_ALIASES:
        return _TOOL_ALIASES[lowered]

    normalized = _normalize_upstream_tool_name(cleaned)
    norm_lower = normalized.lower()
    if norm_lower in _TOOL_ALIASES:
        return _TOOL_ALIASES[norm_lower]

    return normalized


def canonicalize_tool_steps(
    steps: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Normalize toolName values on structured tool steps."""
    if not steps:
        return [], False
    changed = False
    out: list[dict[str, Any]] = []
    for step in steps:
        step_copy = dict(step)
        raw = str(step_copy.get("toolName") or step_copy.get("name") or "").strip()
        if not raw:
            out.append(step_copy)
            continue
        canon = canonicalize_tool_name(raw)
        if canon != raw:
            changed = True
        step_copy["toolName"] = canon
        step_copy.pop("name", None)
        out.append(step_copy)
    return out, changed


def rewrite_tool_names_in_text(text: str) -> tuple[str, bool]:
    """Replace shorthand tool names in markdown workflow text."""
    if not text.strip():
        return text, False

    changed = False

    def _backtick_repl(match: re.Match[str]) -> str:
        nonlocal changed
        name = match.group(1)
        canon = canonicalize_tool_name(name)
        if canon != name:
            changed = True
            return f"`{canon}`"
        return match.group(0)

    updated = _BACKTICK_TOOL.sub(_backtick_repl, text)

    def _bare_mcp_repl(match: re.Match[str]) -> str:
        nonlocal changed
        name = match.group(1)
        canon = canonicalize_tool_name(name)
        if canon != name:
            changed = True
            return canon
        return name

    updated = _BARE_MCP_TOOL.sub(_bare_mcp_repl, updated)

    def _alias_repl(match: re.Match[str]) -> str:
        nonlocal changed
        name = match.group(1)
        canon = canonicalize_tool_name(name)
        if canon != name:
            changed = True
            return canon
        return name

    updated = _BARE_ALIAS.sub(_alias_repl, updated)
    return updated, changed


def canonicalize_skill_draft(draft: SkillDraft) -> tuple[SkillDraft, bool]:
    """Fix tool names on a draft before persistence."""
    changed = False
    new_body, body_changed = rewrite_tool_names_in_text(draft.body)
    if body_changed:
        changed = True
    steps, steps_changed = canonicalize_tool_steps(draft.tool_steps)
    if steps_changed:
        changed = True
    if not changed:
        return draft, False
    return SkillDraft(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=new_body,
        tool_steps=steps,
        slots=draft.slots,
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope,
    ), True


def canonicalize_skill(skill: Skill) -> tuple[Skill, bool]:
    """Fix tool names on a stored skill."""
    working = copy.deepcopy(skill)
    new_body, body_changed = rewrite_tool_names_in_text(working.body)
    if body_changed:
        working.body = new_body
    steps, steps_changed = canonicalize_tool_steps(working.tool_steps)
    if steps_changed:
        working.tool_steps = steps
    changed = body_changed or steps_changed
    return working, changed

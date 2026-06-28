"""Shipped markdown skills and refresh helpers for known broken learned skills."""

from __future__ import annotations

import re
from pathlib import Path

from .markdown import apply_draft_to_skill, draft_from_markdown
from .models import Skill
from .tool_names import (
    IMAP_GET_MESSAGE,
    IMAP_MAILBOX_STATUS,
    IMAP_SEARCH_MESSAGES,
    canonicalize_tool_name,
)

# Learned skills that should match the bundled email workflow.
BUNDLED_SKILL_FILES: dict[str, str] = {
    "check-and-read-unread-emails": "check-and-read-unread-emails.md",
    "check-unread-emails": "check-and-read-unread-emails.md",
    "email-management": "check-and-read-unread-emails.md",
}

_STALE_EMAIL_MARKERS = (
    "imap_fetch_message",
    "mail_mcp__fetch_message",
    "mail_mcp__mailbox_status",
    "mail_mcp__search_messages",
)
_BARE_SHORT_TOOLS = re.compile(
    r"\b(mailbox_status|search_messages|get_message|fetch_message)\b",
    re.IGNORECASE,
)


def bundled_skill_path(slug: str) -> Path | None:
    """Return the packaged markdown path for a bundled skill slug."""
    filename = BUNDLED_SKILL_FILES.get(slug)
    if not filename:
        return None
    root = Path(__file__).resolve().parent / "bundled"
    path = root / filename
    return path if path.is_file() else None


def load_bundled_skill_text(slug: str) -> str | None:
    """Read bundled markdown for a slug."""
    path = bundled_skill_path(slug)
    if path is None:
        return None
    return path.read_text(encoding="utf-8")


def email_skill_needs_refresh(skill: Skill) -> bool:
    """Return True when a learned email skill still has broken workflow text."""
    if skill.is_builtin:
        return False
    route = (skill.route_scope or "").lower()
    slug = skill.slug.lower()
    if route != "email" and slug not in BUNDLED_SKILL_FILES:
        return False

    blob = f"{skill.body}\n{skill.tool_steps}".lower()
    if any(marker in blob for marker in _STALE_EMAIL_MARKERS):
        return True
    if _BARE_SHORT_TOOLS.search(skill.body):
        return True

    for step in skill.tool_steps:
        raw = str(step.get("toolName") or "").strip()
        if raw and canonicalize_tool_name(raw) != raw:
            return True
        args = step.get("arguments")
        if not isinstance(args, dict):
            continue
        name = canonicalize_tool_name(raw)
        if name == IMAP_MAILBOX_STATUS and not args.get("mailbox"):
            return True
        if name == IMAP_SEARCH_MESSAGES and (
            not args.get("mailbox") or "unread_only" not in args
        ):
            return True
        if name == IMAP_GET_MESSAGE and not args.get("message_id"):
            return True

    expected = {IMAP_MAILBOX_STATUS, IMAP_SEARCH_MESSAGES}
    step_names = {
        canonicalize_tool_name(str(step.get("toolName") or ""))
        for step in skill.tool_steps
    }
    return (
        route == "email"
        and slug in BUNDLED_SKILL_FILES
        and not expected <= step_names
    )


def apply_bundled_skill(skill: Skill) -> bool:
    """Replace skill fields from bundled markdown. Returns True when applied."""
    filename = BUNDLED_SKILL_FILES.get(skill.slug)
    if filename is None and (skill.route_scope or "").lower() == "email":
        filename = BUNDLED_SKILL_FILES["check-and-read-unread-emails"]
    if filename is None:
        return False

    path = Path(__file__).resolve().parent / "bundled" / filename
    if not path.is_file():
        return False

    draft, _slug_override, _explicit = draft_from_markdown(
        path.read_text(encoding="utf-8"),
        filename_slug=skill.slug,
    )
    apply_draft_to_skill(skill, draft)
    return True


def seed_missing_bundled_skills(store, directory: Path) -> int:
    """Insert the primary bundled email skill when no matching slug exists."""
    primary = "check-and-read-unread-emails"
    if store.get_skill_by_slug(primary) is not None:
        return 0
    for slug in BUNDLED_SKILL_FILES:
        if store.get_skill_by_slug(slug) is not None:
            return 0

    path = bundled_skill_path(primary)
    if path is None:
        return 0

    draft, slug_override, _explicit = draft_from_markdown(
        path.read_text(encoding="utf-8"),
        filename_slug=primary,
    )
    skill = store.insert_skill(
        title=draft.title,
        description=draft.description,
        triggers=draft.triggers,
        body=draft.body,
        tool_steps=draft.tool_steps,
        slots=draft.slots,
        preconditions=draft.preconditions,
        parent_id=draft.parent_id,
        route_scope=draft.route_scope,
        slug=slug_override or primary,
    )
    from .body import normalize_skill
    from .markdown import skill_to_markdown

    normalize_skill(skill)
    store.update_skill(skill)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{skill.slug}.md"
    path.write_text(skill_to_markdown(skill), encoding="utf-8")
    return 1


def list_bundled_slugs() -> tuple[str, ...]:
    """Return slugs that have packaged markdown templates."""
    return tuple(BUNDLED_SKILL_FILES.keys())

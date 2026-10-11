"""Default skill slots keyed by ``route_scope`` (data, not domain branches)."""

from __future__ import annotations

from .models import Skill, SkillSlot

# Optional built-in slot templates for known scopes. Looked up by route_scope
# string only — new scopes work without code changes when they omit defaults.
_ROUTE_DEFAULT_SLOTS: dict[str, list[SkillSlot]] = {
    "email": [
        SkillSlot(
            name="mailbox",
            description="IMAP mailbox folder (INBOX, Junk, Sent, etc.)",
            source="default",
            default="INBOX",
        ),
    ],
}


def default_slots_for_route(route: str | None) -> list[SkillSlot]:
    """Return default slots for a route value."""
    if not route:
        return []
    return list(_ROUTE_DEFAULT_SLOTS.get(route.lower(), []))


def apply_route_defaults(skill: Skill) -> Skill:
    """Merge route default slots into a skill when slots are empty."""
    route = skill.route_scope or ""
    if skill.slots:
        return skill
    defaults = default_slots_for_route(route)
    if defaults:
        skill.slots = list(defaults)
    return skill


def apply_route_defaults_to_draft(draft) -> None:
    """Merge route defaults into a draft in place."""
    if draft.slots:
        return
    route = getattr(draft, "route_scope", None) or ""
    defaults = default_slots_for_route(route)
    if defaults:
        draft.slots = list(defaults)

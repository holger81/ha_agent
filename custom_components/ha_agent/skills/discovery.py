"""Skill discovery via FTS search."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from .format import format_skills_for_context
from .models import Skill
from .store import get_skill_store

_DISCOVERY_HISTORY_MESSAGES = 6
_DISCOVERY_MAX_CHARS = 2000


def build_discovery_query(
    user_text: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Combine the current turn with recent history for skill FTS search."""
    parts = [user_text.strip()]
    if history:
        for message in history[-_DISCOVERY_HISTORY_MESSAGES:]:
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(content)
    combined = "\n".join(part for part in parts if part)
    return combined[:_DISCOVERY_MAX_CHARS]


async def discover_skills(
    hass: HomeAssistant,
    entry_id: str,
    query: str,
    *,
    history: list[dict[str, str]] | None = None,
    max_inject: int = 3,
    enabled_only: bool = True,
) -> list[Skill]:
    """Find and load the best-matching skills for a user query."""
    search_text = build_discovery_query(query, history)
    if max_inject <= 0 or not search_text.strip():
        return []

    store = get_skill_store(hass, entry_id)

    def _search() -> list[Skill]:
        matches = store.search(
            search_text,
            limit=max_inject * 4,
            enabled_only=enabled_only,
        )
        skill_ids = [match.id for match in matches[: max_inject * 2]]
        skills = store.load_skills_by_ids(skill_ids)
        if enabled_only:
            skills = [skill for skill in skills if skill.enabled]
        return skills[:max_inject]

    return await hass.async_add_executor_job(_search)


def build_skill_hints(
    skills: list[Skill],
    *,
    route: str | None = None,
    slot_bindings: dict[str, str] | None = None,
) -> str:
    """Format matched skills for injection into tool context."""
    return format_skills_for_context(
        skills, route=route, slot_bindings=slot_bindings
    )

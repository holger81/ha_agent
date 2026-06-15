"""Skill discovery via FTS search."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from .format import format_skills_for_context
from .models import Skill
from .store import get_skill_store


async def discover_skills(
    hass: HomeAssistant,
    entry_id: str,
    query: str,
    *,
    max_inject: int = 3,
    enabled_only: bool = True,
) -> list[Skill]:
    """Find and load the best-matching skills for a user query."""
    if max_inject <= 0 or not query.strip():
        return []

    store = get_skill_store(hass, entry_id)

    def _search() -> list[Skill]:
        matches = store.search(query, limit=max_inject * 4, enabled_only=enabled_only)
        skill_ids = [match.id for match in matches[: max_inject * 2]]
        skills = store.load_skills_by_ids(skill_ids)
        if enabled_only:
            skills = [skill for skill in skills if skill.enabled]
        return skills[:max_inject]

    return await hass.async_add_executor_job(_search)


def build_skill_hints(skills: list[Skill]) -> str:
    """Format matched skills for injection into tool context."""
    return format_skills_for_context(skills)

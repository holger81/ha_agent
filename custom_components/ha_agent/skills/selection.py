"""LLM-assisted skill selection for agent turns."""

from __future__ import annotations

import json
import re
from typing import Any

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from .discovery import build_discovery_query
from .models import Skill
from .store import get_skill_store

_SELECT_PROMPT = (
    "You choose which saved workflow skill (if any) applies to the user's request.\n"
    'Return ONLY valid JSON: {{"skill_slugs": ["exact-slug"]}} or '
    '{{"skill_slugs": []}}.\n'
    "Rules:\n"
    "- Pick at most {max_select} skill(s).\n"
    "- Only use slugs from AVAILABLE SKILLS.\n"
    "- Prefer a skill when the request clearly matches its title, triggers, or "
    "description.\n"
    "- Return [] when no skill applies or a generic reply suffices."
)

_ROUTE_SEARCH_HINTS: dict[str, str] = {
    "email": "email mail inbox unread messages",
    "news": "news headlines briefing curate",
}

_CATALOG_LIMIT = 30


def parse_skill_selection(content: str) -> list[str]:
    """Parse skill slug list from an LLM selection response."""
    text = (content or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    slugs = data.get("skill_slugs")
    if not isinstance(slugs, list):
        return []
    return [str(slug).strip() for slug in slugs if str(slug).strip()]


async def select_skills_with_llm(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    route: str | None,
    catalog: list[Skill],
    max_select: int = 1,
) -> list[Skill]:
    """Ask the chat model which catalog skill(s) apply to this turn."""
    if not catalog or max_select <= 0:
        return []

    entries = [
        {
            "slug": skill.slug,
            "title": skill.title,
            "description": skill.description,
            "triggers": skill.triggers,
        }
        for skill in catalog
    ]
    messages = [
        {
            "role": "system",
            "content": _SELECT_PROMPT.format(max_select=max_select),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "route": route or "general",
                    "available_skills": entries,
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill selection LLM call failed: %s", err)
        return []

    by_slug = {skill.slug: skill for skill in catalog}
    selected: list[Skill] = []
    for slug in parse_skill_selection(result.content or ""):
        if skill := by_slug.get(slug):
            selected.append(skill)
        if len(selected) >= max_select:
            break
    return selected


def _merge_catalog(*groups: list[Skill]) -> list[Skill]:
    """Return skills in order without duplicates."""
    merged: list[Skill] = []
    seen: set[str] = set()
    for group in groups:
        for skill in group:
            if skill.id in seen:
                continue
            merged.append(skill)
            seen.add(skill.id)
    return merged


def _load_skill_candidates(
    store: Any,
    *,
    user_text: str,
    history: list[dict[str, str]] | None,
    route: str | None,
    max_inject: int,
) -> tuple[list[Skill], list[Skill]]:
    """Load candidate skills and FTS matches for selection.

    For a specialized route (e.g. ``news``/``email``) candidates are restricted
    to skills relevant to that route so an off-route skill (such as an email
    workflow) is never offered for a clearly news-routed query. For unrouted
    turns the full enabled catalog is offered so generic discovery still works.
    """
    enabled = store.list_enabled(limit=_CATALOG_LIMIT)
    if not enabled:
        return [], []

    search_text = build_discovery_query(user_text, history)
    fts_rows = store.search(
        search_text,
        limit=max(max_inject * 4, 8),
        enabled_only=True,
    )
    fts_skills = store.load_skills_by_ids([row.id for row in fts_rows])

    route_hint = _ROUTE_SEARCH_HINTS.get(route or "")
    if route_hint is None:
        # Unrouted turn: offer the whole enabled catalog for discovery.
        return enabled, fts_skills

    # Routed turn: keep only skills relevant to this route. A skill is relevant
    # when it matches the route hint or directly matched the user's query.
    hint_rows = store.search(
        route_hint,
        limit=max(max_inject * 4, 8),
        enabled_only=True,
    )
    relevant_ids = {row.id for row in hint_rows}
    relevant_ids.update(skill.id for skill in fts_skills)
    candidates = [skill for skill in enabled if skill.id in relevant_ids]
    return candidates, fts_skills


async def resolve_skills_for_turn(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    user_text: str,
    *,
    history: list[dict[str, str]] | None = None,
    route: str | None = None,
    max_inject: int = 3,
) -> list[Skill]:
    """Pick skill(s) for a turn via FTS candidates and LLM selection."""
    if max_inject <= 0:
        return []

    store = get_skill_store(hass, entry_id)

    def _load() -> tuple[list[Skill], list[Skill]]:
        return _load_skill_candidates(
            store,
            user_text=user_text,
            history=history,
            route=route,
            max_inject=max_inject,
        )

    candidates, fts_matches = await hass.async_add_executor_job(_load)
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[:max_inject]

    # FTS already pinned a single skill — trust it and skip the extra LLM call.
    if len(fts_matches) == 1:
        return fts_matches[:max_inject]

    catalog = _merge_catalog(fts_matches, candidates)
    selected = await select_skills_with_llm(
        llm,
        backend,
        user_text=user_text,
        route=route,
        catalog=catalog,
        max_select=max_inject,
    )
    if selected:
        return selected[:max_inject]

    if fts_matches:
        return fts_matches[:max_inject]

    return []

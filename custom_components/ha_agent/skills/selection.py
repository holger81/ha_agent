"""LLM-assisted skill selection for agent turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..context import is_casual_chat_query, is_chat_route
from ..llm_client import LlmClient
from ..structured_output import SKILL_SELECT_SCHEMA, json_schema_format
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
    "- Return [] when no skill applies or a generic reply suffices.\n"
    "- Never pick a skill whose domain conflicts with route (e.g. email skill "
    "when route is news, or news skill when route is email).\n"
    "- When route is news, email, or action, only pick skills that clearly "
    "belong to that domain."
)

_ROUTE_DOMAIN_MARKERS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b(email|e-?mail|inbox|imap|mailbox|unread)\b",
        re.IGNORECASE,
    ),
    "news": re.compile(
        r"\b(news|headline|briefing|rss|nachrichten|curate)\b",
        re.IGNORECASE,
    ),
    "action": re.compile(
        r"\b("
        r"light|switch|cover|fan|lock|climate|camera|entity_id|snapshot|"
        r"ha_call_service|turn\s+on|turn\s+off|toggle"
        r")\b",
        re.IGNORECASE,
    ),
}

_SPECIALIZED_ROUTES = frozenset(_ROUTE_DOMAIN_MARKERS)

_ROUTE_SEARCH_HINTS: dict[str, str] = {
    "email": "email mail inbox unread messages",
    "news": "news headlines briefing curate",
}

_ROUTE_TOOL_MARKERS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"mail|imap|inbox|email|mailbox", re.IGNORECASE),
    "news": re.compile(r"news|curate|headline|rss", re.IGNORECASE),
    "action": re.compile(
        r"ha_call_service|turn_on|turn_off|snapshot|open_cover|close_cover",
        re.IGNORECASE,
    ),
}

_CATALOG_LIMIT = 30


@dataclass(frozen=True, slots=True)
class SkillSelectionResult:
    """Outcome of skill matching for one agent turn."""

    skills: list[Skill]
    method: str
    summary: str
    detail: str
    candidate_count: int = 0
    classifier_raw: str | None = None


def _skill_text(skill: Skill) -> str:
    """Return searchable skill text for route matching."""
    return " ".join(
        [
            skill.title,
            skill.description,
            skill.body,
            *[str(trigger) for trigger in skill.triggers],
        ]
    )


def skill_matches_route(skill: Skill, route: str | None) -> bool:
    """Return True when a skill plausibly belongs on the active route."""
    route_key = (route or "").lower()
    if route_key not in _SPECIALIZED_ROUTES:
        return True

    target = _ROUTE_DOMAIN_MARKERS[route_key]
    text = _skill_text(skill)
    if target.search(text):
        return True

    for other_route, other_pattern in _ROUTE_DOMAIN_MARKERS.items():
        if other_route == route_key:
            continue
        if other_pattern.search(text):
            return False

    return False


def _filter_by_route(skills: list[Skill], route: str | None) -> list[Skill]:
    """Drop skills whose domain conflicts with the active route."""
    return [skill for skill in skills if skill_matches_route(skill, route)]


def tool_step_matches_route(tool_name: str, route: str | None) -> bool:
    """Return True when a structured tool step fits the active route."""
    route_key = (route or "").lower()
    if route_key not in _SPECIALIZED_ROUTES:
        return True

    name_lower = tool_name.lower()
    target = _ROUTE_TOOL_MARKERS[route_key]
    if target.search(name_lower):
        return True

    for other_route, other_pattern in _ROUTE_TOOL_MARKERS.items():
        if other_route == route_key:
            continue
        if other_pattern.search(name_lower):
            return False

    return False


def filter_tool_steps_for_route(
    steps: list[dict[str, Any]] | None,
    route: str | None,
) -> list[dict[str, Any]] | None:
    """Drop skill tool steps that conflict with the router's active route."""
    if not steps:
        return None
    filtered = [
        step
        for step in steps
        if tool_step_matches_route(str(step.get("toolName") or ""), route)
    ]
    return filtered or None


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
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> tuple[list[Skill], str]:
    """Ask the classifier model which catalog skill(s) apply; return (skills, raw)."""
    from ..llm_telemetry import record_llm_call

    if not catalog or max_select <= 0:
        return [], ""

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
    response_format = (
        json_schema_format("skill_select", SKILL_SELECT_SCHEMA)
        if structured_output_enabled
        else None
    )
    try:
        result = await llm.chat(
            messages,
            backend,
            tools=[],
            response_format=response_format,
        )
        record_llm_call(trace, role="skill_select", backend=backend, result=result)
    except Exception as err:
        LOGGER.warning("Skill selection LLM call failed: %s", err)
        record_llm_call(trace, role="skill_select", backend=backend, error=str(err))
        return [], ""

    raw = (result.content or "").strip()
    by_slug = {skill.slug: skill for skill in catalog}
    selected: list[Skill] = []
    for slug in parse_skill_selection(raw):
        if skill := by_slug.get(slug):
            selected.append(skill)
        if len(selected) >= max_select:
            break
    return selected, raw


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
    fts_skills = _filter_by_route(fts_skills, route)

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
    return _filter_by_route(candidates, route), fts_skills


def _resolve_chat_route_skills(
    store: Any,
    user_text: str,
    *,
    max_inject: int,
) -> SkillSelectionResult:
    """On chat routes, only pin a skill when the user text alone matches one."""
    rows = store.search(user_text.strip(), limit=2, enabled_only=True)
    if len(rows) != 1:
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chat route)",
            detail="Chat turns skip learned skills unless one clearly matches.",
        )
    skills = store.load_skills_by_ids([rows[0].id])
    skill = skills[0] if skills else None
    if skill is None or skill.is_builtin:
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chat route)",
            detail="Chat turns skip learned skills unless one clearly matches.",
        )
    return SkillSelectionResult(
        skills=[skill][:max_inject],
        method="fts_only",
        summary=f"FTS → {skill.slug}",
        detail=f"User text pinned skill {skill.title!r} on chat route.",
        candidate_count=1,
    )


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
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> SkillSelectionResult:
    """Pick skill(s) for a turn via FTS candidates and LLM selection."""
    if max_inject <= 0 or is_casual_chat_query(user_text):
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chitchat)",
            detail="Skill classifier skipped for greeting/chitchat.",
        )

    store = get_skill_store(hass, entry_id)

    if is_chat_route(route):

        def _chat_only() -> SkillSelectionResult:
            return _resolve_chat_route_skills(
                store,
                user_text,
                max_inject=max_inject,
            )

        return await hass.async_add_executor_job(_chat_only)

    def _load() -> tuple[list[Skill], list[Skill]]:
        return _load_skill_candidates(
            store,
            user_text=user_text,
            history=history,
            route=route,
            max_inject=max_inject,
        )

    candidates, fts_matches = await hass.async_add_executor_job(_load)
    candidates = _filter_by_route(candidates, route)
    fts_matches = _filter_by_route(fts_matches, route)
    if not candidates and not fts_matches:
        return SkillSelectionResult(
            skills=[],
            method="none",
            summary="no skill (no candidates)",
            detail="No enabled skills matched this query on the active route.",
        )

    # FTS already pinned a single route-relevant skill — skip the extra LLM call.
    if len(fts_matches) == 1:
        skill = fts_matches[0]
        return SkillSelectionResult(
            skills=fts_matches[:max_inject],
            method="fts_only",
            summary=f"FTS → {skill.slug}",
            detail=f"Keyword search pinned skill {skill.title!r} ({skill.slug}).",
            candidate_count=1,
        )

    catalog = _filter_by_route(_merge_catalog(fts_matches, candidates), route)
    if not catalog:
        return SkillSelectionResult(
            skills=[],
            method="none",
            summary="no skill (route filter)",
            detail="Candidates were filtered out for the active route.",
        )

    selected, raw = await select_skills_with_llm(
        llm,
        backend,
        user_text=user_text,
        route=route,
        catalog=catalog,
        max_select=max_inject,
        structured_output_enabled=structured_output_enabled,
        trace=trace,
    )
    raw_preview = raw[:240] if raw else None
    if selected:
        filtered = _filter_by_route(selected, route)[:max_inject]
        slugs = ", ".join(skill.slug for skill in filtered)
        return SkillSelectionResult(
            skills=filtered,
            method="llm",
            summary=f"LLM → {slugs}",
            detail=(
                f"Classifier picked {len(filtered)} skill(s) from "
                f"{len(catalog)} candidate(s): {slugs}."
            ),
            candidate_count=len(catalog),
            classifier_raw=raw_preview,
        )

    if fts_matches:
        skill = fts_matches[0]
        return SkillSelectionResult(
            skills=fts_matches[:max_inject],
            method="fts_fallback",
            summary=f"LLM none, FTS → {skill.slug}",
            detail=(
                f"Classifier returned no skill from {len(catalog)} candidate(s); "
                f"using FTS match {skill.title!r} ({skill.slug})."
            ),
            candidate_count=len(catalog),
            classifier_raw=raw_preview,
        )

    return SkillSelectionResult(
        skills=[],
        method="llm_empty",
        summary="LLM → none",
        detail=(
            f"Classifier returned no skill from {len(catalog)} candidate(s)."
        ),
        candidate_count=len(catalog),
        classifier_raw=raw_preview,
    )

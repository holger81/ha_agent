"""LLM-assisted skill selection for agent turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..context import is_casual_chat_query, is_chat_route, is_short_follow_up_query
from ..llm_client import LlmClient
from ..structured_output import SKILL_SELECT_SCHEMA, json_schema_format
from .discovery import build_discovery_query
from .models import Skill
from .store import get_skill_store

_SELECT_PROMPT = (
    "You choose which saved workflow skill (if any) matches the user's INTENT.\n"
    'Return ONLY valid JSON: {{"skill_slugs": ["exact-slug"]}} or '
    '{{"skill_slugs": []}}.\n'
    "Rules:\n"
    "- Pick at most {max_select} skill(s).\n"
    "- Only use slugs from AVAILABLE SKILLS.\n"
    "- Match on intent (what the user wants done), not exact wording. Triggers "
    "are examples — paraphrases of the same workflow should still match.\n"
    "- Prefer skills whose tools/workflow fit the request (read/status vs "
    "control/mutation, email vs news vs device, etc.).\n"
    "- Short follow-ups (again, check again, retry, that one) continue the "
    "recent conversation topic from recent_messages — do not switch to an "
    "unrelated domain skill that only shares a verb like check/read.\n"
    "- Return [] when unsure, when no skill fits, or when a generic reply "
    "suffices. Do not guess.\n"
    "- Return [] for status/read questions when the only candidates are "
    "control/mutation skills (turn on/off, toggle).\n"
    "- Never pick an unrelated device, area, or domain skill.\n"
    "- When a domain_hint is present, prefer skills for that domain.\n"
    "- Never pick a skill whose tools/domain conflict with the request "
    "(e.g. Home Assistant entity search for an email ask, or mail tools for "
    "a device-control request)."
)

_ROUTE_DOMAIN_MARKERS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b(emails?|e-?mails?|inbox|imap|mailbox|unread)\b",
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
        r"ha_call_service|turn_on|turn_off|snapshot|open_cover|close_cover|"
        r"home_assistant|ha_search|ha_get_|ha_bulk_|ha_set_",
        re.IGNORECASE,
    ),
}

# Soft workflow domains on chat (not device-control/action).
_SOFT_DOMAIN_HINTS = frozenset(key for key in _ROUTE_DOMAIN_MARKERS if key != "action")

_CATALOG_LIMIT = 30
# Minimum Jaccard overlap between user tokens and skill triggers/title before
# unsupervised FTS may pin a skill (without LLM confirmation). LLM/prepass
# picks are trusted for intent and are not re-gated by this threshold.
_MIN_FTS_TRIGGER_SCORE = 0.22
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "what",
        "how",
        "who",
        "when",
        "where",
        "why",
        "can",
        "you",
        "please",
        "just",
        "into",
        "about",
    }
)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]{3,}", text.lower()) if tok}


def _content_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOPWORDS


def trigger_overlap_score(user_text: str, skill: Skill) -> float:
    """Jaccard-ish overlap of user tokens vs skill triggers/title/description."""
    user_tokens = _tokenize(user_text)
    if not user_tokens:
        return 0.0
    skill_tokens = _tokenize(
        " ".join(
            [
                skill.title,
                skill.description,
                *[str(trigger) for trigger in skill.triggers],
            ]
        )
    )
    if not skill_tokens:
        return 0.0
    overlap = user_tokens & skill_tokens
    return len(overlap) / max(len(user_tokens), 1)


def _skill_content_tokens(skill: Skill) -> set[str]:
    # Title + triggers only: descriptions add unrelated tokens that dilute overlap.
    return _content_tokens(
        " ".join([skill.title, *[str(trigger) for trigger in skill.triggers]])
    )


def _jaccard_content_overlap(user_text: str, skill: Skill) -> float:
    user_tokens = _content_tokens(user_text)
    skill_tokens = _skill_content_tokens(skill)
    if not user_tokens or not skill_tokens:
        return 0.0
    overlap = user_tokens & skill_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(user_tokens | skill_tokens)


def _strong_fts_match(user_text: str, skill: Skill) -> bool:
    """Return True when FTS match is strong enough to pin without LLM."""
    return trigger_overlap_score(user_text, skill) >= _MIN_FTS_TRIGGER_SCORE


def skill_applies_to_user_text(user_text: str, skill: Skill) -> bool:
    """Return True when lexical overlap is strong enough for unsupervised FTS pin.

    Do not use this to veto LLM/prepass intent picks — those already judged intent.
    """
    return _jaccard_content_overlap(user_text, skill) >= _MIN_FTS_TRIGGER_SCORE


def _prefer_overlapping_catalog(user_text: str, catalog: list[Skill]) -> list[Skill]:
    """Prefer skills that apply; else any content overlap; else full catalog."""
    if not catalog:
        return catalog
    strong = [
        skill for skill in catalog if skill_applies_to_user_text(user_text, skill)
    ]
    if strong:
        return strong
    overlapping = [
        skill for skill in catalog if _jaccard_content_overlap(user_text, skill) > 0
    ]
    return overlapping if overlapping else catalog


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


def infer_soft_domain_hint(user_text: str) -> str | None:
    """Infer a soft chat-domain hint from user text via shared markers.

    Returns a domain only when exactly one soft domain matches (not action).
    """
    text = (user_text or "").strip()
    if not text:
        return None
    hits = [
        domain
        for domain in _SOFT_DOMAIN_HINTS
        if _ROUTE_DOMAIN_MARKERS[domain].search(text)
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _skill_step_names(skill: Skill) -> list[str]:
    return [
        str(step.get("toolName") or "")
        for step in (skill.tool_steps or [])
        if step.get("toolName")
    ]


def _skill_tool_domains(skill: Skill) -> set[str]:
    """Specialized domains implied by a skill's concrete tool steps."""
    domains: set[str] = set()
    for name in _skill_step_names(skill):
        for domain, pattern in _ROUTE_TOOL_MARKERS.items():
            if pattern.search(name):
                domains.add(domain)
    return domains


def skill_matches_route(
    skill: Skill,
    route: str | None,
    *,
    domain_hint: str | None = None,
) -> bool:
    """Return True when a skill plausibly belongs on the active route/domain."""
    route_key = (route or "").lower()
    hint = (domain_hint or "").lower()
    scope = (skill.route_scope or "").lower()

    # Soft domain on chat: prefer matching scope/tools; reject other domains.
    if route_key in {"", "chat"} and hint in _SOFT_DOMAIN_HINTS:
        if scope and scope != hint and scope in _SPECIALIZED_ROUTES:
            return False
        if scope == hint:
            return True
        target = _ROUTE_DOMAIN_MARKERS.get(hint)
        if target and target.search(_skill_text(skill)):
            return True
        step_names = _skill_step_names(skill)
        marker = _ROUTE_TOOL_MARKERS.get(hint)
        if marker and any(marker.search(name) for name in step_names):
            return True
        tool_domains = _skill_tool_domains(skill)
        # Concrete tools for a different specialized domain cannot serve this hint
        # (e.g. ha_search status skill on an email ask).
        if tool_domains and hint not in tool_domains:
            return False
        # Keep unmarked / tool-less skills eligible for FTS/LLM.
        return scope not in _SPECIALIZED_ROUTES or not scope

    if route_key not in _SPECIALIZED_ROUTES:
        return True

    # Explicit scope wins when it matches the active specialized route.
    if scope == route_key:
        return True

    step_names = [
        str(step.get("toolName") or "")
        for step in (skill.tool_steps or [])
        if step.get("toolName")
    ]
    # Structured tools win over vague title/trigger text: drop email/news
    # workflows on action (and vice versa) even when FTS ranked them highly.
    if step_names and any(
        not tool_step_matches_route(name, route_key) for name in step_names
    ):
        return False

    target = _ROUTE_DOMAIN_MARKERS[route_key]
    text = _skill_text(skill)
    if target.search(text):
        return True

    for other_route, other_pattern in _ROUTE_DOMAIN_MARKERS.items():
        if other_route == route_key:
            continue
        if other_pattern.search(text):
            return False

    return bool(
        step_names
        and all(tool_step_matches_route(name, route_key) for name in step_names)
    )


def _filter_by_route(
    skills: list[Skill],
    route: str | None,
    *,
    domain_hint: str | None = None,
) -> list[Skill]:
    """Drop skills whose domain conflicts with the active route/hint."""
    return [
        skill
        for skill in skills
        if skill_matches_route(skill, route, domain_hint=domain_hint)
    ]


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

    # On action, allow non-conflicting tools (e.g. ha_search / ha_get_state).
    # Soft domains still require a positive marker match.
    return route_key == "action"


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
    history: list[dict[str, str]] | None = None,
    trace: Any | None = None,
    domain_hint: str | None = None,
) -> tuple[list[Skill], str]:
    """Ask the classifier model which catalog skill(s) apply; return (skills, raw)."""
    from ..llm_telemetry import record_llm_call

    if not catalog or max_select <= 0:
        return [], ""

    entries = []
    for skill in catalog:
        tool_names: list[str] = []
        for step in skill.tool_steps or []:
            if not isinstance(step, dict):
                continue
            name = str(step.get("toolName") or step.get("name") or "").strip()
            if name:
                tool_names.append(name)
        entries.append(
            {
                "slug": skill.slug,
                "title": skill.title,
                "description": skill.description,
                "triggers": skill.triggers,
                "route_scope": skill.route_scope or "",
                "tools": tool_names[:8],
            }
        )
    recent_messages: list[dict[str, str]] = []
    for message in (history or [])[-4:]:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        recent_messages.append({"role": role, "content": content[:240]})
    payload: dict[str, Any] = {
        "user_text": user_text,
        "route": route or "general",
        "available_skills": entries,
    }
    if domain_hint:
        payload["domain_hint"] = domain_hint
    if recent_messages:
        payload["recent_messages"] = recent_messages
    messages = [
        {
            "role": "system",
            "content": _SELECT_PROMPT.format(max_select=max_select),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=True),
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
    domain_hint: str | None = None,
) -> SkillSelectionResult:
    """On chat routes, pin a skill when user text or domain hint clearly matches."""
    query = user_text.strip()
    hint = (domain_hint or "").lower().strip()
    if hint in _ROUTE_SEARCH_HINTS:
        query = f"{query} {_ROUTE_SEARCH_HINTS[hint]}".strip()
    rows = store.search(query, limit=3 if hint else 2, enabled_only=True)
    skills = store.load_skills_by_ids([row.id for row in rows]) if rows else []
    skills = _filter_by_route(skills, "chat", domain_hint=hint or None)

    if hint and skills:
        scoped = [
            skill
            for skill in skills
            if (skill.route_scope or "").lower() == hint
            or (
                hint in _ROUTE_DOMAIN_MARKERS
                and _ROUTE_DOMAIN_MARKERS[hint].search(_skill_text(skill))
            )
        ]
        if len(scoped) == 1:
            skill = scoped[0]
            return SkillSelectionResult(
                skills=[skill][:max_inject],
                method="fts_only",
                summary=f"FTS → {skill.slug}",
                detail=(
                    f"Domain hint {hint!r} pinned skill {skill.title!r} on chat route."
                ),
                candidate_count=len(rows),
            )

    if len(rows) != 1 or not skills:
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chat route)",
            detail="Chat turns skip learned skills unless one clearly matches.",
            candidate_count=len(rows),
        )
    skill = skills[0]
    if not _strong_fts_match(user_text, skill) and not (
        hint and (skill.route_scope or "").lower() == hint
    ):
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chat route)",
            detail="Chat FTS hit was too weak to pin without stronger overlap.",
            candidate_count=1,
        )
    if is_short_follow_up_query(user_text):
        # Force history-aware LLM selection instead of verb-overlap FTS pins.
        return SkillSelectionResult(
            skills=[],
            method="skipped",
            summary="no skill (chat route)",
            detail="Short follow-up skipped unsupervised FTS pin.",
            candidate_count=1,
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
    domain_hint: str | None = None,
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
    # Prefer router/prepass hint; otherwise infer from user text markers so
    # soft-domain asks (any scope) cannot pin conflicting HA/status skills.
    effective_hint = (domain_hint or "").strip().lower() or infer_soft_domain_hint(
        user_text
    )

    # Cheap unsupervised pins on chat (domain hint / strong FTS). Weak or
    # ambiguous FTS hits fall through to the LLM intent classifier below.
    if is_chat_route(route):

        def _chat_only() -> SkillSelectionResult:
            return _resolve_chat_route_skills(
                store,
                user_text,
                max_inject=max_inject,
                domain_hint=effective_hint,
            )

        chat_result = await hass.async_add_executor_job(_chat_only)
        if chat_result.skills:
            return chat_result
        if chat_result.candidate_count < 1:
            return chat_result

    def _load() -> tuple[list[Skill], list[Skill]]:
        return _load_skill_candidates(
            store,
            user_text=user_text,
            history=history,
            route=route,
            max_inject=max_inject,
        )

    candidates, fts_matches = await hass.async_add_executor_job(_load)
    candidates = _filter_by_route(candidates, route, domain_hint=effective_hint)
    fts_matches = _filter_by_route(fts_matches, route, domain_hint=effective_hint)
    if not candidates and not fts_matches:
        if is_chat_route(route):
            return SkillSelectionResult(
                skills=[],
                method="skipped",
                summary="no skill (chat route)",
                detail="Chat turns skip learned skills unless one clearly matches.",
            )
        return SkillSelectionResult(
            skills=[],
            method="none",
            summary="no skill (no candidates)",
            detail="No enabled skills matched this query on the active route.",
        )

    # FTS already pinned a single route-relevant skill — skip the extra LLM call
    # only when trigger overlap is strong enough for small models. Short
    # follow-ups ("check again") must not pin from a shared verb alone.
    follow_up = bool(history) and is_short_follow_up_query(user_text)
    if (
        len(fts_matches) == 1
        and _strong_fts_match(user_text, fts_matches[0])
        and not follow_up
    ):
        skill = fts_matches[0]
        return SkillSelectionResult(
            skills=fts_matches[:max_inject],
            method="fts_only",
            summary=f"FTS → {skill.slug}",
            detail=f"Keyword search pinned skill {skill.title!r} ({skill.slug}).",
            candidate_count=1,
        )

    catalog = _filter_by_route(
        _merge_catalog(fts_matches, candidates),
        route,
        domain_hint=effective_hint,
    )
    if not catalog:
        return SkillSelectionResult(
            skills=[],
            method="none",
            summary="no skill (route filter)",
            detail="Candidates were filtered out for the active route.",
        )
    # Prefer overlapping candidates for ranking, but keep the full catalog so the
    # intent classifier can still see paraphrases with weak lexical overlap.
    preferred = _prefer_overlapping_catalog(user_text, catalog)
    catalog = _merge_catalog(preferred, catalog)

    selected, raw = await select_skills_with_llm(
        llm,
        backend,
        user_text=user_text,
        route=route,
        catalog=catalog,
        max_select=max_inject,
        structured_output_enabled=structured_output_enabled,
        history=history,
        trace=trace,
        domain_hint=effective_hint,
    )
    raw_preview = raw[:240] if raw else None
    if selected:
        # Trust the classifier's intent pick; only apply route/domain filters.
        filtered = _filter_by_route(selected, route, domain_hint=effective_hint)[
            :max_inject
        ]
        if not filtered:
            return SkillSelectionResult(
                skills=[],
                method="llm_empty",
                summary="LLM → none (route filter)",
                detail=(
                    f"Classifier picked skill(s) from {len(catalog)} candidate(s), "
                    "but none remained after route filtering."
                ),
                candidate_count=len(catalog),
                classifier_raw=raw_preview,
            )
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

    # Weak FTS matches must not pin an active skill plan after LLM said none.
    # Follow-ups also skip unsupervised FTS fallback (needs history-aware LLM).
    strong_fts = [
        skill
        for skill in fts_matches
        if _strong_fts_match(user_text, skill) and not follow_up
    ]
    if strong_fts:
        skill = strong_fts[0]
        return SkillSelectionResult(
            skills=strong_fts[:max_inject],
            method="fts_fallback",
            summary=f"LLM none, FTS → {skill.slug}",
            detail=(
                f"Classifier returned no skill from {len(catalog)} candidate(s); "
                f"using strong FTS match {skill.title!r} ({skill.slug})."
            ),
            candidate_count=len(catalog),
            classifier_raw=raw_preview,
        )

    return SkillSelectionResult(
        skills=[],
        method="llm_empty",
        summary="LLM → none",
        detail=(f"Classifier returned no skill from {len(catalog)} candidate(s)."),
        candidate_count=len(catalog),
        classifier_raw=raw_preview,
    )

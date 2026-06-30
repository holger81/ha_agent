"""Merged turn prepass: route, complexity, skill, and slots in one call."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant

from .config_helpers import LlmBackend, RouterConfig
from .const import LOGGER
from .context import is_casual_chat_query
from .llm_client import LlmClient
from .llm_telemetry import record_llm_call
from .orchestrator import Complexity, OrchestrationPlan, heuristic_complexity
from .router import (
    _ROUTE_VALUE_TO_TASK,
    RouteResolution,
    TaskRoute,
    classify_route_with_detail,
)
from .skills.models import Skill
from .skills.params import apply_slot_defaults
from .skills.selection import (
    SkillSelectionResult,
    _filter_by_route,
    _load_skill_candidates,
    _merge_catalog,
    _resolve_chat_route_skills,
    is_chat_route,
)
from .skills.store import get_skill_store
from .structured_output import PREPASS_SCHEMA, json_schema_format

_PREPAS_PROMPT = (
    "You classify a Home Assistant agent turn in one pass.\n"
    "Pick route, complexity, optional learned skill slug from the catalog, "
    "and slot_bindings for that skill.\n"
    "Rules:\n"
    "- route: chat|email|news|action\n"
    "- complexity: simple (no tools), single (one workflow), complex (multi-domain)\n"
    "- skill_slug: empty string when no catalog skill applies\n"
    "- slot_bindings: only keys listed for the chosen skill; use empty strings "
    "when unknown\n"
    "- Prefer keyword_hint and heuristic_complexity when they are clear"
)


@dataclass(frozen=True, slots=True)
class TurnPrepassResult:
    """Outputs normally produced by several pre-loop classifiers."""

    route_resolution: RouteResolution
    orch_plan: OrchestrationPlan
    skill_selection: SkillSelectionResult | None
    slot_bindings: dict[str, str]
    method: str


def _skill_catalog_entries(skills: list[Skill]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for skill in skills:
        slot_names = [slot.name for slot in skill.slots]
        if not slot_names:
            from .skills.params import default_slots_for_skill

            slot_names = [slot.name for slot in default_slots_for_skill(skill)]
        entries.append(
            {
                "slug": skill.slug,
                "title": skill.title,
                "description": skill.description,
                "slots": slot_names,
            }
        )
    return entries


def _parse_prepass_payload(
    data: dict[str, Any],
    *,
    catalog_by_slug: dict[str, Skill],
    keyword_decision,
    heuristic: Complexity,
) -> TurnPrepassResult | None:
    route_value = str(data.get("route", "")).strip()
    if route_value not in _ROUTE_VALUE_TO_TASK:
        return None
    route = _ROUTE_VALUE_TO_TASK[route_value]

    raw_complexity = str(data.get("complexity", heuristic.value)).lower()
    try:
        complexity = Complexity(raw_complexity)
    except ValueError:
        complexity = heuristic

    reason = str(data.get("reason", "")).strip() or "prepass"
    skill_slug = str(data.get("skill_slug", "")).strip()
    bindings_raw = data.get("slot_bindings")
    bindings = (
        {str(k): str(v) for k, v in bindings_raw.items()}
        if isinstance(bindings_raw, dict)
        else {}
    )

    selected_skills: list[Skill] = []
    if skill_slug and skill_slug in catalog_by_slug:
        selected_skills = [catalog_by_slug[skill_slug]]

    skill_selection = None
    if selected_skills:
        skill = selected_skills[0]
        skill_selection = SkillSelectionResult(
            skills=selected_skills,
            method="prepass",
            summary=f"prepass → {skill.slug}",
            detail=f"Prepass selected skill {skill.title!r}.",
            candidate_count=len(catalog_by_slug),
        )
        slot_bindings = apply_slot_defaults(
            bindings,
            skill,
            route=route.value,
        )
    else:
        slot_bindings = {}

    return TurnPrepassResult(
        route_resolution=RouteResolution(
            route=route,
            method="prepass",
            classifier_summary=f"prepass → {route.value}",
            classifier_detail=(
                f"Prepass route {route.value}. "
                f"Keyword hint: {keyword_decision.summary}."
            ),
            keyword_hint=keyword_decision.summary,
            classifier_raw=json.dumps(data, ensure_ascii=True)[:240],
        ),
        orch_plan=OrchestrationPlan(
            complexity=complexity,
            reason=reason,
        ),
        skill_selection=skill_selection,
        slot_bindings=slot_bindings,
        method="prepass",
    )


async def run_turn_prepass(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    history: list[dict[str, str]] | None,
    router_config: RouterConfig,
    route_keywords: dict[str, list[str]] | None,
    skills_enabled: bool,
    max_inject: int,
    structured_output_enabled: bool,
    trace: Any | None = None,
) -> TurnPrepassResult | None:
    """Run the merged prepass when heuristics do not fully resolve the turn."""
    keyword_decision = classify_route_with_detail(
        user_text,
        [],
        router_config,
        route_keywords=route_keywords,
        history=history,
    )
    heuristic = heuristic_complexity(user_text)

    if is_casual_chat_query(user_text) and keyword_decision.route == TaskRoute.CHAT:
        return TurnPrepassResult(
            route_resolution=RouteResolution(
                route=TaskRoute.CHAT,
                method="heuristic",
                classifier_summary="heuristic → chat",
                classifier_detail="Chitchat fast-path skipped prepass LLM call.",
                keyword_hint=keyword_decision.summary,
            ),
            orch_plan=OrchestrationPlan(
                complexity=Complexity.SIMPLE,
                reason="chitchat",
            ),
            skill_selection=SkillSelectionResult(
                skills=[],
                method="skipped",
                summary="no skill (chitchat)",
                detail="Prepass skipped skill selection for chitchat.",
            ),
            slot_bindings={},
            method="heuristic",
        )

    catalog: list[Skill] = []
    fts_matches: list[Skill] = []
    if skills_enabled and max_inject > 0:
        store = get_skill_store(hass, entry_id)

        def _load() -> tuple[list[Skill], list[Skill]]:
            if is_chat_route(keyword_decision.route.value):
                chat_sel = _resolve_chat_route_skills(
                    store,
                    user_text,
                    max_inject=max_inject,
                )
                return chat_sel.skills, chat_sel.skills
            return _load_skill_candidates(
                store,
                user_text=user_text,
                history=history,
                route=keyword_decision.route.value,
                max_inject=max_inject,
            )

        candidates, fts_matches = await hass.async_add_executor_job(_load)
        catalog = _filter_by_route(
            _merge_catalog(fts_matches, candidates),
            keyword_decision.route.value,
        )

    if (
        skills_enabled
        and len(fts_matches) == 1
        and keyword_decision.method in {"keyword", "follow_up", "default"}
        and heuristic != Complexity.COMPLEX
    ):
        skill = fts_matches[0]
        bindings = apply_slot_defaults({}, skill, route=keyword_decision.route.value)
        return TurnPrepassResult(
            route_resolution=RouteResolution(
                route=keyword_decision.route,
                method="fts_fast_path",
                classifier_summary=f"keyword → {keyword_decision.route.value}",
                classifier_detail=(
                    f"Keyword route with single FTS skill {skill.slug}; "
                    "skipped prepass LLM call."
                ),
                keyword_hint=keyword_decision.summary,
            ),
            orch_plan=OrchestrationPlan(
                complexity=heuristic,
                reason="fts_fast_path",
            ),
            skill_selection=SkillSelectionResult(
                skills=[skill][:max_inject],
                method="fts_only",
                summary=f"FTS → {skill.slug}",
                detail=f"Keyword + FTS pinned skill {skill.title!r}.",
                candidate_count=1,
            ),
            slot_bindings=bindings,
            method="fts_fast_path",
        )

    catalog_by_slug = {skill.slug: skill for skill in catalog}
    recent = [
        turn.get("content", "")
        for turn in (history or [])[-4:]
        if turn.get("role") == "user"
    ]
    messages = [
        {"role": "system", "content": _PREPAS_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "recent_user_turns": recent,
                    "keyword_hint": keyword_decision.summary,
                    "heuristic_complexity": heuristic.value,
                    "available_routes": [
                        route
                        for route in ("chat", "email", "news", "action")
                        if route != "action"
                        or (
                            router_config.action_enabled
                            and router_config.action_backend
                        )
                    ],
                    "skill_catalog": _skill_catalog_entries(catalog[:12]),
                },
                ensure_ascii=True,
            ),
        },
    ]
    response_format = (
        json_schema_format("turn_prepass", PREPASS_SCHEMA, strict=False)
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
        record_llm_call(trace, role="prepass", backend=backend, result=result)
    except Exception as err:
        LOGGER.warning("Turn prepass failed: %s", err)
        record_llm_call(trace, role="prepass", backend=backend, error=str(err))
        return None

    raw = (result.content or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    parsed = _parse_prepass_payload(
        data,
        catalog_by_slug=catalog_by_slug,
        keyword_decision=keyword_decision,
        heuristic=heuristic,
    )
    if parsed is None:
        return None

    if (
        parsed.route_resolution.route == TaskRoute.HA_ACTION
        and not (
            router_config.action_enabled and router_config.action_backend
        )
    ):
        return TurnPrepassResult(
            route_resolution=RouteResolution(
                route=keyword_decision.route,
                method="keyword_fallback",
                classifier_summary=(
                    f"prepass wanted action (disabled) → "
                    f"{keyword_decision.route.value}"
                ),
                classifier_detail=(
                    "Prepass returned action but action routing is disabled."
                ),
                keyword_hint=keyword_decision.summary,
                classifier_raw=parsed.route_resolution.classifier_raw,
            ),
            orch_plan=parsed.orch_plan,
            skill_selection=parsed.skill_selection,
            slot_bindings=parsed.slot_bindings,
            method="keyword_fallback",
        )
    return parsed

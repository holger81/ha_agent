"""Orchestrator: complexity triage and subtask planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .const import LOGGER
from .context import is_device_action_query, is_email_query, is_news_query
from .llm_client import LlmClient
from .llm_telemetry import record_llm_call
from .role_registry import ModelRole, RoleRegistry
from .structured_output import (
    COMPLEXITY_SCHEMA,
    PLAN_SUBTASKS_SCHEMA,
    json_schema_format,
)

_COMPLEXITY_PROMPT = (
    "Classify request complexity for a Home Assistant agent.\n"
    'Return ONLY JSON: {"complexity": "simple"|"single"|"complex", '
    '"routes": ["chat"|"email"|"news"|"action", ...], "reason": "..."}.\n'
    "- simple: greeting, joke, factual chat, no tools.\n"
    "- single: one domain, one workflow (email OR device OR news).\n"
    "- complex: multiple domains, multiple unrelated goals, or AND-chained tasks."
)

_REPLAN_PROMPT = (
    "A worker subtask failed during a multi-step Home Assistant plan.\n"
    "Return a revised remainder plan as JSON subtasks only for unfinished work.\n"
    'Return ONLY JSON: {"subtasks": [{"id": "t1", "subgoal": "...", '
    '"route": "email|news|action|chat", "depends_on": []}]}. '
    "Max 4 subtasks."
)

_PLAN_PROMPT = (
    "Decompose a complex user request into ordered subtasks for specialized workers.\n"
    'Return ONLY JSON: {"subtasks": [{"id": "t1", "subgoal": "...", '
    '"route": "email|news|action|chat", "depends_on": []}]}. '
    "Max 4 subtasks. depends_on lists prior subtask ids."
)


class Complexity(StrEnum):
    """Turn complexity tier."""

    SIMPLE = "simple"
    SINGLE = "single"
    COMPLEX = "complex"


@dataclass(slots=True)
class SubtaskSpec:
    """One worker subtask in an orchestration plan."""

    id: str
    subgoal: str
    route: str
    depends_on: list[str] = field(default_factory=list)
    skill_slug: str | None = None
    slot_bindings: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestrationPlan:
    """Planner output for a complex turn."""

    complexity: Complexity
    subtasks: list[SubtaskSpec] = field(default_factory=list)
    reason: str = ""
    routes: list[str] = field(default_factory=list)


def _strip_json(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def heuristic_complexity(user_text: str) -> Complexity:
    """Fast complexity estimate without LLM."""
    text = user_text.lower()
    domains = 0
    if is_email_query(user_text):
        domains += 1
    if is_news_query(user_text):
        domains += 1
    if is_device_action_query(user_text):
        domains += 1
    if re.search(r"\band\b", text) and domains >= 1:
        return Complexity.COMPLEX
    if domains >= 2:
        return Complexity.COMPLEX
    if domains == 1:
        return Complexity.SINGLE
    if len(text.split()) <= 6 and not re.search(
        r"\b(mail|email|news|light|turn|open|close|inbox)\b", text
    ):
        return Complexity.SIMPLE
    return Complexity.SINGLE


def _parse_subtasks_payload(
    data: dict[str, Any] | None,
    *,
    user_text: str,
    fallback_route: str = "chat",
) -> list[SubtaskSpec]:
    subtasks_raw = data.get("subtasks") if isinstance(data, dict) else None
    if not isinstance(subtasks_raw, list) or not subtasks_raw:
        return [SubtaskSpec(id="t1", subgoal=user_text, route=fallback_route)]

    subtasks: list[SubtaskSpec] = []
    for index, item in enumerate(subtasks_raw[:4]):
        if not isinstance(item, dict):
            continue
        subtasks.append(
            SubtaskSpec(
                id=str(item.get("id") or f"t{index + 1}"),
                subgoal=str(item.get("subgoal") or user_text),
                route=str(item.get("route") or fallback_route),
                depends_on=[
                    str(d) for d in (item.get("depends_on") or []) if d
                ],
            )
        )
    return subtasks or [
        SubtaskSpec(id="t1", subgoal=user_text, route=fallback_route)
    ]


async def triage_complexity(
    llm: LlmClient,
    registry: RoleRegistry,
    *,
    user_text: str,
    history: list[dict[str, str]] | None = None,
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> OrchestrationPlan:
    """Classify turn complexity; LLM only when heuristic suggests complex."""
    hint = heuristic_complexity(user_text)
    if hint != Complexity.COMPLEX:
        return OrchestrationPlan(complexity=hint, reason="heuristic")
    backend = registry.backend_for(ModelRole.PLANNER)
    messages = [
        {"role": "system", "content": _COMPLEXITY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "heuristic": hint.value,
                    "recent": (history or [])[-4:],
                },
                ensure_ascii=True,
            ),
        },
    ]
    response_format = (
        json_schema_format("complexity", COMPLEXITY_SCHEMA, strict=False)
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
        record_llm_call(trace, role="planner_triage", backend=backend, result=result)
        data = json.loads(_strip_json(result.content or ""))
    except Exception as err:
        LOGGER.debug("Complexity triage LLM failed: %s", err)
        record_llm_call(trace, role="planner_triage", backend=backend, error=str(err))
        return OrchestrationPlan(complexity=hint, reason="heuristic fallback")

    if not isinstance(data, dict):
        return OrchestrationPlan(complexity=hint, reason="heuristic fallback")

    raw = str(data.get("complexity", hint.value)).lower()
    try:
        complexity = Complexity(raw)
    except ValueError:
        complexity = hint
    routes_raw = data.get("routes")
    routes = (
        [str(r) for r in routes_raw if isinstance(routes_raw, list)]
        if isinstance(routes_raw, list)
        else []
    )
    return OrchestrationPlan(
        complexity=complexity,
        reason=str(data.get("reason", "")),
        routes=routes,
    )


async def plan_subtasks(
    llm: LlmClient,
    registry: RoleRegistry,
    *,
    user_text: str,
    plan: OrchestrationPlan,
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> OrchestrationPlan:
    """Decompose a complex turn into subtasks."""
    if plan.complexity != Complexity.COMPLEX:
        return plan

    backend = registry.backend_for(ModelRole.PLANNER)
    messages = [
        {"role": "system", "content": _PLAN_PROMPT},
        {
            "role": "user",
            "content": json.dumps({"user_text": user_text}, ensure_ascii=True),
        },
    ]
    response_format = (
        json_schema_format("plan_subtasks", PLAN_SUBTASKS_SCHEMA, strict=False)
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
        record_llm_call(trace, role="planner", backend=backend, result=result)
        data = json.loads(_strip_json(result.content or ""))
    except Exception as err:
        LOGGER.warning("Planner LLM failed: %s", err)
        record_llm_call(trace, role="planner", backend=backend, error=str(err))
        fallback_route = plan.routes[0] if plan.routes else "chat"
        plan.subtasks = [
            SubtaskSpec(id="t1", subgoal=user_text, route=fallback_route)
        ]
        return plan

    fallback_route = plan.routes[0] if plan.routes else "chat"
    plan.subtasks = _parse_subtasks_payload(
        data if isinstance(data, dict) else None,
        user_text=user_text,
        fallback_route=fallback_route,
    )
    return plan


async def replan_after_failure(
    llm: LlmClient,
    registry: RoleRegistry,
    *,
    user_text: str,
    plan: OrchestrationPlan,
    failed_subtask: SubtaskSpec,
    completed_summaries: list[dict[str, str]],
    structured_output_enabled: bool = True,
    trace: Any | None = None,
) -> OrchestrationPlan:
    """Revise the remainder of a complex plan after a worker step fails."""
    backend = registry.backend_for(ModelRole.PLANNER)
    messages = [
        {"role": "system", "content": _REPLAN_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_request": user_text,
                    "failed_subtask": {
                        "id": failed_subtask.id,
                        "subgoal": failed_subtask.subgoal,
                        "route": failed_subtask.route,
                    },
                    "completed": completed_summaries,
                },
                ensure_ascii=True,
            ),
        },
    ]
    response_format = (
        json_schema_format("replan_subtasks", PLAN_SUBTASKS_SCHEMA, strict=False)
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
        record_llm_call(trace, role="replan", backend=backend, result=result)
        data = json.loads(_strip_json(result.content or ""))
    except Exception as err:
        LOGGER.warning("Replan LLM failed: %s", err)
        record_llm_call(trace, role="replan", backend=backend, error=str(err))
        return plan

    plan.subtasks = _parse_subtasks_payload(
        data if isinstance(data, dict) else None,
        user_text=failed_subtask.subgoal,
        fallback_route=failed_subtask.route,
    )
    plan.reason = "replan"
    return plan

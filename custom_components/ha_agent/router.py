"""Route user requests to chat or action LLM backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .config_helpers import LlmBackend, RouterConfig
from .const import LOGGER
from .context import (
    entity_matches_query,
    route_keyword_match,
)
from .structured_output import ROUTE_SCHEMA, json_schema_format

if TYPE_CHECKING:
    from .llm_client import LlmClient
    from .skills.models import TurnTrace


class TaskRoute(StrEnum):
    """Agent loop backend selection for one user turn."""

    CHAT = "chat"
    HA_ACTION = "action"


@dataclass(frozen=True)
class RouteDecision:
    """Keyword routing outcome for one user turn."""

    route: TaskRoute
    method: str
    detail: str
    domain_hint: str | None = None

    @property
    def summary(self) -> str:
        """Human-readable classification label for the chat UI."""
        if self.method == "default":
            return "default chat (no route keyword)"
        if self.method == "domain_hint":
            return f"domain hint → {self.domain_hint} ({self.detail})"
        if self.method == "follow_up":
            label = self.domain_hint or self.route.value
            return f"follow-up → {label} ({self.detail})"
        return f"keyword → {self.route.value} ({self.detail})"


_ROUTE_CLASSIFIER_PROMPT = (
    "You classify the user's latest request into exactly one agent route.\n"
    'Return ONLY valid JSON: {{"route": "chat"|"action"}}.\n'
    "Rules:\n"
    "- chat: greetings, jokes, chitchat, general questions, email, news, "
    "capabilities, and anything that is not device control.\n"
    "- action: control devices (lights, covers, locks, climate) or camera "
    "snapshots.\n"
    "- Pick chat when the request is not clearly device control."
)

_ROUTE_CLASSIFIER_CATALOG: tuple[tuple[str, str, str], ...] = (
    (
        "chat",
        "General chat",
        "Greetings, jokes, chitchat, email, news, general knowledge, and "
        "requests that are not device control.",
    ),
    (
        "action",
        "Device action",
        "The user asks to control or check a device, such as lights, "
        "switches, covers, locks, climate, or a camera snapshot.",
    ),
)

_ROUTE_VALUE_TO_TASK: dict[str, TaskRoute] = {
    "chat": TaskRoute.CHAT,
    "action": TaskRoute.HA_ACTION,
}


@dataclass(frozen=True)
class RouteResolution:
    """Final route for a turn plus classifier and keyword context."""

    route: TaskRoute
    method: str
    classifier_summary: str
    classifier_detail: str
    keyword_hint: str
    classifier_raw: str | None = None
    domain_hint: str | None = None


def task_route_for_skill_scope(scope: str | None) -> TaskRoute | None:
    """Map a skill ``route_scope`` to the turn TaskRoute.

    - ``action`` → action backend
    - any other non-empty scope (email, news, custom domains) → chat
    - empty/None → does not decide (caller keeps classifier route)
    """
    key = (scope or "").strip().lower()
    if not key:
        return None
    if key == TaskRoute.HA_ACTION.value:
        return TaskRoute.HA_ACTION
    return TaskRoute.CHAT


def _concrete_skill_tool_names(tool_steps: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for step in tool_steps or []:
        if not isinstance(step, dict):
            continue
        name = str(step.get("toolName") or "").strip().lower()
        if not name:
            continue
        if "searchtool" in name or "searchtoolsfordomain" in name:
            continue
        names.append(name)
    return names


def _tools_are_status_lookup(tool_names: list[str]) -> bool:
    """True when every concrete tool is entity search/state (no control)."""
    if not tool_names:
        return False
    lookup_markers = (
        "ha_search",
        "get_state",
        "get_entity",
        "ha_get_history",
        "ha_get_overview",
    )
    control_markers = (
        "call_service",
        "bulk_control",
        "turn_on",
        "turn_off",
        "set_position",
        "set_volume",
        "media_",
        "light_set",
        "fan_set",
        "hass",
    )
    if any(any(marker in name for marker in control_markers) for name in tool_names):
        return False
    return all(any(marker in name for marker in lookup_markers) for name in tool_names)


def effective_skill_route_scope(
    scope: str | None,
    tool_steps: list[dict[str, Any]] | None = None,
) -> str | None:
    """Normalize skill scope for routing.

    Lookup-only skills mis-tagged as ``action`` (common for distilled status
    skills) run on chat so readings are not forced through the action backend.
    """
    key = (scope or "").strip() or None
    if not key:
        return None
    if key.lower() == TaskRoute.HA_ACTION.value and _tools_are_status_lookup(
        _concrete_skill_tool_names(tool_steps)
    ):
        return "chat"
    return key


def align_route_to_skill(
    route: TaskRoute,
    *,
    skill_scope: str | None = None,
    domain_hint: str | None = None,
    skill_tool_steps: list[dict[str, Any]] | None = None,
) -> tuple[TaskRoute, str | None, str | None]:
    """Align turn route with a matched skill / soft domain hint.

    Returns ``(route, domain_hint, reason_suffix)``. Selected skills own the
    route via ``route_scope``; soft domain hints always run on chat.
    """
    reason: str | None = None
    hint = (domain_hint or "").strip().lower() or None
    raw_scope = (skill_scope or "").strip().lower() or None
    scope = effective_skill_route_scope(skill_scope, skill_tool_steps)
    if scope:
        scope = scope.strip().lower() or None

    skill_route = task_route_for_skill_scope(scope)
    if skill_route is not None:
        if scope and scope != TaskRoute.HA_ACTION.value:
            hint = hint or scope
        if skill_route != route:
            if raw_scope == TaskRoute.HA_ACTION.value and scope == "chat":
                reason = "aligned route to chat for lookup-only skill"
            else:
                reason = f"aligned route to skill scope {scope!r}"
            route = skill_route
    elif hint and route == TaskRoute.HA_ACTION:
        # Soft domain hints (any value) are chat workflows, not device control.
        route = TaskRoute.CHAT
        reason = f"aligned action→chat for domain hint {hint!r}"

    return route, hint, reason


def parse_route_classifier_response(content: str) -> str | None:
    """Parse the route value from a classifier LLM response."""
    text = (content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    route = data.get("route")
    if isinstance(route, str) and route.strip() in _ROUTE_VALUE_TO_TASK:
        return route.strip()
    return None


def _route_classifier_catalog(router_config: RouterConfig) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for route, title, when in _ROUTE_CLASSIFIER_CATALOG:
        if route == "action" and not (
            router_config.action_enabled and router_config.action_backend
        ):
            continue
        entries.append({"route": route, "title": title, "when_to_apply": when})
    return entries


async def select_route_with_llm(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    history: list[dict[str, str]] | None,
    router_config: RouterConfig,
    structured_output_enabled: bool = True,
    trace: TurnTrace | None = None,
) -> tuple[str | None, str]:
    """Ask the classifier model which route applies; return (route, raw)."""
    from .llm_telemetry import record_llm_call

    catalog = _route_classifier_catalog(router_config)
    if not catalog:
        return None, ""
    recent = [
        turn.get("content", "")
        for turn in (history or [])[-4:]
        if turn.get("role") == "user"
    ]
    messages = [
        {"role": "system", "content": _ROUTE_CLASSIFIER_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "recent_user_turns": recent,
                    "available_routes": catalog,
                },
                ensure_ascii=True,
            ),
        },
    ]
    response_format = (
        json_schema_format("route", ROUTE_SCHEMA) if structured_output_enabled else None
    )
    try:
        result = await llm.chat(
            messages,
            backend,
            tools=[],
            response_format=response_format,
        )
        record_llm_call(trace, role="router", backend=backend, result=result)
    except Exception as err:
        LOGGER.warning("Route classifier LLM call failed: %s", err)
        record_llm_call(trace, role="router", backend=backend, error=str(err))
        return None, ""
    raw = (result.content or "").strip()
    return parse_route_classifier_response(raw), raw


async def resolve_route_with_classifier(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    exposed_entities: list[dict],
    router_config: RouterConfig,
    route_keywords: dict[str, list[str]] | None = None,
    history: list[dict[str, str]] | None = None,
    structured_output_enabled: bool = True,
    trace: TurnTrace | None = None,
) -> RouteResolution:
    """Pick a route via the classifier LLM, falling back to keyword rules."""
    keyword_decision = classify_route_with_detail(
        user_text,
        exposed_entities,
        router_config,
        route_keywords=route_keywords,
        history=history,
    )
    llm_route_value, raw = await select_route_with_llm(
        llm,
        backend,
        user_text=user_text,
        history=history,
        router_config=router_config,
        structured_output_enabled=structured_output_enabled,
        trace=trace,
    )
    raw_preview = raw[:240] if raw else None
    if llm_route_value:
        route = _ROUTE_VALUE_TO_TASK[llm_route_value]
        if route == TaskRoute.HA_ACTION and not (
            router_config.action_enabled and router_config.action_backend
        ):
            return RouteResolution(
                route=keyword_decision.route,
                method="keyword_fallback",
                classifier_summary=(
                    f"LLM wanted action (disabled) → {keyword_decision.route.value}"
                ),
                classifier_detail=(
                    "Classifier returned action but action routing is disabled; "
                    f"using keyword route {keyword_decision.route.value} "
                    f"({keyword_decision.summary})."
                ),
                keyword_hint=keyword_decision.summary,
                classifier_raw=raw_preview,
                domain_hint=keyword_decision.domain_hint,
            )
        # Domain hints come from matched skills (route_scope), not keywords.
        return RouteResolution(
            route=route,
            method="llm",
            classifier_summary=f"LLM → {route.value}",
            classifier_detail=(
                f"Classifier picked {route.value}. "
                f"Keyword hint: {keyword_decision.summary}."
            ),
            keyword_hint=keyword_decision.summary,
            classifier_raw=raw_preview,
            domain_hint=None,
        )

    return RouteResolution(
        route=keyword_decision.route,
        method="keyword_fallback",
        classifier_summary=f"keyword fallback → {keyword_decision.route.value}",
        classifier_detail=(
            "Classifier returned no valid route; "
            f"using keyword rules ({keyword_decision.summary})."
        ),
        keyword_hint=keyword_decision.summary,
        classifier_raw=raw_preview,
        domain_hint=keyword_decision.domain_hint,
    )


def classify_route_with_detail(
    user_text: str,
    exposed_entities: list[dict],
    router_config: RouterConfig,
    *,
    route_keywords: dict[str, list[str]] | None = None,
    history: list[dict[str, str]] | None = None,
) -> RouteDecision:
    """Pick the route for this user turn and explain how it was chosen.

    ``route_keywords`` carries optional action keyword overrides
    (``{"action": [...]}``). Soft domain workflows (email, news, …) are
    owned by skills via triggers / ``route_scope``, not keyword routes.
    """
    del exposed_entities  # reserved for future entity-aware routing
    del history  # follow-ups no longer invent domain routes
    overrides = route_keywords or {}

    if (
        router_config.action_enabled
        and router_config.action_backend
        and (match := route_keyword_match(user_text, "action", overrides.get("action")))
    ):
        return RouteDecision(TaskRoute.HA_ACTION, "keyword", match)

    return RouteDecision(TaskRoute.CHAT, "default", "general chat")


def classify_route(
    user_text: str,
    exposed_entities: list[dict],
    router_config: RouterConfig,
    *,
    route_keywords: dict[str, list[str]] | None = None,
    history: list[dict[str, str]] | None = None,
) -> TaskRoute:
    """Pick the route for this user turn."""
    return classify_route_with_detail(
        user_text,
        exposed_entities,
        router_config,
        route_keywords=route_keywords,
        history=history,
    ).route


def has_exposed_match(user_text: str, exposed_entities: list[dict]) -> bool:
    """Return True when an exposed entity matches the user query."""
    return any(entity_matches_query(entity, user_text) for entity in exposed_entities)


def backend_for_route(
    route: TaskRoute,
    *,
    chat_backend: LlmBackend,
    router_config: RouterConfig,
    prefer_action: bool = True,
) -> LlmBackend:
    """Return the LLM backend for the active route (action or chat)."""
    if prefer_action and route == TaskRoute.HA_ACTION and router_config.action_backend:
        return router_config.action_backend
    return chat_backend

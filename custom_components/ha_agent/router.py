"""Route user requests to chat or action LLM backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .config_helpers import LlmBackend, RouterConfig
from .const import LOGGER
from .context import (
    _recent_email_context,
    _recent_news_context,
    entity_matches_query,
    is_email_query,
    is_informational_follow_up,
    is_news_query,
    route_keyword_match,
)
from .playbooks import default_playbook_body, playbook_key_for_route
from .structured_output import ROUTE_SCHEMA, json_schema_format

if TYPE_CHECKING:
    from .llm_client import LlmClient
    from .skills.models import TurnTrace


class TaskRoute(StrEnum):
    """Agent loop backend and playbook selection."""

    CHAT = "chat"
    HA_ACTION = "action"
    EMAIL = "email"
    NEWS = "news"


@dataclass(frozen=True)
class RouteDecision:
    """Keyword routing outcome for one user turn."""

    route: TaskRoute
    method: str
    detail: str

    @property
    def summary(self) -> str:
        """Human-readable classification label for the chat UI."""
        if self.method == "default":
            return "default chat (no route keyword)"
        if self.method == "follow_up":
            return f"follow-up → {self.route.value} ({self.detail})"
        return f"keyword → {self.route.value} ({self.detail})"


_ROUTE_CLASSIFIER_PROMPT = (
    "You classify the user's latest request into exactly one agent route.\n"
    'Return ONLY valid JSON: {{"route": "chat"|"email"|"news"|"action"}}.\n'
    "Rules:\n"
    "- chat: greetings, jokes, chitchat, general questions, capabilities.\n"
    "- email: inbox, mail, unread messages, reading or searching email.\n"
    "- news: headlines, news briefings, RSS, current events.\n"
    "- action: control devices (lights, covers, locks, climate) or camera "
    "snapshots.\n"
    "- Use recent user turns for follow-ups (e.g. 'tell me more' after news "
    "stays news).\n"
    "- Pick chat when none of the specialized routes clearly apply."
)

_ROUTE_CLASSIFIER_CATALOG: tuple[tuple[str, str, str], ...] = (
    (
        "chat",
        "General chat",
        "Greetings, jokes, chitchat, general knowledge, and requests that "
        "do not need email, news, or device tools.",
    ),
    (
        "email",
        "Email",
        "The user asks about email, mail, inbox, or unread messages.",
    ),
    (
        "news",
        "News",
        "The user asks for news, headlines, or a briefing.",
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
    "email": TaskRoute.EMAIL,
    "news": TaskRoute.NEWS,
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
        json_schema_format("route", ROUTE_SCHEMA)
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
                    f"LLM wanted action (disabled) → "
                    f"{keyword_decision.route.value}"
                ),
                classifier_detail=(
                    "Classifier returned action but action routing is disabled; "
                    f"using keyword route {keyword_decision.route.value} "
                    f"({keyword_decision.summary})."
                ),
                keyword_hint=keyword_decision.summary,
                classifier_raw=raw_preview,
            )
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

    ``route_keywords`` carries optional per-route UI keyword overrides
    (``{"email": [...], "news": [...], "action": [...]}``). A route absent
    from the map uses its shipped default matcher.
    """
    del exposed_entities  # reserved for future entity-aware routing
    overrides = route_keywords or {}
    prior = history or []
    if match := route_keyword_match(user_text, "email", overrides.get("email")):
        return RouteDecision(TaskRoute.EMAIL, "keyword", match)

    if match := route_keyword_match(user_text, "news", overrides.get("news")):
        return RouteDecision(TaskRoute.NEWS, "keyword", match)

    if (
        router_config.action_enabled
        and router_config.action_backend
        and (
            match := route_keyword_match(
                user_text, "action", overrides.get("action")
            )
        )
    ):
        return RouteDecision(TaskRoute.HA_ACTION, "keyword", match)

    if (
        _recent_news_context(prior)
        and is_informational_follow_up(user_text)
        and not is_email_query(user_text, overrides.get("email"))
    ):
        return RouteDecision(
            TaskRoute.NEWS,
            "follow_up",
            "recent news context",
        )

    if (
        _recent_email_context(prior)
        and is_informational_follow_up(user_text)
        and not is_news_query(user_text, overrides.get("news"))
    ):
        return RouteDecision(
            TaskRoute.EMAIL,
            "follow_up",
            "recent email context",
        )

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


def route_playbook(route: TaskRoute) -> str:
    """Return the default route playbook text (UI-editable overrides win).

    This is the shipped fallback used when no per-entry override exists; the
    editable copy lives in the playbook store. See ``playbooks.py``.
    """
    return default_playbook_body(playbook_key_for_route(route.value))


def backend_for_route(
    route: TaskRoute,
    *,
    chat_backend: LlmBackend,
    router_config: RouterConfig,
    prefer_action: bool = True,
) -> LlmBackend:
    """Return the LLM backend for the active route."""
    if (
        prefer_action
        and route == TaskRoute.HA_ACTION
        and router_config.action_backend
    ):
        return router_config.action_backend
    if route == TaskRoute.EMAIL and router_config.email_backend:
        return router_config.email_backend
    if route == TaskRoute.NEWS and router_config.news_backend:
        return router_config.news_backend
    return chat_backend

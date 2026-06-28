"""Route user requests to chat or action LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config_helpers import LlmBackend, RouterConfig
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

"""Route user requests to chat or action LLM backends."""

from __future__ import annotations

from enum import StrEnum

from .config_helpers import LlmBackend, RouterConfig
from .context import (
    entity_matches_query,
    is_device_action_query,
    is_email_query,
    is_news_query,
)
from .playbooks import default_playbook_body, playbook_key_for_route


class TaskRoute(StrEnum):
    """Agent loop backend and playbook selection."""

    CHAT = "chat"
    HA_ACTION = "action"
    EMAIL = "email"
    NEWS = "news"


def classify_route(
    user_text: str,
    exposed_entities: list[dict],
    router_config: RouterConfig,
    *,
    route_keywords: dict[str, list[str]] | None = None,
) -> TaskRoute:
    """Pick the route for this user turn.

    ``route_keywords`` carries optional per-route UI keyword overrides
    (``{"email": [...], "news": [...], "action": [...]}``). A route absent
    from the map uses its shipped default matcher.
    """
    overrides = route_keywords or {}
    if is_email_query(user_text, overrides.get("email")):
        return TaskRoute.EMAIL

    if is_news_query(user_text, overrides.get("news")):
        return TaskRoute.NEWS

    if (
        router_config.action_enabled
        and router_config.action_backend
        and is_device_action_query(user_text, overrides.get("action"))
    ):
        return TaskRoute.HA_ACTION

    return TaskRoute.CHAT


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
) -> LlmBackend:
    """Return the LLM backend for the active route."""
    if route == TaskRoute.HA_ACTION and router_config.action_backend:
        return router_config.action_backend
    return chat_backend

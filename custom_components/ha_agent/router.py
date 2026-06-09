"""Route user requests to chat or action LLM backends."""

from __future__ import annotations

from enum import StrEnum

from .config_helpers import LlmBackend, RouterConfig
from .context import entity_matches_query, is_device_action_query


class TaskRoute(StrEnum):
    """Agent loop backend selection."""

    CHAT = "chat"
    HA_ACTION = "action"


def classify_route(
    user_text: str,
    exposed_entities: list[dict],
    router_config: RouterConfig,
) -> TaskRoute:
    """Pick the backend route for this user turn."""
    if not router_config.action_enabled or not router_config.action_backend:
        return TaskRoute.CHAT

    if not is_device_action_query(user_text):
        return TaskRoute.CHAT

    return TaskRoute.HA_ACTION


def has_exposed_match(user_text: str, exposed_entities: list[dict]) -> bool:
    """Return True when an exposed entity matches the user query."""
    return any(entity_matches_query(entity, user_text) for entity in exposed_entities)


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

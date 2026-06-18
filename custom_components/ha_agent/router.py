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
) -> TaskRoute:
    """Pick the route for this user turn."""
    if is_email_query(user_text):
        return TaskRoute.EMAIL

    if is_news_query(user_text):
        return TaskRoute.NEWS

    if (
        router_config.action_enabled
        and router_config.action_backend
        and is_device_action_query(user_text)
    ):
        return TaskRoute.HA_ACTION

    return TaskRoute.CHAT


def has_exposed_match(user_text: str, exposed_entities: list[dict]) -> bool:
    """Return True when an exposed entity matches the user query."""
    return any(entity_matches_query(entity, user_text) for entity in exposed_entities)


def route_playbook(route: TaskRoute) -> str:
    """Return route-specific workflow guidance for the system prompt."""
    if route == TaskRoute.EMAIL:
        return (
            "EMAIL PLAYBOOK:\n"
            "1. Discover tools in domain email if needed.\n"
            "2. Call mailbox_status for unseen count.\n"
            "3. Call search_messages with unread_only=true and a small limit.\n"
            "4. Call get_message only for messages you will cite.\n"
            "5. Answer using tool results only; never invent subjects or counts."
        )
    if route == TaskRoute.NEWS:
        return (
            "NEWS PLAYBOOK:\n"
            "1. Call mcp_news__news_curate with {\"limit\": 5}.\n"
            "2. Summarize headlines from that result only.\n"
            "3. Use searchToolsForDomain only if news_curate fails."
        )
    if route == TaskRoute.HA_ACTION:
        return (
            "DEVICE PLAYBOOK:\n"
            "1. Prefer an exposed-entity shortcut when one clearly matches.\n"
            "2. If no shortcut fits, discover entities in domain smart-home "
            "with searchToolsForDomain, then callTool.\n"
            "3. Call ha_call_service with domain, service, and entity_id "
            "(e.g. camera.snapshot for photos).\n"
            "4. Read VERIFICATION lines in tool results before telling the user "
            "the action succeeded."
        )
    return (
        "GENERAL PLAYBOOK:\n"
        "Gather evidence with tools before answering. Cite tool results. "
        "Exposed entities in context are shortcuts only; discover more in domain "
        "smart-home when needed. If a tool fails, change strategy using "
        "RECOVERY HINTS."
    )


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

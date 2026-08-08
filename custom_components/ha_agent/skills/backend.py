"""Resolve the LLM backend for a matched skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config_helpers import LlmBackend, RouterConfig
from ..router import TaskRoute

if TYPE_CHECKING:
    from .models import Skill


def backend_for_skill(
    skill: Skill | None,
    *,
    route: TaskRoute | str,
    chat_backend: LlmBackend,
    action_backend: LlmBackend | None = None,
    router_config: RouterConfig | None = None,
) -> LlmBackend:
    """Pick the worker backend for a turn given route + optional matched skill.

    Resolution order:
    1. Skill ``llm_model`` (optional ``llm_base_url``, else chat base URL)
    2. Legacy entry email/news backends when skill ``route_scope`` matches
    3. Action backend when route is action
    4. Chat backend
    """
    route_value = route.value if isinstance(route, TaskRoute) else str(route or "")
    route_value = route_value.lower()

    if skill is not None:
        model = (skill.llm_model or "").strip()
        if model:
            base = (skill.llm_base_url or "").strip().rstrip("/")
            if not base:
                base = chat_backend.base_url
            return LlmBackend(
                base_url=base,
                model=model,
                api_key=chat_backend.api_key,
                max_tokens=chat_backend.max_tokens,
                temperature=chat_backend.temperature,
                timeout=chat_backend.timeout,
                thinking_level="off",
            )
        scope = (skill.route_scope or "").strip().lower()
        if router_config is not None:
            if scope == "email" and router_config.email_backend is not None:
                return router_config.email_backend
            if scope == "news" and router_config.news_backend is not None:
                return router_config.news_backend

    if route_value == TaskRoute.HA_ACTION.value and action_backend is not None:
        return action_backend
    if (
        route_value == TaskRoute.HA_ACTION.value
        and router_config is not None
        and router_config.action_enabled
        and router_config.action_backend is not None
    ):
        return router_config.action_backend
    return chat_backend

"""Model role registry for orchestrated multi-model agent turns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_helpers import LlmBackend, RouterConfig


class ModelRole(StrEnum):
    """Logical LLM roles in the orchestrated agent loop."""

    ROUTER = "router"
    PLANNER = "planner"
    VERIFIER = "verifier"
    OBSERVER = "observer"
    WORKER_CHAT = "worker_chat"
    WORKER_ACTION = "worker_action"
    WORKER_EMAIL = "worker_email"
    WORKER_NEWS = "worker_news"


ROLE_CAPABILITIES: dict[ModelRole, frozenset[str]] = {
    ModelRole.ROUTER: frozenset({"fast", "classification"}),
    ModelRole.PLANNER: frozenset({"reasoning", "decomposition"}),
    ModelRole.VERIFIER: frozenset({"fast", "critique"}),
    ModelRole.OBSERVER: frozenset({"fast", "distillation"}),
    ModelRole.WORKER_CHAT: frozenset({"tool_use", "chat"}),
    ModelRole.WORKER_ACTION: frozenset({"tool_use", "action"}),
    ModelRole.WORKER_EMAIL: frozenset({"tool_use", "email"}),
    ModelRole.WORKER_NEWS: frozenset({"tool_use", "news"}),
}


@dataclass(frozen=True, slots=True)
class RoleRegistry:
    """Maps orchestration roles to concrete LLM backends."""

    chat_backend: LlmBackend
    roles: dict[ModelRole, LlmBackend]
    action_enabled: bool = False

    def backend_for(self, role: ModelRole) -> LlmBackend:
        """Return the backend for a role, falling back to chat."""
        return self.roles.get(role, self.chat_backend)

    def worker_for_route(self, route_value: str) -> LlmBackend:
        """Return the worker backend for a router route value."""
        mapping = {
            "chat": ModelRole.WORKER_CHAT,
            "action": ModelRole.WORKER_ACTION,
            "email": ModelRole.WORKER_EMAIL,
            "news": ModelRole.WORKER_NEWS,
        }
        return self.backend_for(mapping.get(route_value, ModelRole.WORKER_CHAT))

    def capabilities(self, role: ModelRole) -> frozenset[str]:
        """Return capability tags for a role."""
        return ROLE_CAPABILITIES.get(role, frozenset())

    def chip_for(self, role: ModelRole) -> dict[str, str]:
        """Return model/host chip metadata for UI."""
        backend = self.backend_for(role)
        host = backend.base_url.split("//", 1)[-1].split("/", 1)[0]
        return {"model": backend.model, "host": host, "role": role.value}


def build_role_registry(
    chat_backend: LlmBackend,
    router_config: RouterConfig,
) -> RoleRegistry:
    """Build a role registry from legacy per-route config (backwards compatible)."""
    classifier = router_config.classifier_backend or chat_backend
    planner = router_config.planner_backend or classifier
    verifier = router_config.verifier_backend or classifier
    observer = router_config.observer_backend or classifier
    action = (
        router_config.action_backend
        if router_config.action_enabled and router_config.action_backend
        else chat_backend
    )
    email = router_config.email_backend or chat_backend
    news = router_config.news_backend or chat_backend
    roles = {
        ModelRole.ROUTER: classifier,
        ModelRole.PLANNER: planner,
        ModelRole.VERIFIER: verifier,
        ModelRole.OBSERVER: observer,
        ModelRole.WORKER_CHAT: chat_backend,
        ModelRole.WORKER_ACTION: action,
        ModelRole.WORKER_EMAIL: email,
        ModelRole.WORKER_NEWS: news,
    }
    return RoleRegistry(
        chat_backend=chat_backend,
        roles=roles,
        action_enabled=router_config.action_enabled,
    )

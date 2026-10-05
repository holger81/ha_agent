"""Agent user identity registry and resolution."""

from .models import AgentUser, ResolvedIdentity, UserKind
from .resolver import resolve_agent_user
from .runtime import (
    clear_identity_override,
    get_identity_override,
    set_identity_override,
)
from .store import (
    async_setup_identity_store,
    close_identity_store,
    get_identity_store,
)

__all__ = [
    "AgentUser",
    "ResolvedIdentity",
    "UserKind",
    "async_setup_identity_store",
    "clear_identity_override",
    "close_identity_store",
    "get_identity_override",
    "get_identity_store",
    "resolve_agent_user",
    "set_identity_override",
]

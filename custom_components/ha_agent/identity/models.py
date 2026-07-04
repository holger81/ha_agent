"""Data models for agent user identity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class UserKind(StrEnum):
    """Registered household member vs ephemeral guest."""

    REGISTERED = "registered"
    GUEST = "guest"


class IdentitySource(StrEnum):
    """How the active agent user was chosen."""

    OVERRIDE = "override"
    LOGIN = "login"
    VOICE = "voice"
    FALLBACK = "fallback"
    ASSIST_GUEST = "assist_guest"


@dataclass(slots=True)
class AgentUser:
    """A household member or guest profile."""

    id: str
    kind: UserKind
    display_name: str
    ha_user_id: str | None = None
    person_entity_id: str | None = None
    is_default: bool = False
    merged_into: str | None = None
    notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    last_seen_at: float | None = None


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """Resolved identity for one agent turn."""

    user: AgentUser
    source: IdentitySource
    ha_user_id: str | None = None
    override_by_ha_user_id: str | None = None
    speaker_confidence: float | None = None

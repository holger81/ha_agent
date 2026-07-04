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
    CORRECTED = "corrected"


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
class VoiceProfile:
    """Stored speaker embedding centroid for one agent user."""

    id: str
    agent_user_id: str
    backend: str
    model: str | None
    sample_count: int
    avg_confidence: float | None
    centroid: list[float] | None
    created_at: float
    updated_at: float
    last_seen_at: float | None = None


@dataclass(frozen=True, slots=True)
class SpeakerMatch:
    """Speaker embedding payload from STT / voice cache."""

    backend: str
    embedding: list[float] | None
    quality: str = "ok"
    model: str | None = None
    duration_ms: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """Resolved identity for one agent turn."""

    user: AgentUser
    source: IdentitySource
    ha_user_id: str | None = None
    override_by_ha_user_id: str | None = None
    speaker_confidence: float | None = None

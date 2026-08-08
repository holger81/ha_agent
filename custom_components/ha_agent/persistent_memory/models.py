"""Data models for durable agent memory (preferences, aliases, facts)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MemoryScope(StrEnum):
    """Who owns a memory entry."""

    SYSTEM = "system"
    USER = "user"


@dataclass(slots=True)
class MemoryEntry:
    """One durable key/value memory record."""

    key: str
    value: Any
    scope: MemoryScope
    agent_user_id: str | None = None
    route_scope: str | None = None
    updated_at: float = 0.0
    created_at: float = 0.0
    notes: str = ""


@dataclass(frozen=True, slots=True)
class MergedMemory:
    """Merged memory after applying precedence (user > system)."""

    values: dict[str, Any]
    sources: dict[str, MemoryScope]
    entries: list[MemoryEntry]

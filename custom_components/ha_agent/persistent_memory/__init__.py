"""Durable agent memory (preferences, aliases, household facts)."""

from __future__ import annotations

from .extract import ExtractedMemory, extract_memory_writes
from .inject import (
    async_load_memory_context,
    format_memory_context,
    load_merged_memory,
    should_include_user_memory,
)
from .intent import (
    MemoryIntent,
    MemoryIntentKind,
    detect_memory_intent,
    is_preference_shaped_turn,
)
from .models import MemoryEntry, MemoryScope, MergedMemory
from .runtime import (
    apply_memory_defaults_to_slots,
    async_handle_memory_intent,
)
from .store import (
    PersistentMemoryStore,
    close_persistent_memory_store,
    get_persistent_memory_store,
)

__all__ = [
    "ExtractedMemory",
    "MemoryEntry",
    "MemoryIntent",
    "MemoryIntentKind",
    "MemoryScope",
    "MergedMemory",
    "PersistentMemoryStore",
    "apply_memory_defaults_to_slots",
    "async_handle_memory_intent",
    "async_load_memory_context",
    "close_persistent_memory_store",
    "detect_memory_intent",
    "extract_memory_writes",
    "format_memory_context",
    "get_persistent_memory_store",
    "is_preference_shaped_turn",
    "load_merged_memory",
    "should_include_user_memory",
]

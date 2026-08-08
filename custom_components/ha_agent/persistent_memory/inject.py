"""Load and format durable memory for agent turns."""

from __future__ import annotations

import json
from typing import Any

from homeassistant.core import HomeAssistant

from ..identity.config import DEFAULT_GUEST_MATCH_THRESHOLD
from ..identity.models import ResolvedIdentity, UserKind
from .extract import keys_for_route
from .models import MemoryScope, MergedMemory
from .store import PersistentMemoryStore, get_persistent_memory_store


def should_include_user_memory(identity: ResolvedIdentity | None) -> bool:
    """Guests / low-confidence voice matches get system memory only."""
    if identity is None:
        return False
    if identity.user.kind != UserKind.REGISTERED:
        return False
    confidence = identity.speaker_confidence
    if confidence is not None and confidence < DEFAULT_GUEST_MATCH_THRESHOLD:
        # Low-confidence voice on a registered profile: still allow if source
        # is login/override (no speaker). Confidence only gates voice.
        from ..identity.models import IdentitySource

        if identity.source in {IdentitySource.VOICE}:
            return False
    return True


def load_merged_memory(
    store: PersistentMemoryStore,
    *,
    identity: ResolvedIdentity | None,
    route: str | None = None,
) -> MergedMemory:
    """Load merged memory for the active user and route."""
    include_user = should_include_user_memory(identity)
    agent_user_id = identity.user.id if identity is not None else None
    merged = store.merge_for_turn(
        agent_user_id=agent_user_id,
        include_user=include_user,
        route=route,
    )
    prefixes = keys_for_route(route)
    if not prefixes:
        return merged

    filtered_values: dict[str, Any] = {}
    filtered_sources: dict[str, MemoryScope] = {}
    filtered_entries = []
    for entry in merged.entries:
        if any(entry.key.startswith(prefix) for prefix in prefixes):
            filtered_values[entry.key] = entry.value
            filtered_sources[entry.key] = entry.scope
            filtered_entries.append(entry)
        elif entry.route_scope is None and not entry.key.startswith(
            ("news.", "email.", "entity.alias.", "ha.")
        ):
            # Global household facts
            filtered_values[entry.key] = entry.value
            filtered_sources[entry.key] = entry.scope
            filtered_entries.append(entry)
    return MergedMemory(
        values=filtered_values,
        sources=filtered_sources,
        entries=filtered_entries,
    )


def format_memory_context(merged: MergedMemory) -> str:
    """Compact JSON block for the system prompt."""
    if not merged.values:
        return ""
    payload = {
        key: {
            "value": value,
            "source": merged.sources.get(key, MemoryScope.SYSTEM).value,
        }
        for key, value in sorted(merged.values.items())
    }
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(body) > 1200:
        body = body[:1197] + "..."
    return (
        "DURABLE MEMORY (user overrides household; apply as defaults "
        f"when relevant):\n{body}"
    )


async def async_load_memory_context(
    hass: HomeAssistant,
    entry_id: str,
    *,
    identity: ResolvedIdentity | None,
    route: str | None = None,
) -> tuple[MergedMemory, str]:
    """Load and format memory for injection into the system message."""
    store = get_persistent_memory_store(hass, entry_id)

    def _load() -> MergedMemory:
        return load_merged_memory(store, identity=identity, route=route)

    merged = await hass.async_add_executor_job(_load)
    return merged, format_memory_context(merged)

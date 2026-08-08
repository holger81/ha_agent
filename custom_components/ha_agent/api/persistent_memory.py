"""Persistent memory API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..persistent_memory.models import MemoryEntry, MemoryScope
from ..persistent_memory.store import get_persistent_memory_store
from .serialize import memory_entry_to_dict


async def list_memory(
    hass: HomeAssistant,
    entry_id: str,
    *,
    scope: str = "system",
    agent_user_id: str | None = None,
) -> dict[str, Any]:
    """List durable memory entries for household or one user."""
    store = get_persistent_memory_store(hass, entry_id)
    parsed = MemoryScope(scope)

    def _load() -> list[MemoryEntry]:
        if parsed == MemoryScope.USER:
            if not agent_user_id:
                raise HomeAssistantError("agent_user_id is required for user memory")
            return store.list_user(agent_user_id)
        return store.list_system()

    try:
        entries = await hass.async_add_executor_job(_load)
    except HomeAssistantError:
        raise
    except Exception as err:
        raise HomeAssistantError(str(err)) from err
    return {
        "scope": parsed.value,
        "agent_user_id": agent_user_id,
        "entries": [memory_entry_to_dict(entry) for entry in entries],
    }


async def set_memory(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create or update one durable memory entry."""
    key = str(payload.get("key", "")).strip()
    if not key:
        raise HomeAssistantError("key is required")
    if "value" not in payload:
        raise HomeAssistantError("value is required")
    scope = MemoryScope(str(payload.get("scope", "system")))
    route_scope = payload.get("route_scope")
    notes = str(payload.get("notes", "") or "")
    store = get_persistent_memory_store(hass, entry_id)

    def _set() -> MemoryEntry:
        if scope == MemoryScope.USER:
            agent_user_id = str(payload.get("agent_user_id", "")).strip()
            if not agent_user_id:
                raise HomeAssistantError("agent_user_id is required for user memory")
            return store.set_user(
                agent_user_id,
                key,
                payload["value"],
                route_scope=route_scope,
                notes=notes,
            )
        return store.set_system(
            key,
            payload["value"],
            route_scope=route_scope,
            notes=notes,
        )

    try:
        entry = await hass.async_add_executor_job(_set)
    except HomeAssistantError:
        raise
    except Exception as err:
        raise HomeAssistantError(str(err)) from err
    return memory_entry_to_dict(entry)


async def delete_memory(
    hass: HomeAssistant,
    entry_id: str,
    *,
    scope: str,
    key: str,
    agent_user_id: str | None = None,
) -> bool:
    """Delete one durable memory entry."""
    store = get_persistent_memory_store(hass, entry_id)
    parsed = MemoryScope(scope)
    clean_key = key.strip()
    if not clean_key:
        raise HomeAssistantError("key is required")

    def _delete() -> bool:
        if parsed == MemoryScope.USER:
            if not agent_user_id:
                raise HomeAssistantError("agent_user_id is required for user memory")
            return store.delete_user(agent_user_id, clean_key)
        return store.delete_system(clean_key)

    try:
        deleted = await hass.async_add_executor_job(_delete)
    except HomeAssistantError:
        raise
    except Exception as err:
        raise HomeAssistantError(str(err)) from err
    if not deleted:
        raise HomeAssistantError(f"Memory key not found: {clean_key}")
    return True

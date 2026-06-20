"""Playbooks API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..playbooks import Playbook, get_playbook_store
from .serialize import playbook_to_dict


async def list_playbooks(hass: HomeAssistant, entry_id: str) -> list[dict[str, Any]]:
    """Return all editable route playbooks for an entry."""
    store = get_playbook_store(hass, entry_id)
    playbooks = await hass.async_add_executor_job(store.list_playbooks)
    return [playbook_to_dict(playbook) for playbook in playbooks]


async def update_playbook(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update an editable route playbook."""
    store = get_playbook_store(hass, entry_id)
    title = payload.get("title")
    body = payload.get("body")
    enabled = payload.get("enabled")

    def _update() -> Playbook | None:
        return store.update_playbook(
            route,
            title=None if title is None else str(title),
            body=None if body is None else str(body),
            enabled=None if enabled is None else bool(enabled),
        )

    playbook = await hass.async_add_executor_job(_update)
    if playbook is None:
        raise HomeAssistantError(f"Playbook not found: {route}")
    return playbook_to_dict(playbook)


async def set_playbook_enabled(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a route playbook."""
    store = get_playbook_store(hass, entry_id)

    def _set() -> Playbook | None:
        return store.update_playbook(route, enabled=enabled)

    playbook = await hass.async_add_executor_job(_set)
    if playbook is None:
        raise HomeAssistantError(f"Playbook not found: {route}")
    return playbook_to_dict(playbook)


async def reset_playbook(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
) -> dict[str, Any]:
    """Restore a route playbook to its shipped default."""
    store = get_playbook_store(hass, entry_id)

    def _reset() -> Playbook | None:
        return store.reset_playbook(route)

    playbook = await hass.async_add_executor_job(_reset)
    if playbook is None:
        raise HomeAssistantError(f"Playbook not found: {route}")
    return playbook_to_dict(playbook)

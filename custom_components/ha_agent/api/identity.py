"""Identity API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..identity.models import UserKind
from ..identity.runtime import (
    GLOBAL_OVERRIDE_KEY,
    get_identity_override,
    set_identity_override,
)
from ..identity.store import get_identity_store
from .serialize import agent_user_to_dict, voice_profile_to_dict


async def list_users(
    hass: HomeAssistant,
    entry_id: str,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Return agent users and the active console override."""
    store = get_identity_store(hass, entry_id)
    parsed_kind = UserKind(kind) if kind else None

    def _load() -> list[dict[str, Any]]:
        users = store.list_users(kind=parsed_kind)
        serialized = []
        for user in users:
            payload = agent_user_to_dict(user)
            profile = store.get_voice_profile_for_user(user.id)
            payload["voice_profile"] = (
                voice_profile_to_dict(profile) if profile is not None else None
            )
            serialized.append(payload)
        return serialized

    users = await hass.async_add_executor_job(_load)
    override_id = get_identity_override(hass, entry_id)
    return {
        "users": users,
        "override_user_id": override_id,
    }


async def update_user(
    hass: HomeAssistant,
    entry_id: str,
    user_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update one agent user."""
    store = get_identity_store(hass, entry_id)
    ha_user_id = payload.get("ha_user_id")
    person_entity_id = payload.get("person_entity_id")

    def _update():
        return store.update_user(
            user_id,
            display_name=payload.get("display_name"),
            ha_user_id=ha_user_id if "ha_user_id" in payload else _UNSET,
            person_entity_id=(
                person_entity_id if "person_entity_id" in payload else _UNSET
            ),
            is_default=payload.get("is_default"),
            notes=payload.get("notes"),
        )

    user = await hass.async_add_executor_job(_update)
    if user is None:
        raise HomeAssistantError("User not found or update rejected")
    return agent_user_to_dict(user)


async def create_guest(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a guest profile."""
    display_name = str(payload.get("display_name", "")).strip()
    if not display_name:
        raise HomeAssistantError("display_name is required")
    store = get_identity_store(hass, entry_id)

    def _create():
        return store.create_guest(display_name, notes=str(payload.get("notes") or ""))

    user = await hass.async_add_executor_job(_create)
    return agent_user_to_dict(user)


async def promote_guest(
    hass: HomeAssistant,
    entry_id: str,
    *,
    guest_id: str,
    registered_id: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Promote a guest voice profile onto a registered member."""
    store = get_identity_store(hass, entry_id)

    def _promote():
        try:
            user = store.promote_guest(
                guest_id,
                registered_id,
                display_name=display_name,
            )
            profile = store.get_voice_profile_for_user(user.id)
            return user, profile
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err

    user, profile = await hass.async_add_executor_job(_promote)
    payload = agent_user_to_dict(user)
    payload["voice_profile"] = (
        voice_profile_to_dict(profile) if profile is not None else None
    )
    return {"user": payload, "guest_id": guest_id}


async def merge_guests(
    hass: HomeAssistant,
    entry_id: str,
    *,
    guest_ids: list[str],
    survivor_id: str | None = None,
) -> dict[str, Any]:
    """Merge multiple guest profiles into one survivor guest."""
    store = get_identity_store(hass, entry_id)

    def _merge():
        try:
            user = store.merge_guests(guest_ids, survivor_id=survivor_id)
            profile = store.get_voice_profile_for_user(user.id)
            return user, profile
        except ValueError as err:
            raise HomeAssistantError(str(err)) from err

    user, profile = await hass.async_add_executor_job(_merge)
    payload = agent_user_to_dict(user)
    payload["voice_profile"] = (
        voice_profile_to_dict(profile) if profile is not None else None
    )
    return {"user": payload, "merged_guest_ids": guest_ids}


async def set_override(
    hass: HomeAssistant,
    entry_id: str,
    *,
    agent_user_id: str | None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Set or clear the console identity override."""
    if agent_user_id:
        store = get_identity_store(hass, entry_id)

        def _get():
            return store.get_user(agent_user_id)

        user = await hass.async_add_executor_job(_get)
        if user is None or user.merged_into:
            raise HomeAssistantError("Unknown agent user")
    set_identity_override(
        hass,
        entry_id,
        agent_user_id,
        conversation_id=conversation_id,
    )
    return {
        "agent_user_id": agent_user_id,
        "conversation_id": conversation_id or GLOBAL_OVERRIDE_KEY,
    }


async def get_override(
    hass: HomeAssistant,
    entry_id: str,
    *,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Return the active console override."""
    override_id = get_identity_override(
        hass,
        entry_id,
        conversation_id=conversation_id,
    )
    return {"agent_user_id": override_id}


_UNSET = object()

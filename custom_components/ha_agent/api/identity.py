"""Identity API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..activity import get_turn, update_turn_identity
from ..identity.models import UserKind
from ..identity.runtime import (
    GLOBAL_OVERRIDE_KEY,
    clear_enrollment_target,
    get_enrollment_session,
    get_identity_override,
    set_enrollment_target,
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
    session = get_enrollment_session(hass, entry_id)
    enrollment = None
    if session is not None:
        enrollment = {
            "agent_user_id": session.agent_user_id,
            "samples_collected": session.samples_collected,
        }
    return {
        "users": users,
        "override_user_id": override_id,
        "enrollment": enrollment,
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


async def start_voice_enrollment(
    hass: HomeAssistant,
    entry_id: str,
    *,
    agent_user_id: str,
) -> dict[str, Any]:
    """Begin collecting voice samples for a registered member."""
    store = get_identity_store(hass, entry_id)

    def _validate():
        user = store.get_user(agent_user_id)
        if user is None or user.merged_into:
            raise ValueError("User not found")
        if user.kind != UserKind.REGISTERED:
            raise ValueError("Voice enrollment applies to registered members only")
        return user

    try:
        user = await hass.async_add_executor_job(_validate)
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err

    session = set_enrollment_target(hass, entry_id, agent_user_id)
    profile = await hass.async_add_executor_job(
        store.get_voice_profile_for_user,
        agent_user_id,
    )
    return {
        "agent_user_id": agent_user_id,
        "display_name": user.display_name,
        "samples_collected": session.samples_collected if session else 0,
        "voice_profile": (
            voice_profile_to_dict(profile) if profile is not None else None
        ),
    }


async def stop_voice_enrollment(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Cancel an active voice enrollment session."""
    session = get_enrollment_session(hass, entry_id)
    clear_enrollment_target(hass, entry_id)
    return {
        "cancelled": session is not None,
        "agent_user_id": session.agent_user_id if session else None,
        "samples_collected": session.samples_collected if session else 0,
    }


async def get_voice_enrollment(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Return the active enrollment session, if any."""
    session = get_enrollment_session(hass, entry_id)
    if session is None:
        return {"active": False}
    store = get_identity_store(hass, entry_id)
    user = await hass.async_add_executor_job(store.get_user, session.agent_user_id)
    profile = await hass.async_add_executor_job(
        store.get_voice_profile_for_user,
        session.agent_user_id,
    )
    return {
        "active": True,
        "agent_user_id": session.agent_user_id,
        "display_name": user.display_name if user else session.agent_user_id,
        "samples_collected": session.samples_collected,
        "voice_profile": (
            voice_profile_to_dict(profile) if profile is not None else None
        ),
    }


async def reassign_turn_identity(
    hass: HomeAssistant,
    entry_id: str,
    *,
    timestamp: float,
    agent_user_id: str,
    corrected_by_ha_user_id: str | None,
) -> dict[str, Any]:
    """Reassign a past activity turn to another agent user."""
    turn = get_turn(hass, entry_id, timestamp=timestamp)
    if turn is None:
        raise HomeAssistantError("Activity turn not found")

    store = get_identity_store(hass, entry_id)

    def _load_user():
        user = store.get_user(agent_user_id)
        if user is None or user.merged_into:
            raise ValueError("User not found")
        return user

    try:
        user = await hass.async_add_executor_job(_load_user)
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err

    updated = update_turn_identity(
        hass,
        entry_id,
        timestamp,
        agent_user_id=user.id,
        agent_user_display_name=user.display_name,
        agent_user_kind=user.kind.value,
        corrected_by_ha_user_id=corrected_by_ha_user_id,
        original_user_id=str(turn.get("agent_user_id") or ""),
        original_display_name=str(turn.get("agent_user_display_name") or ""),
    )
    if updated is None:
        raise HomeAssistantError("Could not update activity turn")
    return {"turn": updated}


_UNSET = object()

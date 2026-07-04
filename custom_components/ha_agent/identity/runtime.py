"""In-memory console identity overrides and voice enrollment sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass

from homeassistant.core import HomeAssistant, callback

from ..const import DATA_KEY

IDENTITY_OVERRIDE_KEY = "identity_overrides"
IDENTITY_ENROLLMENT_KEY = "identity_enrollment"
GLOBAL_OVERRIDE_KEY = "__global__"


@dataclass(slots=True)
class EnrollmentSession:
    """Active voice enrollment for one registered member."""

    agent_user_id: str
    samples_collected: int = 0
    started_at: float = 0.0


@callback
def _override_store(hass: HomeAssistant) -> dict[str, dict[str, str]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(IDENTITY_OVERRIDE_KEY, {})


@callback
def _enrollment_store(hass: HomeAssistant) -> dict[str, EnrollmentSession]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(IDENTITY_ENROLLMENT_KEY, {})


@callback
def set_identity_override(
    hass: HomeAssistant,
    entry_id: str,
    agent_user_id: str | None,
    *,
    conversation_id: str | None = None,
) -> None:
    """Set or clear an admin identity override for console chat."""
    entry_overrides = _override_store(hass).setdefault(entry_id, {})
    key = conversation_id or GLOBAL_OVERRIDE_KEY
    if agent_user_id:
        entry_overrides[key] = agent_user_id
    else:
        entry_overrides.pop(key, None)


@callback
def get_identity_override(
    hass: HomeAssistant,
    entry_id: str,
    *,
    conversation_id: str | None = None,
) -> str | None:
    """Return override agent user id for a conversation or global console default."""
    entry_overrides = _override_store(hass).get(entry_id, {})
    if conversation_id and conversation_id in entry_overrides:
        return entry_overrides[conversation_id]
    return entry_overrides.get(GLOBAL_OVERRIDE_KEY)


@callback
def clear_identity_override(
    hass: HomeAssistant,
    entry_id: str,
    *,
    conversation_id: str | None = None,
) -> None:
    """Clear conversation or global override."""
    set_identity_override(
        hass,
        entry_id,
        None,
        conversation_id=conversation_id,
    )


@callback
def set_enrollment_target(
    hass: HomeAssistant,
    entry_id: str,
    agent_user_id: str | None,
) -> EnrollmentSession | None:
    """Start or clear voice enrollment for a registered member."""
    store = _enrollment_store(hass)
    if not agent_user_id:
        store.pop(entry_id, None)
        return None
    session = EnrollmentSession(
        agent_user_id=agent_user_id,
        samples_collected=0,
        started_at=time.time(),
    )
    store[entry_id] = session
    return session


@callback
def get_enrollment_session(
    hass: HomeAssistant,
    entry_id: str,
) -> EnrollmentSession | None:
    """Return the active enrollment session, if any."""
    return _enrollment_store(hass).get(entry_id)


@callback
def record_enrollment_sample(
    hass: HomeAssistant,
    entry_id: str,
) -> EnrollmentSession | None:
    """Increment enrollment sample count and return the updated session."""
    session = _enrollment_store(hass).get(entry_id)
    if session is None:
        return None
    session.samples_collected += 1
    return session


@callback
def clear_enrollment_target(hass: HomeAssistant, entry_id: str) -> None:
    """Clear an active enrollment session."""
    _enrollment_store(hass).pop(entry_id, None)

"""In-memory console identity overrides (per conversation or global)."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, callback

from ..const import DATA_KEY

IDENTITY_OVERRIDE_KEY = "identity_overrides"
GLOBAL_OVERRIDE_KEY = "__global__"


@callback
def _override_store(hass: HomeAssistant) -> dict[str, dict[str, str]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(IDENTITY_OVERRIDE_KEY, {})


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

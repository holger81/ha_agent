"""Optional bridge to ha_liquidai voice turn cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import DEFAULT_VOICE_BACKEND
from .models import SpeakerMatch

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def pop_speaker_match_for_text(
    hass: HomeAssistant,
    *,
    user_text: str,
) -> SpeakerMatch | None:
    """Return a speaker match from ha_liquidai when that integration is loaded."""
    if "ha_liquidai_custom" not in getattr(hass.config, "components", set()):
        return None
    try:
        from custom_components.ha_liquidai_custom.voice_cache import (
            pop_matching_voice_turn,
        )
    except ImportError:
        return None

    payload = pop_matching_voice_turn(hass, user_text=user_text)
    if payload is None:
        return None

    return SpeakerMatch(
        backend=DEFAULT_VOICE_BACKEND,
        embedding=payload.embedding,
        quality=payload.quality,
        model=payload.model,
        duration_ms=payload.duration_ms,
    )

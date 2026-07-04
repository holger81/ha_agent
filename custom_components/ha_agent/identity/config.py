"""Voice identity clustering configuration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from ..const import (
    CONF_IDENTITY_AUTO_NAME_ENABLED,
    CONF_IDENTITY_GUEST_CREATE_THRESHOLD,
    CONF_IDENTITY_GUEST_MATCH_THRESHOLD,
    CONF_IDENTITY_GUEST_TIE_MARGIN,
    CONF_IDENTITY_MIN_UTTERANCE_MS,
    CONF_IDENTITY_VOICE_ENABLED,
)

DEFAULT_VOICE_EMBED_ENABLED = True
DEFAULT_GUEST_MATCH_THRESHOLD = 0.75
DEFAULT_GUEST_CREATE_THRESHOLD = 0.52
DEFAULT_GUEST_TIE_MARGIN = 0.05
DEFAULT_MIN_UTTERANCE_MS = 800
DEFAULT_VOICE_BACKEND = "sherpa-onnx"
DEFAULT_AUTO_NAME_ENABLED = True
DEFAULT_ENROLLMENT_SAMPLES_TARGET = 3

SKIP_EMBED_QUALITIES = frozenset({"skipped", "error", "too_short"})


@dataclass(frozen=True, slots=True)
class IdentityVoiceConfig:
    """Thresholds for embedding-based guest clustering."""

    enabled: bool = DEFAULT_VOICE_EMBED_ENABLED
    guest_match_threshold: float = DEFAULT_GUEST_MATCH_THRESHOLD
    guest_create_threshold: float = DEFAULT_GUEST_CREATE_THRESHOLD
    guest_tie_margin: float = DEFAULT_GUEST_TIE_MARGIN
    min_utterance_ms: int = DEFAULT_MIN_UTTERANCE_MS
    auto_name_enabled: bool = DEFAULT_AUTO_NAME_ENABLED
    enrollment_samples_target: int = DEFAULT_ENROLLMENT_SAMPLES_TARGET


IDENTITY_VOICE_CONFIG = IdentityVoiceConfig()


def identity_voice_config_from_entry(entry: ConfigEntry) -> IdentityVoiceConfig:
    """Build voice identity settings from a config entry."""
    data = entry.data
    return IdentityVoiceConfig(
        enabled=bool(
            data.get(CONF_IDENTITY_VOICE_ENABLED, DEFAULT_VOICE_EMBED_ENABLED)
        ),
        guest_match_threshold=float(
            data.get(
                CONF_IDENTITY_GUEST_MATCH_THRESHOLD,
                DEFAULT_GUEST_MATCH_THRESHOLD,
            )
        ),
        guest_create_threshold=float(
            data.get(
                CONF_IDENTITY_GUEST_CREATE_THRESHOLD,
                DEFAULT_GUEST_CREATE_THRESHOLD,
            )
        ),
        guest_tie_margin=float(
            data.get(CONF_IDENTITY_GUEST_TIE_MARGIN, DEFAULT_GUEST_TIE_MARGIN)
        ),
        min_utterance_ms=int(
            data.get(CONF_IDENTITY_MIN_UTTERANCE_MS, DEFAULT_MIN_UTTERANCE_MS)
        ),
        auto_name_enabled=bool(
            data.get(CONF_IDENTITY_AUTO_NAME_ENABLED, DEFAULT_AUTO_NAME_ENABLED)
        ),
    )

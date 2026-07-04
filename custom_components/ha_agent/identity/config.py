"""Voice identity clustering configuration."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_VOICE_EMBED_ENABLED = True
DEFAULT_GUEST_MATCH_THRESHOLD = 0.75
DEFAULT_GUEST_CREATE_THRESHOLD = 0.65
DEFAULT_MIN_UTTERANCE_MS = 800
DEFAULT_VOICE_BACKEND = "sherpa-onnx"

SKIP_EMBED_QUALITIES = frozenset({"skipped", "error", "too_short"})


@dataclass(frozen=True, slots=True)
class IdentityVoiceConfig:
    """Thresholds for embedding-based guest clustering."""

    enabled: bool = DEFAULT_VOICE_EMBED_ENABLED
    guest_match_threshold: float = DEFAULT_GUEST_MATCH_THRESHOLD
    guest_create_threshold: float = DEFAULT_GUEST_CREATE_THRESHOLD
    min_utterance_ms: int = DEFAULT_MIN_UTTERANCE_MS


IDENTITY_VOICE_CONFIG = IdentityVoiceConfig()

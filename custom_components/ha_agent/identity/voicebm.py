"""VoiceBM / voice identity hooks (Phase 9b)."""

from __future__ import annotations

import json
import re
from typing import Any

_IDENTITY_JSON = re.compile(
    r"HA_AGENT_IDENTITY\s*:\s*(\{.*?\})",
    re.IGNORECASE | re.DOTALL,
)


def parse_voice_identity(extra_system_prompt: str | None) -> dict[str, Any] | None:
    """Parse an optional identity block from extra_system_prompt.

    Expected shape (VoiceBM or automation bridge)::

        HA_AGENT_IDENTITY: {"speaker_id": "...", "display_name": "...",
        "confidence": 0.92}
    """
    if not extra_system_prompt or not extra_system_prompt.strip():
        return None
    match = _IDENTITY_JSON.search(extra_system_prompt)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

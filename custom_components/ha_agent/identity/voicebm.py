"""VoiceBM / voice identity hooks (Phase 9b)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

_IDENTITY_JSON = re.compile(
    r"HA_AGENT_IDENTITY\s*:\s*(\{.*?\})",
    re.IGNORECASE | re.DOTALL,
)

# Confidence used when a configured bridge secret is missing a valid HMAC.
_UNSIGNED_CONFIDENCE = 0.0


def _canonical_payload(data: dict[str, Any]) -> str:
    """Stable JSON for HMAC (exclude signature field)."""
    body = {key: value for key, value in data.items() if key != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def verify_identity_signature(
    data: dict[str, Any],
    *,
    secret: str,
) -> bool:
    """Return True when ``signature`` matches HMAC-SHA256 of the payload."""
    provided = str(data.get("signature") or "").strip()
    if not provided or not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        _canonical_payload(data).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(provided.lower(), expected.lower())


def parse_voice_identity(
    extra_system_prompt: str | None,
    *,
    bridge_secret: str | None = None,
) -> dict[str, Any] | None:
    """Parse an optional identity block from extra_system_prompt.

    Expected shape (VoiceBM or automation bridge)::

        HA_AGENT_IDENTITY: {"speaker_id": "...", "display_name": "...",
        "confidence": 0.92, "signature": "<hmac>"}

    When ``bridge_secret`` is set, a valid ``signature`` is required for the
    declared confidence to stand. Unsigned or invalid payloads are still
    returned but with confidence forced to ``0.0`` (no user-memory unlock).
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
    if not isinstance(data, dict):
        return None

    secret = (bridge_secret or "").strip()
    if secret:
        if verify_identity_signature(data, secret=secret):
            data["_signature_ok"] = True
        else:
            data["confidence"] = _UNSIGNED_CONFIDENCE
            data["_signature_ok"] = False
    return data

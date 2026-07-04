"""Guest auto-naming from STT self-introduction phrases."""

from __future__ import annotations

import re

_GUEST_NAME_RE = re.compile(r"^Guest \d+$", re.IGNORECASE)
_INTRO_PATTERNS = (
    re.compile(r"(?i)\b(?:i'?m|i am)\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i)\bmy name is\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i)\bthis is\s+([A-Z][a-z]{1,20})\b"),
    re.compile(r"(?i)\b(?:call me|it'?s)\s+([A-Z][a-z]{1,20})\b"),
)
_NAME_BLOCKLIST = frozenset(
    {
        "sorry",
        "fine",
        "good",
        "back",
        "home",
        "here",
        "ready",
        "done",
        "sure",
        "okay",
        "ok",
        "not",
        "just",
        "still",
        "trying",
    }
)


def is_default_guest_name(display_name: str) -> bool:
    """Return True when the name is the auto-assigned Guest N label."""
    return bool(_GUEST_NAME_RE.match(display_name.strip()))


def extract_self_intro_name(user_text: str) -> str | None:
    """Extract a first name from common self-introduction phrases."""
    text = user_text.strip()
    if not text:
        return None
    for pattern in _INTRO_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = match.group(1).strip()
        if not name or name.lower() in _NAME_BLOCKLIST:
            continue
        return name[0].upper() + name[1:]
    return None

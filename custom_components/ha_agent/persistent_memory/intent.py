"""Remember / prefer / forget intent detection for durable memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class MemoryIntentKind(StrEnum):
    """Detected memory-related user intent."""

    REMEMBER = "remember"
    PREFER = "prefer"
    FORGET = "forget"
    CLARIFY = "clarify"
    NONE = "none"


# Workflow phrasing that should stay on the skill path.
_SKILL_SAVE = re.compile(
    r"\b(?:save|remember|store)\b.*\b(?:as a skill|this as a skill|how to do this)\b",
    re.IGNORECASE,
)

_FORGET = re.compile(
    r"\b(?:forget|stop remembering|clear memory of|don't remember|do not remember)\b",
    re.IGNORECASE,
)

_REMEMBER = re.compile(
    r"\b(?:remember(?:\s+that)?|from now on|always(?:\s+use)?|"
    r"i prefer|my preference|prefer that|"
    r"save this as (?:a )?(?:default|preference|fact)|"
    r"make that (?:my )?default)\b",
    re.IGNORECASE,
)

_WORKFLOW_HINT = re.compile(
    r"\b(?:workflow|procedure|steps?|then (?:search|read|call|open)|"
    r"multi[- ]?step|how to (?:check|read|fetch))\b",
    re.IGNORECASE,
)

_CLARIFY_COPY = (
    "Should I save this as a reusable skill (workflow), "
    "or remember it as a preference/default?"
)


@dataclass(frozen=True, slots=True)
class MemoryIntent:
    """Parsed memory intent for one user utterance."""

    kind: MemoryIntentKind
    is_workflow: bool = False
    clarify_message: str = ""
    fragment: str = ""


def detect_memory_intent(user_text: str) -> MemoryIntent:
    """Classify remember/prefer/forget vs skill-save vs none (rules first)."""
    text = (user_text or "").strip()
    if not text:
        return MemoryIntent(kind=MemoryIntentKind.NONE)

    if _SKILL_SAVE.search(text):
        return MemoryIntent(
            kind=MemoryIntentKind.NONE,
            is_workflow=True,
            fragment=text,
        )

    if _FORGET.search(text):
        return MemoryIntent(
            kind=MemoryIntentKind.FORGET,
            fragment=_strip_lead_in(text, _FORGET),
        )

    if _REMEMBER.search(text):
        if _WORKFLOW_HINT.search(text):
            return MemoryIntent(
                kind=MemoryIntentKind.CLARIFY,
                is_workflow=True,
                clarify_message=_CLARIFY_COPY,
                fragment=text,
            )
        kind = (
            MemoryIntentKind.PREFER
            if re.search(r"\b(?:prefer|preference|default)\b", text, re.I)
            else MemoryIntentKind.REMEMBER
        )
        return MemoryIntent(
            kind=kind,
            fragment=_strip_lead_in(text, _REMEMBER),
        )

    return MemoryIntent(kind=MemoryIntentKind.NONE)


def is_preference_shaped_turn(user_text: str) -> bool:
    """Return True when the utterance looks like a preference, not a workflow."""
    intent = detect_memory_intent(user_text)
    return intent.kind in {
        MemoryIntentKind.REMEMBER,
        MemoryIntentKind.PREFER,
        MemoryIntentKind.FORGET,
    }


def _strip_lead_in(text: str, pattern: re.Pattern[str]) -> str:
    cleaned = pattern.sub("", text, count=1).strip()
    cleaned = re.sub(r"^[\s:,\-]+", "", cleaned)
    return cleaned or text.strip()

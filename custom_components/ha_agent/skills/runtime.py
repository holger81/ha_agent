"""Runtime state for skill learning and evaluation."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, callback

from ..const import DATA_KEY
from .models import PendingSkillDraft, TurnTrace

PENDING_DRAFTS_KEY = "skill_pending_drafts"
EVAL_PENDING_KEY = "skill_eval_pending"


@callback
def _pending_store(hass: HomeAssistant) -> dict[str, PendingSkillDraft]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(PENDING_DRAFTS_KEY, {})


@callback
def set_pending_draft(hass: HomeAssistant, draft: PendingSkillDraft) -> None:
    """Store a skill draft awaiting user confirmation."""
    _pending_store(hass)[draft.conversation_id] = draft


@callback
def pop_pending_draft(
    hass: HomeAssistant,
    conversation_id: str | None,
) -> PendingSkillDraft | None:
    """Remove and return a pending draft for a conversation."""
    if not conversation_id:
        return None
    return _pending_store(hass).pop(conversation_id, None)


@callback
def get_pending_draft(
    hass: HomeAssistant,
    conversation_id: str | None,
) -> PendingSkillDraft | None:
    """Return a pending draft without removing it."""
    if not conversation_id:
        return None
    return _pending_store(hass).get(conversation_id)


@callback
def _eval_pending_store(hass: HomeAssistant) -> dict[str, dict]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(EVAL_PENDING_KEY, {})


@callback
def set_eval_pending(
    hass: HomeAssistant,
    conversation_id: str,
    payload: dict,
) -> None:
    """Store deferred evaluation data for the next user turn."""
    _eval_pending_store(hass)[conversation_id] = payload


@callback
def pop_eval_pending(hass: HomeAssistant, conversation_id: str | None) -> dict | None:
    """Pop deferred evaluation payload for a conversation."""
    if not conversation_id:
        return None
    return _eval_pending_store(hass).pop(conversation_id, None)


def should_offer_skill_creation(
    trace: TurnTrace,
    *,
    learning_enabled: bool,
) -> bool:
    """Return True when a successful multi-step turn qualifies for learning."""
    if not learning_enabled:
        return False
    if trace.fallback or trace.tool_errors > 0:
        return False
    if not trace.tool_calls:
        return False
    if trace.matched_skill_ids:
        return False
    multi_step = len(trace.tool_calls) >= 2 or (
        trace.history_len >= 2 and len(trace.tool_calls) >= 1
    )
    return multi_step

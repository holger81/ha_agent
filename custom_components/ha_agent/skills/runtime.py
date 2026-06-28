"""Runtime state for skill learning and evaluation."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, callback

from ..const import DATA_KEY
from .models import PendingSkillDraft, TurnTrace
from .observer import is_discovery_tool

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


def _is_content_extraction_turn(trace: TurnTrace) -> bool:
    """Return True when an email/news turn is mostly content, not a workflow."""
    route = (trace.route or "").lower()
    if route not in {"news", "email"}:
        return False
    non_discovery = [
        call
        for call in trace.tool_calls
        if not is_discovery_tool(str(call.get("toolName") or call.get("name") or ""))
    ]
    if len(non_discovery) >= 2:
        return False
    assistant = (trace.assistant_text or "").strip()
    if len(assistant) > 800:
        return True
    return len(non_discovery) <= 1 and trace.iterations <= 1


def should_offer_skill_creation(
    trace: TurnTrace,
    *,
    learning_enabled: bool,
    manual_save: bool = False,
) -> bool:
    """Return True when a turn passes local heuristics for skill learning."""
    if manual_save:
        return bool(trace.tool_calls) and not trace.fallback and trace.tool_errors == 0

    if not learning_enabled:
        return False
    if trace.skill_plan_override:
        return False
    if trace.fallback:
        return False
    if not trace.tool_calls:
        return False
    if trace.matched_learned_skill_ids:
        return False
    if not trace.assistant_text.strip():
        return False

    route = (trace.route or "").lower()
    if route in {"news", "email"} and _is_content_extraction_turn(trace):
        return False

    if trace.tool_errors > 0:
        non_discovery = [
            c
            for c in trace.tool_calls
            if not is_discovery_tool(str(c.get("toolName") or c.get("name") or ""))
        ]
        recovered = (
            trace.tool_errors > 0
            and bool(trace.assistant_text.strip())
            and len(non_discovery) >= 2
        )
        if not recovered:
            return False

    multi_step = len(trace.tool_calls) >= 2 or trace.iterations >= 2
    return multi_step


def override_turn_eligible_for_learning(trace: TurnTrace) -> bool:
    """Return True when a skill-override turn succeeded with a reusable workflow."""
    if not trace.skill_plan_override:
        return False
    if trace.fallback or not trace.assistant_text.strip():
        return False
    if trace.outcome not in {"success", "partial"}:
        return False
    successful = [
        call
        for call in trace.tool_calls
        if call.get("succeeded")
        and not is_discovery_tool(str(call.get("toolName") or call.get("name") or ""))
    ]
    return bool(successful)

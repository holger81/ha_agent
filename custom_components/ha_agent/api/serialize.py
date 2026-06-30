"""JSON serialization for API responses."""

from __future__ import annotations

from typing import Any

from ..playbooks import Playbook
from ..recovery_hints import RecoveryHint
from ..route_keywords import RouteKeywords
from ..skills.models import PendingSkillDraft, Skill, SkillIndexRow, TurnTrace


def playbook_to_dict(playbook: Playbook) -> dict[str, Any]:
    """Serialize an editable route playbook."""
    return {
        "route": playbook.route,
        "title": playbook.title,
        "body": playbook.body,
        "match_text": playbook.match_text,
        "enabled": playbook.enabled,
        "updated_at": playbook.updated_at,
        "is_default": playbook.is_default,
        "is_builtin": playbook.is_builtin,
    }


def route_keywords_to_dict(item: RouteKeywords) -> dict[str, Any]:
    """Serialize an editable route keyword list."""
    return {
        "route": item.route,
        "title": item.title,
        "keywords": list(item.keywords),
        "enabled": item.enabled,
        "updated_at": item.updated_at,
        "is_default": item.is_default,
    }


def recovery_hint_to_dict(hint: RecoveryHint) -> dict[str, Any]:
    """Serialize an editable recovery-hint rule."""
    return {
        "rule_id": hint.rule_id,
        "title": hint.title,
        "tool_substring": hint.tool_substring,
        "error_pattern": hint.error_pattern,
        "body": hint.body,
        "enabled": hint.enabled,
        "is_builtin": hint.is_builtin,
        "priority": hint.priority,
        "updated_at": hint.updated_at,
        "is_default": hint.is_default,
    }


def skill_to_dict(skill: Skill) -> dict[str, Any]:
    """Serialize a full skill record."""
    return {
        "id": skill.id,
        "slug": skill.slug,
        "title": skill.title,
        "description": skill.description,
        "triggers": list(skill.triggers),
        "body": skill.body,
        "tool_steps": list(skill.tool_steps),
        "enabled": skill.enabled,
        "created_at": skill.created_at,
        "last_used_at": skill.last_used_at,
        "use_count": skill.use_count,
        "success_count": skill.success_count,
        "last_improved_at": skill.last_improved_at,
        "last_evaluation_at": skill.last_evaluation_at,
        "version": skill.version,
        "slots": [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "default": s.default,
            }
            for s in skill.slots
        ],
        "preconditions": skill.preconditions,
        "parent_id": skill.parent_id,
        "route_scope": skill.route_scope,
        "score": skill.score,
        "is_builtin": skill.is_builtin,
    }


def skill_index_to_dict(row: SkillIndexRow) -> dict[str, Any]:
    """Serialize an FTS search hit."""
    return {
        "id": row.id,
        "slug": row.slug,
        "title": row.title,
        "description": row.description,
        "rank": row.rank,
    }


def turn_trace_to_dict(
    trace: TurnTrace, *, timestamp: float | None = None
) -> dict[str, Any]:
    """Serialize a turn trace for the activity log."""
    return {
        "timestamp": timestamp,
        "user_text": trace.user_text,
        "assistant_text": trace.assistant_text,
        "conversation_id": trace.conversation_id,
        "history_len": trace.history_len,
        "iterations": trace.iterations,
        "tool_calls": list(trace.tool_calls),
        "tool_errors": trace.tool_errors,
        "fallback": trace.fallback,
        "matched_skill_ids": list(trace.matched_skill_ids),
        "controlled_entity_ids": list(trace.controlled_entity_ids),
        "outcome": trace.outcome,
        "verification_notes": list(trace.verification_notes),
        "route": trace.route,
        "exposed_entities": list(trace.exposed_entities),
        "complexity": trace.complexity,
        "llm_calls": list(trace.llm_calls),
        "verifier_verdict": trace.verifier_verdict,
        "verifier_detail": trace.verifier_detail,
    }


def pending_draft_to_dict(draft: PendingSkillDraft) -> dict[str, Any]:
    """Serialize a pending skill draft awaiting confirmation."""
    payload: dict[str, Any] = {
        "entry_id": draft.entry_id,
        "conversation_id": draft.conversation_id,
        "trace": turn_trace_to_dict(draft.trace),
        "history": list(draft.history),
        "observer_reason": draft.observer_reason,
    }
    if draft.skill_draft is not None:
        payload["skill_draft"] = {
            "title": draft.skill_draft.title,
            "description": draft.skill_draft.description,
            "triggers": list(draft.skill_draft.triggers),
            "body": draft.skill_draft.body,
            "tool_steps": list(draft.skill_draft.tool_steps),
        }
    return payload

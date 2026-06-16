"""JSON serialization for API responses."""

from __future__ import annotations

from typing import Any

from ..skills.models import PendingSkillDraft, Skill, SkillIndexRow, TurnTrace


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
    }


def pending_draft_to_dict(draft: PendingSkillDraft) -> dict[str, Any]:
    """Serialize a pending skill draft awaiting confirmation."""
    return {
        "entry_id": draft.entry_id,
        "conversation_id": draft.conversation_id,
        "trace": turn_trace_to_dict(draft.trace),
        "history": list(draft.history),
    }

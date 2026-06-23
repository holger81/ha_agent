"""Serialize eval benchmark cases."""

from __future__ import annotations

from typing import Any

from .models import EvalCase


def eval_case_to_dict(case: EvalCase) -> dict[str, Any]:
    """Serialize an eval case for API responses."""
    return {
        "id": case.id,
        "task": case.task,
        "user_text": case.user_text,
        "exposed_entities": list(case.exposed_entities),
        "expected_tool": case.expected_tool,
        "expected_tool_args": case.expected_tool_args,
        "expected_text_contains": list(case.expected_text_contains),
        "expected_playbook_route": case.expected_playbook_route,
        "mock_mcp_responses": list(case.mock_mcp_responses),
        "max_iterations": case.max_iterations,
        "source": case.source,
        "promoted_at": case.promoted_at,
        "source_timestamp": case.source_timestamp,
        "source_conversation_id": case.source_conversation_id,
    }


def eval_case_from_dict(data: dict[str, Any]) -> EvalCase:
    """Deserialize a stored eval case."""
    return EvalCase(
        id=str(data["id"]),
        task=str(data["task"]),
        user_text=str(data.get("user_text") or ""),
        exposed_entities=list(data.get("exposed_entities") or []),
        expected_tool=data.get("expected_tool"),
        expected_tool_args=data.get("expected_tool_args"),
        expected_text_contains=list(data.get("expected_text_contains") or []),
        expected_playbook_route=data.get("expected_playbook_route"),
        mock_mcp_responses=list(data.get("mock_mcp_responses") or []),
        max_iterations=int(data.get("max_iterations") or 6),
        source=str(data.get("source") or "promoted"),
        promoted_at=data.get("promoted_at"),
        source_timestamp=data.get("source_timestamp"),
        source_conversation_id=data.get("source_conversation_id"),
    )

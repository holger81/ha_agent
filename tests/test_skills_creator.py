"""Tests for skill distillation helpers."""

from __future__ import annotations

from custom_components.ha_agent.skills.creator import _fallback_skill_draft
from custom_components.ha_agent.skills.models import TurnTrace


def test_fallback_skill_draft_uses_upstream_tool_names() -> None:
    trace = TurnTrace(
        user_text="check my new emails",
        history_len=2,
        assistant_text="You have 3 unread messages.",
        tool_calls=[
            {
                "toolName": "mail_mcp_imap_mailbox_status",
                "name": "mail_mcp_imap_mailbox_status",
                "arguments": {"account_id": "default", "mailbox": "INBOX"},
            },
            {
                "toolName": "mail_mcp_imap_search_messages",
                "name": "mail_mcp_imap_search_messages",
                "arguments": {"unread_only": True},
            },
        ],
    )

    draft = _fallback_skill_draft(trace)

    assert draft is not None
    assert draft.title == "check my new emails"
    assert len(draft.tool_steps) == 2
    assert draft.tool_steps[0]["toolName"] == "mail_mcp_imap_mailbox_status"
    assert "mail_mcp_imap_search_messages" in draft.body


def test_fallback_skill_draft_requires_tool_calls() -> None:
    trace = TurnTrace(user_text="hello", history_len=0)

    assert _fallback_skill_draft(trace) is None

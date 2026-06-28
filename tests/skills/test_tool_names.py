"""Tests for MCP tool name canonicalization in skills."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_modules():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in ("defaults", "models", "tool_names", "body"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    return (
        sys.modules["ha_agent.skills.models"],
        sys.modules["ha_agent.skills.tool_names"],
        sys.modules["ha_agent.skills.body"],
    )


models_mod, tool_names_mod, body_mod = _load_modules()
Skill = models_mod.Skill
SkillDraft = models_mod.SkillDraft
IMAP_GET_MESSAGE = tool_names_mod.IMAP_GET_MESSAGE
IMAP_MAILBOX_STATUS = tool_names_mod.IMAP_MAILBOX_STATUS
IMAP_SEARCH_MESSAGES = tool_names_mod.IMAP_SEARCH_MESSAGES
canonicalize_tool_name = tool_names_mod.canonicalize_tool_name
canonicalize_tool_steps = tool_names_mod.canonicalize_tool_steps
rewrite_tool_names_in_text = tool_names_mod.rewrite_tool_names_in_text
derive_tool_steps_from_body = body_mod.derive_tool_steps_from_body
normalize_skill = body_mod.normalize_skill
normalize_skill_draft = body_mod.normalize_skill_draft


def test_canonicalize_short_email_tool_names() -> None:
    assert canonicalize_tool_name("mailbox_status") == IMAP_MAILBOX_STATUS
    assert canonicalize_tool_name("search_messages") == IMAP_SEARCH_MESSAGES
    assert canonicalize_tool_name("get_message") == IMAP_GET_MESSAGE


def test_canonicalize_wrong_delimiter_email_tool_names() -> None:
    assert (
        canonicalize_tool_name("mail_mcp__mailbox_status") == IMAP_MAILBOX_STATUS
    )
    assert (
        canonicalize_tool_name("mail_mcp_imap_search_messages")
        == IMAP_SEARCH_MESSAGES
    )


def test_rewrite_tool_names_in_markdown_body() -> None:
    body = (
        "# Check inbox\n\n"
        "1. Call `mail_mcp__mailbox_status`.\n"
        "2. Then search_messages with unread_only=true.\n"
    )
    updated, changed = rewrite_tool_names_in_text(body)
    assert changed is True
    assert IMAP_MAILBOX_STATUS in updated
    assert IMAP_SEARCH_MESSAGES in updated
    assert "`mailbox_status`" not in updated
    assert " Then mail_mcp__imap_search_messages" in updated or IMAP_SEARCH_MESSAGES in updated


def test_normalize_skill_draft_fixes_tool_steps() -> None:
    draft = SkillDraft(
        title="Email",
        description="Check inbox",
        triggers=["email"],
        body="Run `mailbox_status` then `search_messages`.",
        tool_steps=[
            {"toolName": "mail_mcp__mailbox_status"},
            {"toolName": "mail_mcp_imap_search_messages"},
        ],
    )
    normalized = normalize_skill_draft(draft)
    names = [step["toolName"] for step in normalized.tool_steps]
    assert names == [IMAP_MAILBOX_STATUS, IMAP_SEARCH_MESSAGES]


def test_normalize_skill_fixes_existing_skill() -> None:
    skill = Skill(
        id="s1",
        slug="check-email",
        title="Check email",
        description="Unread mail",
        triggers=["email"],
        body="Use mailbox_status and mail_mcp__get_message.",
        tool_steps=[{"toolName": "mailbox_status", "arguments": {}}],
    )
    normalize_skill(skill)
    assert IMAP_MAILBOX_STATUS in skill.body
    assert skill.tool_steps[0]["toolName"] == IMAP_MAILBOX_STATUS


def test_derive_tool_steps_after_rewrite() -> None:
    body = "Call `mail_mcp__mailbox_status` with mailbox INBOX."
    updated, _ = rewrite_tool_names_in_text(body)
    steps = derive_tool_steps_from_body(updated)
    assert steps == [{"toolName": IMAP_MAILBOX_STATUS, "arguments": {}}]


def test_canonicalize_imap_get_message_is_not_renamed_to_fetch() -> None:
    assert (
        canonicalize_tool_name("mail_mcp__imap_get_message") == IMAP_GET_MESSAGE
    )


def test_ensure_imap_tool_step_arguments() -> None:
    skill = Skill(
        id="s1",
        slug="email",
        title="Email",
        description="d",
        triggers=["email"],
        body="workflow",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status", "arguments": {}},
            {"toolName": "mail_mcp__imap_search_messages", "arguments": {}},
        ],
    )
    normalize_skill(skill)
    assert skill.tool_steps[0]["arguments"]["mailbox"] == "{{mailbox}}"
    search_args = skill.tool_steps[1]["arguments"]
    assert search_args["mailbox"] == "{{mailbox}}"
    assert search_args["unread_only"] is True
    assert search_args["limit"] == 10


def test_canonicalize_tool_steps_preserves_arguments() -> None:
    steps, changed = canonicalize_tool_steps(
        [
            {
                "toolName": "search_messages",
                "arguments": {"mailbox": "INBOX", "unread_only": True},
            }
        ]
    )
    assert changed is True
    assert steps[0]["toolName"] == IMAP_SEARCH_MESSAGES
    assert steps[0]["arguments"]["mailbox"] == "INBOX"

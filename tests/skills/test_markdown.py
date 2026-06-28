"""Tests for markdown skill files."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_markdown():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in ("defaults", "models", "body"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")

        class HomeAssistantError(Exception):
            pass

        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc

    path = COMPONENT / "skills" / "markdown.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.markdown", path)
    markdown = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.markdown"] = markdown
    spec.loader.exec_module(markdown)
    return markdown, sys.modules["ha_agent.skills.models"]


markdown_mod, models_mod = _load_markdown()
Skill = models_mod.Skill
draft_from_markdown = markdown_mod.draft_from_markdown
skill_to_markdown = markdown_mod.skill_to_markdown

SAMPLE = """---
title: Email Management
description: Check inbox for new mail
triggers:
  - new emails
  - check inbox
route_scope: email
enabled: true
slots:
  - name: mailbox
    default: INBOX
---

# Check email

1. Run `mail_mcp__imap_mailbox_status` with mailbox `{{mailbox}}`.
"""


def test_parse_skill_markdown() -> None:
    draft, slug, explicit = draft_from_markdown(
        SAMPLE,
        filename_slug="email-management",
    )
    assert slug == "email-management"
    assert explicit is False
    assert draft.title == "Email Management"
    assert "new emails" in draft.triggers
    assert draft.route_scope == "email"
    assert any(slot.name == "mailbox" for slot in draft.slots)
    assert draft.tool_steps == [
        {"toolName": "mail_mcp__imap_mailbox_status", "arguments": {}},
    ]


def test_round_trip_skill_markdown() -> None:
    skill = Skill(
        id="1",
        slug="email-management",
        title="Email Management",
        description="Check inbox",
        triggers=["new emails"],
        body="# Check email\n\nRun `mail_mcp__imap_mailbox_status`.",
        tool_steps=[{"toolName": "mail_mcp__imap_mailbox_status", "arguments": {}}],
        route_scope="email",
        slots=[models_mod.SkillSlot(name="mailbox", default="INBOX")],
    )
    text = skill_to_markdown(skill)
    draft, slug, _explicit = draft_from_markdown(text, filename_slug=skill.slug)
    assert slug == "email-management"
    assert draft.title == skill.title
    assert draft.triggers == skill.triggers
    assert "mail_mcp__imap_mailbox_status" in draft.body

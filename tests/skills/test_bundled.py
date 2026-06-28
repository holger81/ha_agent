"""Tests for bundled email skill helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_bundled():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")

        class HomeAssistantError(Exception):
            pass

        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc

    for name in ("defaults", "models", "tool_names", "body", "markdown", "bundled"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    return sys.modules["ha_agent.skills.bundled"], sys.modules["ha_agent.skills.models"]


bundled_mod, models_mod = _load_bundled()
Skill = models_mod.Skill
email_skill_needs_refresh = bundled_mod.email_skill_needs_refresh
apply_bundled_skill = bundled_mod.apply_bundled_skill
load_bundled_skill_text = bundled_mod.load_bundled_skill_text


def test_bundled_email_markdown_loads() -> None:
    text = load_bundled_skill_text("check-and-read-unread-emails")
    assert text is not None
    assert "mail_mcp__imap_mailbox_status" in text
    assert "mail_mcp__imap_get_message" in text
    assert "imap_fetch_message" not in text


def test_stale_email_skill_detected() -> None:
    skill = Skill(
        id="1",
        slug="check-and-read-unread-emails",
        title="Email Management",
        description="Check inbox",
        triggers=["email"],
        body="Call mailbox_status then search_messages.",
        tool_steps=[{"toolName": "mail_mcp__imap_fetch_message", "arguments": {}}],
        route_scope="email",
    )
    assert email_skill_needs_refresh(skill) is True


def test_apply_bundled_skill_rewrites_broken_email_skill() -> None:
    skill = Skill(
        id="1",
        slug="check-and-read-unread-emails",
        title="Email Management",
        description="Check inbox",
        triggers=["email"],
        body="Call mailbox_status.",
        tool_steps=[{"toolName": "mailbox_status", "arguments": {}}],
        route_scope="email",
    )
    assert apply_bundled_skill(skill) is True
    assert "mail_mcp__imap_mailbox_status" in skill.body
    assert any(
        step.get("toolName") == "mail_mcp__imap_search_messages"
        for step in skill.tool_steps
    )
    assert email_skill_needs_refresh(skill) is False

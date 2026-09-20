"""Tests for bundled email skill helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


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
news_skill_needs_refresh = bundled_mod.news_skill_needs_refresh
apply_bundled_skill = bundled_mod.apply_bundled_skill
load_bundled_skill_text = bundled_mod.load_bundled_skill_text


def test_bundled_email_markdown_loads() -> None:
    text = load_bundled_skill_text("check-and-read-unread-emails")
    assert text is not None
    assert "mail_mcp__imap_mailbox_status" in text
    assert "mail_mcp__imap_get_message" in text
    assert "imap_fetch_message" not in text


def test_bundled_news_markdown_uses_digest_scope() -> None:
    text = load_bundled_skill_text("news-briefing")
    assert text is not None
    assert 'digest_scope: "{{digest_scope}}"' in text
    assert 'query: "{{digest_scope}}"' not in text
    assert "mcp_news__news_curate" in text


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


def test_stale_news_skill_detected() -> None:
    skill = Skill(
        id="2",
        slug="news-briefing",
        title="News briefing",
        description="Headlines",
        triggers=["news"],
        body="Call mcp_news__news_curate with query.",
        tool_steps=[{"toolName": "mcp_news__news_curate", "arguments": {}}],
        route_scope="news",
    )
    assert news_skill_needs_refresh(skill) is True


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


def test_apply_bundled_skill_rewrites_broken_news_skill() -> None:
    skill = Skill(
        id="2",
        slug="news-briefing",
        title="News briefing",
        description="Headlines",
        triggers=["news"],
        body="Pass query from digest_scope.",
        tool_steps=[
            {
                "toolName": "mcp_news__news_curate",
                "arguments": {"query": "{{digest_scope}}"},
            }
        ],
        route_scope="news",
    )
    assert apply_bundled_skill(skill) is True
    assert news_skill_needs_refresh(skill) is False
    step = skill.tool_steps[0]
    assert step["toolName"] == "mcp_news__news_curate"
    assert "digest_scope" in step["arguments"]
    assert "query" not in step["arguments"]


def test_stale_status_skill_with_call_service_detected() -> None:
    status_skill_needs_refresh = bundled_mod.status_skill_needs_refresh
    skill = Skill(
        id="3",
        slug="look-up-sensor-or-entity-status-2",
        title="Look up sensor or entity status",
        description="Status",
        triggers=["turn on the dining room lights", "status of {{query}}"],
        body="Call home_assistant__ha_call_service for readings.",
        tool_steps=[{"toolName": "home_assistant__ha_call_service", "arguments": {}}],
        route_scope="chat",
    )
    assert status_skill_needs_refresh(skill) is True
    assert apply_bundled_skill(skill) is True
    assert status_skill_needs_refresh(skill) is False
    assert any(
        "ha_search" in str(step.get("toolName") or "") for step in skill.tool_steps
    )
    assert not any(
        "ha_call_service" in str(step.get("toolName") or "")
        for step in skill.tool_steps
    )
    assert not any("turn on" in trigger.lower() for trigger in skill.triggers)

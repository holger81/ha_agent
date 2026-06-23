"""Unit tests for markdown-first skill body helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_body_module():
    path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", path)
    models = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "skills" / "body.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.body", path)
    body_mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(body_mod)
    return models, body_mod


models_mod, body_mod = _load_body_module()
Skill = models_mod.Skill
SkillDraft = models_mod.SkillDraft
derive_tool_steps_from_body = body_mod.derive_tool_steps_from_body
extract_tool_steps_block = body_mod.extract_tool_steps_block
normalize_skill = body_mod.normalize_skill
normalize_skill_draft = body_mod.normalize_skill_draft
resolve_tool_steps = body_mod.resolve_tool_steps


def test_derive_tool_steps_from_backticks() -> None:
    """Backtick tool names become ordered tool steps."""
    body = (
        "1. Curate headlines with `mcp_news__news_curate`.\n"
        "2. Summarize for the user."
    )
    steps = derive_tool_steps_from_body(body)
    assert steps == [{"toolName": "mcp_news__news_curate", "arguments": {}}]


def test_derive_tool_steps_from_bare_names() -> None:
    """Bare double-underscore tool names are detected."""
    body = "Call mail_mcp__imap_search_messages then reply."
    steps = derive_tool_steps_from_body(body)
    assert steps == [
        {"toolName": "mail_mcp__imap_search_messages", "arguments": {}},
    ]


def test_extract_tool_steps_block_prefers_fence() -> None:
    """A ```tool_steps fence overrides backtick scanning."""
    body = (
        "Use `ignored__tool`.\n"
        "```tool_steps\n"
        '[{"toolName": "mail_mcp__imap_fetch_message", "arguments": {"uid": 1}}]\n'
        "```"
    )
    steps = extract_tool_steps_block(body)
    assert steps == [
        {
            "toolName": "mail_mcp__imap_fetch_message",
            "arguments": {"uid": 1},
        }
    ]
    assert derive_tool_steps_from_body(body) == steps


def test_normalize_skill_draft_derives_when_not_explicit() -> None:
    """Draft normalization derives tool steps from workflow text."""
    draft = SkillDraft(
        title="News",
        description="Briefing",
        triggers=["news"],
        body="Run `mcp_news__news_curate`.",
        tool_steps=[],
    )
    normalized = normalize_skill_draft(draft)
    assert normalized.tool_steps == [
        {"toolName": "mcp_news__news_curate", "arguments": {}},
    ]


def test_normalize_skill_respects_explicit_override() -> None:
    """Explicit override keeps manual tool steps even when body differs."""
    skill = Skill(
        id="1",
        slug="email",
        title="Email",
        description="Check mail",
        triggers=["inbox"],
        body="Run `other__tool`.",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages", "arguments": {}}],
    )
    normalize_skill(skill, explicit_tool_steps=True)
    assert skill.tool_steps == [
        {"toolName": "mail_mcp__imap_search_messages", "arguments": {}},
    ]


def test_resolve_tool_steps_prefers_derived_over_stale() -> None:
    """Without override, body-derived steps replace stale stored steps."""
    steps = resolve_tool_steps(
        "Use `fresh__tool`.",
        [{"toolName": "stale__tool", "arguments": {}}],
        explicit_override=False,
    )
    assert steps == [{"toolName": "fresh__tool", "arguments": {}}]

"""Unit tests for skill context formatting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_format_modules():
    path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", path)
    models = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "skills" / "format.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.format", path)
    fmt = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(fmt)

    path = COMPONENT / "context.py"
    spec = importlib.util.spec_from_file_location("ha_agent.context", path)
    context = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.context"] = context
    spec.loader.exec_module(context)
    return models, fmt, context


models_mod, format_mod, context_mod = _load_format_modules()
Skill = models_mod.Skill
format_skills_for_context = format_mod.format_skills_for_context
build_tool_context = context_mod.build_tool_context


def test_format_skills_for_context() -> None:
    """Matched skills render an ACTIVE SKILLS block."""
    skill = Skill(
        id="1",
        slug="dining-lights",
        title="Dining lights",
        description="Turn dining room ceiling on or off.",
        triggers=["dining room lights"],
        body="Use light.dining_room_ceiling",
        tool_steps=[{"toolName": "home_assistant__ha_call_service"}],
    )
    block = format_skills_for_context([skill])
    assert "ACTIVE SKILLS" in block
    assert "dining-lights" in block
    assert "ha_call_service" in block


def test_build_tool_context_includes_skill_hints() -> None:
    """Skill hints are injected into tool context."""
    hints = format_skills_for_context(
        [
            Skill(
                id="1",
                slug="news",
                title="News briefing",
                description="Fetch headlines.",
                triggers=["news"],
                body="Call news_curate",
                tool_steps=[],
            )
        ]
    )
    context = build_tool_context("what is the news", [], skill_hints=hints)
    assert "ACTIVE SKILLS" in context
    assert "News briefing" in context

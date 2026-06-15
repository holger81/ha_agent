"""Unit tests for skill evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_evaluator():
    path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", path)
    models = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "skills" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(evaluator)
    return models, evaluator


models_mod, evaluator_mod = _load_evaluator()
Skill = models_mod.Skill
TurnTrace = models_mod.TurnTrace
build_run_result = evaluator_mod.build_run_result


def test_build_run_result_success() -> None:
    """Successful trace marks run as succeeded."""
    skill = Skill(
        id="1",
        slug="test",
        title="Test",
        description="Test.",
        triggers=["test"],
        body="body",
        tool_steps=[{"toolName": "callTool"}],
    )
    trace = TurnTrace(
        user_text="test",
        history_len=0,
        tool_calls=[{"toolName": "callTool", "arguments": {}}],
        assistant_text="Done.",
        iterations=1,
    )
    result = build_run_result(skill.id, trace, skill)
    assert result.succeeded is True
    assert result.followed_steps is True


def test_build_run_result_with_tool_error() -> None:
    """Tool errors mark run as failed."""
    skill = Skill(
        id="1",
        slug="test",
        title="Test",
        description="Test.",
        triggers=["test"],
        body="body",
        tool_steps=[],
    )
    trace = TurnTrace(
        user_text="test",
        history_len=0,
        tool_errors=1,
        assistant_text="Sorry",
    )
    result = build_run_result(skill.id, trace, skill)
    assert result.succeeded is False

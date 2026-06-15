"""Unit tests for skill runtime heuristics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_runtime():
    path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", path)
    models = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "skills" / "runtime.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(runtime)
    return models, runtime


models_mod, runtime_mod = _load_runtime()
TurnTrace = models_mod.TurnTrace
should_offer_skill_creation = runtime_mod.should_offer_skill_creation


def test_should_offer_multi_tool_turn() -> None:
    """Two tool calls in one turn qualifies for learning."""
    trace = TurnTrace(
        user_text="do the thing",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_single_tool_first_turn() -> None:
    """One tool on the first turn does not qualify."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=0,
        tool_calls=[{"toolName": "a"}],
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_tool_with_history() -> None:
    """One tool after prior turns qualifies."""
    trace = TurnTrace(
        user_text="try again",
        history_len=4,
        tool_calls=[{"toolName": "a"}],
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_when_skill_matched() -> None:
    """Matched skills skip creation."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=4,
        tool_calls=[{"toolName": "a"}],
        matched_skill_ids=["existing"],
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_learning_disabled() -> None:
    """Learning off blocks creation."""
    trace = TurnTrace(
        user_text="x",
        history_len=4,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
    )
    assert should_offer_skill_creation(trace, learning_enabled=False) is False

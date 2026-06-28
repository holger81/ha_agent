"""Unit tests for skill runtime heuristics."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _ensure_ha_stubs() -> None:
    if "homeassistant.core" in sys.modules:
        return

    ha_pkg = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistant:
        pass

    def callback(func):
        return func

    ha_core.HomeAssistant = HomeAssistant
    ha_core.callback = callback
    sys.modules["homeassistant"] = ha_pkg
    sys.modules["homeassistant.core"] = ha_core


def _load_runtime():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    _ensure_ha_stubs()

    if "ha_agent.const" not in sys.modules:
        path = COMPONENT / "const.py"
        spec = importlib.util.spec_from_file_location("ha_agent.const", path)
        const = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules["ha_agent.const"] = const
        spec.loader.exec_module(const)

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
    sys.modules["ha_agent.skills.runtime"] = runtime
    spec.loader.exec_module(runtime)
    return models, runtime


models_mod, runtime_mod = _load_runtime()
TurnTrace = models_mod.TurnTrace
should_offer_skill_creation = runtime_mod.should_offer_skill_creation
override_turn_eligible_for_learning = runtime_mod.override_turn_eligible_for_learning


def test_should_offer_multi_tool_turn() -> None:
    """Two tool calls in one turn qualifies for learning."""
    trace = TurnTrace(
        user_text="do the thing",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="All set.",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_single_tool_first_turn() -> None:
    """One tool on the first turn does not qualify."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=0,
        tool_calls=[{"toolName": "a"}],
        assistant_text="Done.",
        iterations=1,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_not_offer_single_tool_follow_up() -> None:
    """One tool on a follow-up turn no longer qualifies by history alone."""
    trace = TurnTrace(
        user_text="tell me more",
        history_len=4,
        tool_calls=[{"toolName": "a"}],
        assistant_text="Here is more detail.",
        iterations=1,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_multi_iteration_turn() -> None:
    """Multiple agent iterations with tools qualify."""
    trace = TurnTrace(
        user_text="find and turn off dining lights",
        history_len=0,
        tool_calls=[{"toolName": "a"}],
        assistant_text="Lights are off.",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_without_assistant_text() -> None:
    """Empty replies do not qualify."""
    trace = TurnTrace(
        user_text="do thing",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_tool_with_history() -> None:
    """Two tools in one turn still qualifies."""
    trace = TurnTrace(
        user_text="try again",
        history_len=4,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="Done again.",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_when_learned_skill_matched() -> None:
    """Matched learned skills skip creation."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=4,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        matched_learned_skill_ids=["existing"],
        assistant_text="Done.",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_when_only_builtin_matched() -> None:
    """Builtin route skills do not block auto-learn."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        matched_skill_ids=["builtin-general"],
        matched_learned_skill_ids=[],
        assistant_text="Done.",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_learning_disabled() -> None:
    """Learning off blocks creation."""
    trace = TurnTrace(
        user_text="x",
        history_len=4,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="Done.",
    )
    assert should_offer_skill_creation(trace, learning_enabled=False) is False


def test_should_not_offer_news_content_extraction() -> None:
    """News content summaries are not auto-learned."""
    trace = TurnTrace(
        user_text="what are today's headlines",
        history_len=0,
        route="news",
        tool_calls=[{"toolName": "mcp_news__news_curate"}],
        assistant_text="Here are headlines.",
        iterations=1,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_email_multi_step_workflow() -> None:
    """Multi-step email tool workflows may be auto-learned."""
    trace = TurnTrace(
        user_text="check my inbox for urgent mail",
        history_len=0,
        route="email",
        tool_calls=[
            {"toolName": "mail_mcp__imap_mailbox_status"},
            {"toolName": "mail_mcp__imap_search_messages"},
        ],
        assistant_text="You have 2 urgent messages.",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_manual_save_requires_successful_tools() -> None:
    """Manual save still needs a successful tool workflow."""
    trace = TurnTrace(
        user_text="save this as a skill",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="Done.",
    )
    assert should_offer_skill_creation(
        trace,
        learning_enabled=False,
        manual_save=True,
    ) is True

    failed = TurnTrace(
        user_text="save this as a skill",
        history_len=0,
        tool_calls=[{"toolName": "a"}],
        assistant_text="Done.",
        tool_errors=1,
    )
    assert should_offer_skill_creation(
        failed,
        learning_enabled=False,
        manual_save=True,
    ) is False


def test_override_turn_blocks_generic_skill_creation() -> None:
    """Override turns use dedicated learning instead of generic creation."""
    trace = TurnTrace(
        user_text="mark all emails read",
        history_len=0,
        route="email",
        matched_learned_skill_ids=["skill-1"],
        skill_plan_override=True,
        tool_calls=[
            {"toolName": "mail_mcp__imap_mark_read", "succeeded": True},
        ],
        assistant_text="Marked 3 messages as read.",
        iterations=3,
        outcome="success",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False
    assert override_turn_eligible_for_learning(trace) is True


def test_override_turn_requires_successful_workflow_tools() -> None:
    """Override learning needs at least one successful non-discovery tool."""
    trace = TurnTrace(
        user_text="mark all emails read",
        history_len=0,
        route="email",
        skill_plan_override=True,
        tool_calls=[
            {"toolName": "searchToolsForDomain", "succeeded": True},
        ],
        assistant_text="Could not find a tool.",
        iterations=2,
        outcome="partial",
    )
    assert override_turn_eligible_for_learning(trace) is False

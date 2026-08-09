"""Unit tests for skill runtime heuristics."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


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

    # Minimal observer stub so runtime can import is_discovery_tool.
    # Remove it after load so other test modules can import the real observer.
    stubbed_observer = False
    if "ha_agent.skills.observer" not in sys.modules:
        obs = types.ModuleType("ha_agent.skills.observer")

        def is_discovery_tool(name: str) -> bool:
            lowered = (name or "").lower()
            return "searchtool" in lowered or "tools/list" in lowered

        obs.is_discovery_tool = is_discovery_tool
        obs._is_test_stub = True  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills.observer"] = obs
        stubbed_observer = True

    path = COMPONENT / "skills" / "runtime.py"
    # Always reload so struggle helpers pick up source edits.
    spec = importlib.util.spec_from_file_location("ha_agent.skills.runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.runtime"] = runtime
    spec.loader.exec_module(runtime)
    if stubbed_observer:
        del sys.modules["ha_agent.skills.observer"]
    return models, runtime


models_mod, runtime_mod = _load_runtime()
TurnTrace = models_mod.TurnTrace
should_offer_skill_creation = runtime_mod.should_offer_skill_creation
override_turn_eligible_for_learning = runtime_mod.override_turn_eligible_for_learning
struggle_event_count = runtime_mod.struggle_event_count
is_hard_won_workflow = runtime_mod.is_hard_won_workflow


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
    """Matched learned skills skip creation when followed/unknown."""
    trace = TurnTrace(
        user_text="turn on lights",
        history_len=4,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        matched_learned_skill_ids=["existing"],
        assistant_text="Done.",
        iterations=2,
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_when_matched_skill_not_followed() -> None:
    """Wrong matched skill that was not followed allows new skill learning."""
    trace = TurnTrace(
        user_text="what is the temperature in Jonathans room",
        history_len=0,
        route="action",
        matched_learned_skill_ids=["dining-lights"],
        skill_followed=False,
        tool_calls=[
            {"toolName": "searchTool", "succeeded": True},
            {
                "toolName": "home_assistant__ha_search",
                "succeeded": True,
            },
            {
                "toolName": "home_assistant__ha_get_state",
                "succeeded": True,
            },
        ],
        assistant_text="The temperature in Jonathan's room is 25.0°C.",
        iterations=4,
        tool_errors=1,
        outcome="partial",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_not_offer_when_matched_skill_followed() -> None:
    """Followed learned skills still block new skill creation."""
    trace = TurnTrace(
        user_text="turn off dining room lights",
        history_len=0,
        route="action",
        matched_learned_skill_ids=["dining-lights"],
        skill_followed=True,
        tool_calls=[
            {
                "toolName": "home_assistant__ha_call_service",
                "succeeded": True,
            },
        ],
        controlled_entity_ids=["light.dining"],
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


def test_struggle_and_hard_won_workflow() -> None:
    """Hard-won requires struggle score >= 4 plus a successful MCP tool."""
    easy = TurnTrace(
        user_text="temp?",
        history_len=0,
        route="action",
        tool_calls=[
            {
                "toolName": "home_assistant__ha_search",
                "succeeded": True,
                "arguments": {"query": "x"},
            }
        ],
        assistant_text="22C",
        iterations=1,
        tool_errors=0,
    )
    assert struggle_event_count(easy) == 0
    assert is_hard_won_workflow(easy) is False

    hard = TurnTrace(
        user_text="what is the temperature in Jonathans room?",
        history_len=0,
        route="action",
        tool_calls=[
            {
                "toolName": "home_assistant__ha_search",
                "succeeded": False,
                "arguments": {"query": "jonathan temperature"},
            },
            {
                "toolName": "home_assistant__ha_search",
                "succeeded": False,
                "arguments": {"query": "jonathan"},
            },
            {
                "toolName": "home_assistant__ha_search",
                "succeeded": True,
                "arguments": {"query": "jonathan", "domain_filter": "sensor"},
            },
        ],
        assistant_text="23.3C",
        iterations=6,
        tool_errors=2,
    )
    assert struggle_event_count(hard) >= 4
    assert is_hard_won_workflow(hard) is True


def test_manual_save_requires_successful_tools() -> None:
    """Manual save still needs a successful tool workflow."""
    trace = TurnTrace(
        user_text="save this as a skill",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="Done.",
    )
    assert (
        should_offer_skill_creation(
            trace,
            learning_enabled=False,
            manual_save=True,
        )
        is True
    )

    failed = TurnTrace(
        user_text="save this as a skill",
        history_len=0,
        tool_calls=[{"toolName": "a"}],
        assistant_text="Done.",
        tool_errors=1,
    )
    assert (
        should_offer_skill_creation(
            failed,
            learning_enabled=False,
            manual_save=True,
        )
        is False
    )


def test_override_turn_eligible_for_generic_skill_creation() -> None:
    """Successful override turns can flow into skill creation when eligible."""
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
    assert should_offer_skill_creation(trace, learning_enabled=True) is True
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


def test_should_not_offer_when_assistant_admits_failure() -> None:
    """Failed turns must not prompt skill save even with multi-step tools."""
    trace = TurnTrace(
        user_text="turn off dining room lights",
        history_len=0,
        route="action",
        tool_calls=[
            {"toolName": "light.dining_room", "succeeded": True},
            {"toolName": "searchToolsForDomain", "succeeded": True},
        ],
        assistant_text=(
            "I couldn't find a tool to directly control 'dining room lights'."
        ),
        iterations=3,
        outcome="success",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_not_offer_action_without_mcp_workflow_tool() -> None:
    """Action route needs MCP-shaped tools, not junk names."""
    trace = TurnTrace(
        user_text="turn off dining room lights",
        history_len=0,
        route="action",
        tool_calls=[
            {"toolName": "light.dining_room", "succeeded": True},
            {"toolName": "searchToolsForDomain", "succeeded": True},
        ],
        assistant_text="Tried a local shortcut.",
        iterations=3,
        outcome="success",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is False


def test_should_offer_action_with_read_mcp_tools() -> None:
    """Action-route status reads with MCP tools are learnable."""
    trace = TurnTrace(
        user_text="what is the temperature in Jonathans room",
        history_len=0,
        route="action",
        tool_calls=[
            {"toolName": "home_assistant__ha_search", "succeeded": True},
            {"toolName": "home_assistant__ha_get_state", "succeeded": True},
        ],
        assistant_text="The temperature is 25.0°C.",
        iterations=2,
        outcome="success",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True


def test_should_offer_action_with_control_tool() -> None:
    """Successful HA control turns remain eligible for learning."""
    trace = TurnTrace(
        user_text="turn off dining room lights",
        history_len=0,
        route="action",
        tool_calls=[
            {"toolName": "searchToolsForDomain", "succeeded": True},
            {
                "toolName": "home_assistant__ha_call_service",
                "succeeded": True,
            },
        ],
        controlled_entity_ids=["light.dining_room_lights_ceiling"],
        assistant_text="Done — dining room lights are off.",
        iterations=2,
        outcome="success",
    )
    assert should_offer_skill_creation(trace, learning_enabled=True) is True

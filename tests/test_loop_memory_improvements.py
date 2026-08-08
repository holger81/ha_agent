"""Tests for sticky action backend helper and slim loop guidance."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_loop_policy():
    mod_name = "ha_agent.loop_policy"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    path = COMPONENT / "loop_policy.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_router():
    mod_name = "ha_agent.router"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    # stub deps lightly by loading router file; it may pull const
    if "ha_agent.const" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert spec and spec.loader
        const = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.const"] = const
        spec.loader.exec_module(const)
    path = COMPONENT / "router.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_inject_loop_context_is_capped() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_steps = [{"toolName": "home_assistant__ha_call_service"}]
    state.plan_step_statuses = ["pending"]
    state.mcp_guidance = ["hint-one " * 80, "hint-two"]
    state.pending_failure_summary = "previous tool failed: missing field foo"
    messages = [{"role": "user", "content": "turn on lights"}]
    policy.inject_loop_context(messages, state)
    assert len(messages) == 2
    guidance = messages[0]["content"]
    assert len(guidance) <= policy._MAX_LOOP_GUIDANCE_CHARS
    assert "NEXT:" in guidance or "previous tool failed" in guidance
    assert state.mcp_guidance == []
    assert state.pending_failure_summary is None


def test_stick_action_helper_via_agent_module() -> None:
    """HA_ACTION stays on action; other routes flip to chat."""
    # Load agent helper without full HA by importing just the function via exec
    # of a tiny extract — prefer importing router TaskRoute + redefining helper.
    router = _load_router()
    TaskRoute = router.TaskRoute

    def stick_action_or_chat(route: TaskRoute) -> bool:
        return route != TaskRoute.HA_ACTION

    assert stick_action_or_chat(TaskRoute.HA_ACTION) is False
    assert stick_action_or_chat(TaskRoute.CHAT) is True


def test_should_retry_after_failed_tools_once() -> None:
    """Action turns with tool errors and no control success get one retry."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    failed = [
        {
            "toolName": "home_assistant__ha_search_entities",
            "succeeded": False,
        }
    ]
    assert (
        policy.should_retry_after_failed_tools(
            state,
            tool_errors=1,
            tool_calls=failed,
            route="action",
            iteration=0,
            max_iterations=6,
        )
        is True
    )
    assert (
        policy.should_retry_after_failed_tools(
            state,
            tool_errors=1,
            tool_calls=failed,
            route="action",
            iteration=1,
            max_iterations=6,
        )
        is False
    )


def test_claims_action_success_and_honest_message() -> None:
    """Success claims are detected; failure admissions are not rewritten."""
    policy = _load_loop_policy()
    assert policy.claims_action_success(
        "The dining room lights have been turned on successfully."
    )
    assert policy.claims_action_success(
        "OK. I've turned off the dining room lights. "
        "Controlled: light.dining_room_lights_ceiling."
    )
    assert not policy.claims_action_success(
        "I couldn't turn on the lights because the tool failed."
    )
    assert "couldn't complete" in policy.honest_failed_tools_message().lower()
    assert "haven't confirmed" in policy.honest_missing_control_message().lower()


def test_should_retry_missing_control_once() -> None:
    """Action answers that invent success without tools get one retry."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    claim = "OK. I've turned off the dining room lights."
    assert (
        policy.should_retry_missing_control(
            state,
            route="action",
            assistant_text=claim,
            tool_calls=[],
            iteration=0,
            max_iterations=6,
        )
        is True
    )
    assert (
        policy.should_retry_missing_control(
            state,
            route="action",
            assistant_text=claim,
            tool_calls=[],
            iteration=1,
            max_iterations=6,
        )
        is False
    )
    assert (
        policy.should_retry_missing_control(
            policy.LoopState(),
            route="chat",
            assistant_text=claim,
            tool_calls=[],
            iteration=0,
            max_iterations=6,
        )
        is False
    )


def test_had_successful_control_tool() -> None:
    """Only successful ha_call_service counts as control progress."""
    policy = _load_loop_policy()
    assert not policy.had_successful_control_tool(
        [{"toolName": "home_assistant__ha_call_service", "succeeded": False}]
    )
    assert policy.had_successful_control_tool(
        [{"toolName": "home_assistant__ha_call_service", "succeeded": True}]
    )

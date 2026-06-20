"""Unit tests for agent loop policies."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


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


def test_check_stuck_soft_blocks_first_duplicate() -> None:
    """First duplicate blocks execution but allows the loop to replan."""
    policy = _load_loop_policy()
    state = policy.LoopState()

    assert policy.check_stuck(state, "mail_search", {"unread_only": True}) is None
    blocked = policy.check_stuck(state, "mail_search", {"unread_only": True})

    assert blocked is not None
    assert state.stuck is False
    assert "Review the previous tool result" in blocked


def test_check_stuck_hard_blocks_second_duplicate() -> None:
    """Second duplicate of the same call ends the turn as stuck."""
    policy = _load_loop_policy()
    state = policy.LoopState()

    assert policy.check_stuck(state, "mail_search", {"unread_only": True}) is None
    assert policy.check_stuck(state, "mail_search", {"unread_only": True}) is not None
    blocked = policy.check_stuck(state, "mail_search", {"unread_only": True})

    assert blocked is not None
    assert state.stuck is True
    assert "ask the user for help" in blocked


def test_reasoning_stream_stuck_on_repeat() -> None:
    """Repeated reasoning tails are treated as stuck output."""
    policy = _load_loop_policy()
    phrase = "Wait, I'll try mail_mcp__imap_search_messages with mailbox INBOX. "
    chunk = phrase * 6
    assert policy.reasoning_stream_stuck(chunk) is True


def test_mark_iteration_outcome_stops_after_repeated_blocks() -> None:
    """Two unproductive duplicate-block iterations force stuck."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.iteration_had_duplicate_block = True
    policy.mark_iteration_outcome(state)
    assert state.stuck is False
    state.iteration_had_duplicate_block = True
    policy.mark_iteration_outcome(state)
    assert state.stuck is True


def test_build_pending_failure_summary_for_next_iteration() -> None:
    """Failures compile into an injectable summary for the next loop step."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.record_iteration_failure(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        "Tool error: missing field mailbox",
    )

    policy.build_pending_failure_summary(state)

    assert state.pending_failure_summary is not None
    assert "Do not retry these approaches unchanged" in state.pending_failure_summary
    assert "missing field mailbox" in state.pending_failure_summary
    assert state.iteration_failures == []


def test_inject_pending_failure_summary_appends_user_message() -> None:
    """The next loop step receives the compiled failure summary."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.pending_failure_summary = "TURN PROGRESS SUMMARY\n- failed"
    messages: list[dict[str, str]] = []

    policy.inject_loop_context(messages, state)

    assert len(messages) == 1
    assert messages[0]["role"] == policy.INTERNAL_GUIDANCE_ROLE
    assert messages[0]["role"] != "user"
    assert "TURN PROGRESS SUMMARY" in messages[0]["content"]
    assert state.pending_failure_summary is None


def test_inject_loop_context_uses_system_role_not_user() -> None:
    """Internal guidance is injected as system, never as a user turn."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(state, goal="news briefing", route="news")
    messages: list[dict[str, str]] = []

    policy.inject_loop_context(messages, state)

    assert len(messages) == 1
    assert policy.INTERNAL_GUIDANCE_ROLE == "system"
    assert messages[0]["role"] == "system"
    assert messages[0]["role"] != "user"
    assert "AGENT PLAN PROGRESS" in messages[0]["content"]


def test_initialize_loop_plan_tracks_skill_steps() -> None:
    """Skill tool_steps seed the per-turn plan and focus pointer."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    steps = [
        {"toolName": "mail_mcp__imap_search_messages"},
        {"toolName": "mail_mcp__imap_get_message"},
    ]

    policy.initialize_loop_plan(
        state,
        goal="read the latest email",
        route="email",
        tool_steps=steps,
        skill_title="Read inbox email",
    )

    assert state.plan_current_step_index == 0
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX"},
        succeeded=True,
    )

    assert state.plan_step_statuses == ["done", "pending"]
    assert state.plan_current_step_index == 1


def test_build_plan_progress_summary_marks_needs_work() -> None:
    """Failed tools mark the current step and inject a focus reminder."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="read email",
        route="email",
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {},
        succeeded=False,
    )

    summary = policy.build_plan_progress_summary(state)

    assert summary is not None
    assert "AGENT PLAN PROGRESS" in summary
    assert "[!]" in summary
    assert "still needs work" in summary
    assert "Fix step" in summary


def test_inject_loop_context_includes_plan_and_failures() -> None:
    """Plan progress and failure summary are combined for the next step."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(state, goal="news briefing", route="news")
    state.pending_failure_summary = (
        "TURN PROGRESS SUMMARY\n- news_curate failed"
    )
    messages: list[dict[str, str]] = []

    policy.inject_loop_context(messages, state)

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "AGENT PLAN PROGRESS" in content
    assert "TURN PROGRESS SUMMARY" in content
    assert state.pending_failure_summary is None


def test_describe_plan_next_action_stops_when_all_done() -> None:
    """A completed plan instructs the model to answer instead of calling tools."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(state, goal="news briefing", route="news")
    policy.record_plan_tool_result(state, "news_curate", {}, succeeded=True)

    directive = policy.describe_plan_next_action(state)

    assert "STOP calling tools" in directive
    assert "final" in directive


def test_should_retry_empty_response_caps_attempts() -> None:
    """Empty replies retry a bounded number of times before giving up."""
    policy = _load_loop_policy()
    state = policy.LoopState()

    assert policy.should_retry_empty_response(state, 0, 10) is True
    assert policy.should_retry_empty_response(state, 1, 10) is True
    assert policy.should_retry_empty_response(state, 2, 10) is False
    # Never retry on the final iteration.
    fresh = policy.LoopState()
    assert policy.should_retry_empty_response(fresh, 9, 10) is False


def test_build_empty_response_nudge_includes_next_action() -> None:
    """The empty-response nudge embeds the plan's next directive."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(state, goal="news briefing", route="news")

    nudge = policy.build_empty_response_nudge(state)

    assert "previous reply was empty" in nudge
    assert "news_curate" in nudge


def test_extract_mcp_guidance_pulls_server_context() -> None:
    """serverLlmContext is surfaced from discovery tool output."""
    policy = _load_loop_policy()
    output = json.dumps(
        [
            {"toolName": "a", "serverLlmContext": "Pass mailbox INBOX."},
            {"toolName": "b", "serverLlmContext": "Pass mailbox INBOX."},
            {"toolName": "c"},
        ]
    )

    hints = policy.extract_mcp_guidance("searchToolsForDomain", output)

    assert hints == ["Pass mailbox INBOX."]


def test_extract_mcp_guidance_ignores_non_discovery() -> None:
    """Non-discovery tools and errors yield no guidance."""
    policy = _load_loop_policy()
    payload = json.dumps([{"serverLlmContext": "x"}])
    assert policy.extract_mcp_guidance("ha_call_service", payload) == []
    assert policy.extract_mcp_guidance("searchTool", "Tool error: boom") == []


def test_record_and_inject_mcp_guidance() -> None:
    """Recorded guidance injects once then clears."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps([{"serverLlmContext": "Use domain smart-home."}])

    policy.record_mcp_guidance(state, "searchTool", output)
    assert state.mcp_guidance == ["Use domain smart-home."]

    messages: list[dict[str, str]] = []
    policy.inject_loop_context(messages, state)

    assert "MCP SERVER GUIDANCE" in messages[0]["content"]
    assert "Use domain smart-home." in messages[0]["content"]
    assert state.mcp_guidance == []


def test_enrich_tool_output_adds_email_recovery_hints() -> None:
    policy = _load_loop_policy()
    output = policy.enrich_tool_output(
        "mail_mcp_imap_search_messages",
        {},
        "Tool error: inbox too large to list",
    )

    assert "RECOVERY HINTS" in output
    assert "unread_only" in output


def test_enrich_tool_output_uses_supplied_rules() -> None:
    """Supplied rule objects replace the shipped hardcoded hint logic."""
    policy = _load_loop_policy()
    rule = types.SimpleNamespace(
        enabled=True,
        tool_substring="calendar",
        error_pattern="invalid date",
        body="Retry with an ISO date range.",
    )

    output = policy.enrich_tool_output(
        "calendar_mcp__create_event",
        {},
        "Tool error: invalid date format",
        rules=[rule],
    )

    assert "RECOVERY HINTS" in output
    assert "ISO date range" in output


def test_enrich_tool_output_empty_rules_yield_no_hints() -> None:
    """Supplying an empty rule list suppresses the hardcoded defaults."""
    policy = _load_loop_policy()
    output = policy.enrich_tool_output(
        "mail_mcp_imap_search_messages",
        {},
        "Tool error: inbox too large to list",
        rules=[],
    )

    assert "RECOVERY HINTS" not in output


def test_verify_ha_service_reports_failed_state() -> None:
    policy = _load_loop_policy()
    hass = MagicMock()
    state = MagicMock()
    state.state = "off"
    hass.states.get.return_value = state

    note = policy.verify_ha_service(
        hass,
        "home_assistant__ha_call_service",
        {"entity_id": "light.dining", "service": "turn_on"},
        "ok",
    )

    assert note is not None
    assert note.startswith("VERIFICATION FAILED")

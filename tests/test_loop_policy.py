"""Unit tests for agent loop policies."""

from __future__ import annotations

import importlib.util
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

    policy.inject_pending_failure_summary(messages, state)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "TURN PROGRESS SUMMARY" in messages[0]["content"]
    assert state.pending_failure_summary is None


def test_enrich_tool_output_adds_email_recovery_hints() -> None:
    policy = _load_loop_policy()
    output = policy.enrich_tool_output(
        "mail_mcp_imap_search_messages",
        {},
        "Tool error: inbox too large to list",
    )

    assert "RECOVERY HINTS" in output
    assert "unread_only" in output


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

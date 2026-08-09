"""Unit tests for agent loop policies."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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


def test_check_stuck_allows_repeat_when_pagination_pending() -> None:
    """Paginated tool repeats must not hit duplicate-call blocking."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    args = {"mailbox": "INBOX", "unread_only": True, "limit": 10}
    output = json.dumps(
        {"messages": [{"uid": 1}], "hasMore": True, "offset": 0, "limit": 10}
    )

    assert policy.check_stuck(state, "mail_mcp__imap_search_messages", args) is None
    policy.record_pagination_state(
        state,
        "mail_mcp__imap_search_messages",
        output,
        args,
    )

    assert policy.check_stuck(state, "mail_mcp__imap_search_messages", args) is None
    assert state.stuck is False
    assert state.duplicate_blocks == {}


def test_check_stuck_blocks_duplicate_after_pagination_completes() -> None:
    """Duplicate blocking resumes once pagination is no longer pending."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    args = {"mailbox": "INBOX", "limit": 10}
    paged = json.dumps(
        {"messages": [{"uid": 1}], "hasMore": True, "offset": 0, "limit": 10}
    )
    final = json.dumps({"messages": [{"uid": 2}], "hasMore": False})

    assert policy.check_stuck(state, "mail_mcp__imap_search_messages", args) is None
    policy.record_pagination_state(
        state,
        "mail_mcp__imap_search_messages",
        paged,
        args,
    )
    assert policy.check_stuck(state, "mail_mcp__imap_search_messages", args) is None
    policy.record_pagination_state(
        state,
        "mail_mcp__imap_search_messages",
        final,
        args,
    )

    blocked = policy.check_stuck(state, "mail_mcp__imap_search_messages", args)
    assert blocked is not None
    assert state.stuck is False


def test_reasoning_stream_stuck_on_repeat() -> None:
    """Repeated reasoning tails are treated as stuck output."""
    policy = _load_loop_policy()
    phrase = "Wait, I'll try mail_mcp__imap_search_messages with mailbox INBOX. "
    chunk = phrase * 6
    assert policy.reasoning_stream_stuck(chunk) is True


def test_reasoning_stream_stuck_on_alternating_paraphrases() -> None:
    """Wait/Actually/Let's thrashing on one tool is treated as stuck."""
    policy = _load_loop_policy()
    cycle = (
        "Let's try 'home_assistant__ha_search' with \"jonathan sensor\".\n"
        'Wait, I\'ll try to search for "jonathan" and look for "sensor" '
        "in the 'entity_id'.\n"
        'Actually, I\'ll try to search for "jonathan" and look for "sensor" '
        "in the 'entity_id'.\n"
        "Let's try 'home_assistant__ha_search' with \"jonathan sensor\".\n"
    )
    assert policy.reasoning_stream_stuck(cycle) is True
    assert policy.is_reasoning_loop(cycle, has_tools=False, content="") is True


def test_is_reasoning_loop_ignores_tools_and_answers() -> None:
    """Tool calls or a real answer cancel reasoning-loop detection."""
    policy = _load_loop_policy()
    phrase = "Wait, I'll try mail_mcp__imap_search_messages with mailbox INBOX. "
    stuck = phrase * 6
    assert policy.is_reasoning_loop(stuck, has_tools=False, content="") is True
    assert policy.is_reasoning_loop(stuck, has_tools=True, content="") is False
    assert (
        policy.is_reasoning_loop(
            stuck,
            has_tools=False,
            content="The temperature in Jonathan's room is 25 C.",
        )
        is False
    )


def test_should_retry_reasoning_stuck_caps_attempts() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    assert policy.should_retry_reasoning_stuck(state, 0, 10) is True
    assert policy.should_retry_reasoning_stuck(state, 1, 10) is True
    assert policy.should_retry_reasoning_stuck(state, 2, 10) is False
    fresh = policy.LoopState()
    assert policy.should_retry_reasoning_stuck(fresh, 9, 10) is False


def test_mark_reasoning_stuck_sets_message() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.mark_reasoning_stuck(state)
    assert state.stuck is True
    assert "reasoning" in state.stuck_message.lower()


def test_mark_iteration_outcome_stops_after_repeated_blocks() -> None:
    """Repeated unproductive duplicate-block iterations force stuck."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    for _ in range(policy._MAX_UNPRODUCTIVE_ITERATIONS - 1):
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


def test_initialize_loop_plan_seeds_discovery_guidance_for_action_route() -> None:
    """Action route without skill steps injects smart-home discovery guidance."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="turn off dining room lights",
        route="action",
    )
    assert state.plan_steps == []
    assert state.plan_current_step_index is None
    assert any("smart-home" in hint for hint in state.mcp_guidance)


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
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages"}],
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
    state.pending_failure_summary = "TURN PROGRESS SUMMARY\n- news_curate failed"
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
    policy.initialize_loop_plan(
        state,
        goal="news briefing",
        route="news",
        tool_steps=[{"toolName": "news_curate"}],
    )
    policy.record_plan_tool_result(state, "news_curate", {}, succeeded=True)

    directive = policy.describe_plan_next_action(state)

    assert "STOP calling tools" in directive
    assert "final" in directive


def test_reconcile_plan_after_tools_omits_skipped_prerequisites() -> None:
    """Completing a later step marks earlier pending steps as omitted."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="check inbox",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status"},
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_get_message"},
        ],
    )

    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )

    assert state.plan_step_statuses == ["omitted", "done", "pending"]
    assert "Superseded" in state.plan_step_notes[0]
    assert state.plan_current_step_index == 2


def test_reconcile_plan_before_answer_omits_remaining_pending() -> None:
    """Pending steps are omitted when the model is ready to answer."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="check inbox",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status"},
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_get_message"},
        ],
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX"},
        succeeded=True,
    )

    policy.reconcile_plan_before_answer(state)

    assert state.plan_step_statuses == ["omitted", "done", "omitted"]
    assert "Not required to answer user goal" in state.plan_step_notes[2]
    assert state.plan_current_step_index is None


def test_maybe_omit_plan_steps_from_reasoning() -> None:
    """Explicit OMIT markers in reasoning mark plan steps omitted."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="check inbox",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status"},
            {"toolName": "mail_mcp__imap_search_messages"},
        ],
    )

    policy.maybe_omit_plan_steps_from_reasoning(
        state,
        "OMIT: mail_mcp__imap_mailbox_status — search results already include count.",
    )

    assert state.plan_step_statuses[0] == "omitted"
    assert "search results" in state.plan_step_notes[0]
    assert state.plan_step_statuses[1] == "pending"


def test_build_plan_progress_summary_shows_omitted_steps() -> None:
    """Plan progress lists omitted steps with reasons."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="check inbox",
        route="email",
        tool_steps=[{"toolName": "mail_mcp__imap_mailbox_status"}],
    )
    policy.omit_plan_step(state, 0, "Not required for this request.")

    summary = policy.build_plan_progress_summary(state)

    assert summary is not None
    assert "[~]" in summary
    assert "omitted: Not required for this request." in summary
    assert "OMIT:" in summary


def test_inject_loop_context_includes_plan_on_first_step() -> None:
    """Plan progress is injected before the trailing user turn when present."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="check inbox",
        route="email",
        tool_steps=[{"toolName": "mail_mcp__imap_mailbox_status"}],
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "check inbox"},
    ]

    policy.inject_loop_context(messages, state)

    assert len(messages) == 3
    assert messages[-1]["content"] == "check inbox"
    assert "NEXT: mail_mcp__imap_mailbox_status" in messages[-2]["content"]
    assert len(messages[-2]["content"]) <= policy._MAX_LOOP_GUIDANCE_CHARS


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
    assert "domain `news`" in nudge


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

    assert hints == ["Pass mailbox INBOX.", "Discovered tools: a, b, c"]


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

    assert "MCP:" in messages[0]["content"]
    assert "Use domain smart-home." in messages[0]["content"]
    assert state.mcp_guidance == []
    assert len(messages[0]["content"]) <= policy._MAX_LOOP_GUIDANCE_CHARS


def test_extract_pagination_meta_has_more_offset() -> None:
    """hasMore with offset/limit yields the next offset page."""
    policy = _load_loop_policy()
    output = json.dumps(
        {
            "messages": [{"uid": 1}],
            "hasMore": True,
            "offset": 0,
            "limit": 10,
        }
    )
    meta = policy.extract_pagination_meta(
        output,
        {"mailbox": "INBOX", "limit": 10},
    )
    assert meta == {"kind": "offset", "offset": 10, "limit": 10}


def test_record_pagination_state_injects_guidance() -> None:
    """Paginated tool output adds next-page guidance to the loop state."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps({"items": [1, 2], "hasMore": True, "offset": 10, "limit": 5})

    policy.record_pagination_state(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "offset": 10, "limit": 5},
    )

    assert state.pagination_pending["tool_name"] == "mail_mcp__imap_search_messages"
    assert state.pagination_pending["offset"] == 15
    assert any("PAGINATION" in hint for hint in state.mcp_guidance)


def test_redundant_override_allows_search_when_pagination_pending() -> None:
    """Repeat calls are allowed for any tool while paginated results remain."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "list all news headlines"
    state.plan_route = "news"
    policy.suspend_skill_plan(state, "Goal exceeds active skill.")
    state.plan_steps = [{"toolName": "mcp_news__news_list"}]
    state.plan_step_statuses = ["done"]
    output = json.dumps(
        {"items": [{"title": "Headline"}], "hasMore": True, "offset": 0, "limit": 10}
    )
    policy.record_pagination_state(
        state,
        "mcp_news__news_list",
        output,
        {"limit": 10},
    )

    block = policy.redundant_override_tool_block(state, "mcp_news__news_list")
    assert block is None
    assert any("PAGINATION" in hint for hint in state.mcp_guidance)


def test_redundant_override_allows_email_search_when_pagination_pending() -> None:
    """Email search pagination uses the same universal repeat allowance."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.plan_route = "email"
    state.skill_plan_override = True
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages"}],
    )
    output = json.dumps(
        {"messages": [{"uid": 1}], "hasMore": True, "offset": 0, "limit": 10}
    )
    policy.record_pagination_state(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "unread_only": True, "limit": 10},
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )

    block = policy.redundant_override_tool_block(
        state,
        "mail_mcp__imap_search_messages",
    )
    assert block is None
    assert not any("Do not search" in hint for hint in state.mcp_guidance)


def test_reasoning_execution_mismatch_detects_wrong_tool() -> None:
    """Reasoning that commits to news_curate blocks mail tool execution."""
    policy = _load_loop_policy()
    reasoning = (
        "The user asked for news. I will call `mcp_news__news_curate` with no "
        "arguments."
    )
    mismatch = policy.reasoning_execution_mismatch(
        reasoning,
        ["mail_mcp__imap_search_messages"],
    )

    assert mismatch is not None
    assert "mcp_news__news_curate" in mismatch
    assert "mail_mcp__imap_search_messages" in mismatch


def test_reasoning_execution_mismatch_allows_discovery_tools() -> None:
    """MCP discovery/search tools are never blocked by reasoning mismatch."""
    policy = _load_loop_policy()
    reasoning = "I will call `mcp_news__news_local`."
    assert (
        policy.reasoning_execution_mismatch(
            reasoning,
            ["searchToolsForDomain", "searchTool"],
        )
        is None
    )


def test_reasoning_execution_mismatch_allows_matching_tool() -> None:
    """Aligned reasoning and execution do not produce a mismatch."""
    policy = _load_loop_policy()
    reasoning = "I will call `mcp_news__news_curate`."
    assert (
        policy.reasoning_execution_mismatch(
            reasoning,
            ["mcp_news__news_curate"],
        )
        is None
    )


def test_user_requests_skill_override() -> None:
    """User phrases can explicitly bypass the active skill workflow."""
    policy = _load_loop_policy()
    assert policy.user_requests_skill_override("ignore the skill and search tools")
    assert policy.user_requests_skill_override("mark read without the skill")
    assert not policy.user_requests_skill_override("mark all emails read")


def test_suspend_skill_plan_leaves_discovery_only_override() -> None:
    """Skill suspension clears concrete steps and keeps discovery guidance."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.plan_route = "email"
    policy.suspend_skill_plan(state, "Goal exceeds active skill.")
    assert state.plan_steps == []
    assert any("Discover MCP tools" in hint for hint in state.mcp_guidance)


def test_should_block_reasoning_execution_mismatch_when_plan_suspended() -> None:
    """Reasoning mismatch checks stay off after the skill plan is suspended."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="mark read",
        route="email",
        skill_title="Email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_get_message"},
        ],
    )
    assert policy.should_block_reasoning_execution_mismatch(state) is True
    policy.suspend_skill_plan(state, "Goal exceeds active skill.")
    assert policy.should_block_reasoning_execution_mismatch(state) is False


def test_record_plan_tool_result_keeps_done_on_later_failure() -> None:
    """A completed plan step is not downgraded by later failed retries."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_bulk_update_flags"},
        ],
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )
    assert state.plan_step_statuses[0] == "done"
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=False,
    )
    assert state.plan_step_statuses[0] == "done"


def test_redundant_override_tool_block_after_search() -> None:
    """Repeat search is blocked once the override exploration plan advances."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.plan_route = "email"
    state.skill_plan_override = True
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_bulk_update_flags"},
        ],
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )
    block = policy.redundant_override_tool_block(
        state,
        "mail_mcp__imap_search_messages",
    )
    assert block is not None
    assert "bulk_update_flags" in block


def test_reasoning_skill_override_marker() -> None:
    """SKILL_OVERRIDE marker suspends the enforced skill plan."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="mark emails read",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_get_message"},
        ],
        skill_title="Check unread email",
    )

    reasoning = (
        "The active skill only reads mail. SKILL_OVERRIDE: user wants mark-as-read."
    )
    assert policy.maybe_suspend_skill_plan_from_reasoning(state, reasoning) is True
    assert state.skill_plan_override is True
    assert state.plan_steps == []
    assert not policy.skill_plan_blocks_discovery(state)


def test_skill_plan_seeds_calltool_not_discovery() -> None:
    """Skill tool_steps should steer to callTool, not searchTool."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="what is the outdoor air quality",
        route="action",
        tool_steps=[
            {
                "toolName": "home_assistant__ha_get_state",
                "arguments": {"entity_id": "sensor.home_outdoor_aqi_5min_mean"},
            },
            {"toolName": "home_assistant__ha_search"},
        ],
        skill_title="Look up sensor or entity status",
    )
    assert policy.skill_plan_blocks_discovery(state)
    assert any("callTool" in hint for hint in state.mcp_guidance)
    assert any("home_assistant__ha_get_state" in hint for hint in state.mcp_guidance)
    assert not any(
        "Discover this tool with searchTool" in hint for hint in state.mcp_guidance
    )
    next_action = policy.describe_plan_next_action(state)
    assert "callTool" in next_action
    assert "home_assistant__ha_get_state" in next_action


def test_skill_discovery_block_steers_to_calltool() -> None:
    """Blocked searchTool must nudge callTool and count as unproductive."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="what is the outdoor air quality",
        route="action",
        tool_steps=[
            {"toolName": "home_assistant__ha_get_state"},
            {"toolName": "home_assistant__ha_search"},
        ],
        skill_title="Look up sensor or entity status",
    )
    blocked = policy.build_skill_discovery_block_message(state)
    assert "callTool" in blocked
    assert "home_assistant__ha_get_state" in blocked
    assert "searchTool" in blocked.lower() or "discovery" in blocked.lower()

    policy.record_skill_discovery_block_guidance(state, "searchTool", blocked)
    assert state.iteration_had_duplicate_block is True
    assert state.discovery_streak == 1
    assert any("DISCOVERY BLOCKED" in hint for hint in state.mcp_guidance)
    assert any(
        "do not run searchTool" in hint.lower() or "Call callTool" in hint
        for hint in state.mcp_guidance
    )

    adherence = policy.build_mcp_tool_adherence_hint(
        state,
        "home_assistant__ha_get_state",
        lead_in="Required next skill-plan tool:",
    )
    assert "Call callTool" in adherence
    assert "Discover this tool with searchTool" not in adherence


def test_reasoning_declares_skill_mismatch_without_marker() -> None:
    """Explicit mismatch reasoning allows discovery without the marker."""
    policy = _load_loop_policy()
    reasoning = (
        "Active skills are check-and-read-unread-emails. Neither includes a "
        "mark-as-read step. I need to discover a mark-as-read tool."
    )
    assert policy.reasoning_declares_skill_mismatch(reasoning)
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="mark all above emails read",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_get_message"},
        ],
        skill_title="Check unread email",
    )
    assert policy.maybe_suspend_skill_plan_from_reasoning(state, reasoning) is True
    assert not policy.skill_plan_blocks_discovery(state)


def test_build_plan_progress_summary_when_skill_overridden() -> None:
    """Override keeps the full step checklist and suspension reason."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.plan_route = "email"
    policy.suspend_skill_plan(state, "Skill only covers unread checks.")

    summary = policy.build_plan_progress_summary(state)

    assert summary is not None
    assert "suspended" in summary
    assert "Skill only covers unread checks." in summary
    assert "No concrete override steps seeded" in summary
    assert "Next action:" in summary
    assert "Discover tools in domain `email`" in summary


def test_build_plan_progress_summary_includes_rejected_draft() -> None:
    """Verifier retries inject the previous answer into plan guidance."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    policy.mark_iteration_preserve_stream(
        state,
        draft_answer="I marked all unread emails as read.",
    )
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[{"toolName": "mail_mcp__imap_bulk_update_flags"}],
    )

    summary = policy.build_plan_progress_summary(state)

    assert summary is not None
    assert "Previous answer attempt" in summary
    assert "I marked all unread emails as read." in summary
    assert "mail_mcp__imap_bulk_update_flags" in summary


def test_mark_iteration_after_tools_clears_stream_preservation() -> None:
    """Tool iterations reset stream preservation for the next UI pass."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.mark_iteration_preserve_stream(state, draft_answer="draft")
    policy.mark_iteration_after_tools(state)
    assert state.preserve_stream_ui is False
    assert state.last_draft_answer == "draft"


def test_record_override_block_guidance_injects_next_action() -> None:
    """Blocked repeat tools inject the required next plan action."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.skill_plan_override = True
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_bulk_update_flags"},
        ],
    )
    policy.cache_mcp_tool_catalog_entry(
        state,
        "mail_mcp__imap_bulk_update_flags",
        description="Bulk update message flags.",
        parameters="Required: mailbox, message_ids, flags",
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )

    policy.record_override_block_guidance(
        state,
        "mail_mcp__imap_search_messages",
        "Tool error: Unread search already succeeded.",
    )

    assert state.iteration_had_duplicate_block is True
    assert state.override_block_count == 1
    assert any("Adhere strictly" in hint for hint in state.mcp_guidance)
    assert any("message_ids" in hint for hint in state.mcp_guidance)


def test_analyze_search_tool_result_steers_exploration_after_search() -> None:
    """After search during exploration, inject MCP adherence for next tool."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails in INBOX as read"
    state.skill_plan_override = True
    policy.cache_mcp_tool_catalog_entry(
        state,
        "mail_mcp__imap_bulk_update_flags",
        description="Bulk update message flags.",
        parameters="Required: mailbox, message_ids, flags",
    )
    output = json.dumps({"messages": [{"uid": 42}], "total": 1})

    policy.analyze_search_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "unread_only": True},
    )

    assert any("bulk_update_flags" in hint for hint in state.mcp_guidance)
    assert any("message_ids" in hint for hint in state.mcp_guidance)


def test_analyze_search_tool_result_tells_model_to_answer_for_inbox_check() -> None:
    """Check-inbox goals should answer from search results instead of re-searching."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "any new emails in INBOX?"
    output = json.dumps({"messages": [{"uid": 1, "subject": "Hi"}], "total": 1})

    policy.analyze_search_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "unread_only": True},
    )

    assert any(
        "Answer the user from these results" in hint for hint in state.mcp_guidance
    )
    assert any("Do not repeat" in hint for hint in state.mcp_guidance)


def test_analyze_discovery_tool_result_prompts_direct_call() -> None:
    """searchTool for a concrete tool name should steer to calling that tool."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    state.skill_plan_override = True
    output = json.dumps(
        [
            {
                "toolName": "mail_mcp__imap_bulk_update_flags",
                "description": "Bulk update message flags.",
                "inputSchema": {
                    "required": ["mailbox", "message_ids", "flags"],
                    "properties": {"message_ids": {"description": "From search"}},
                },
            }
        ]
    )
    policy.analyze_discovery_tool_result(
        state,
        "searchTool",
        output,
        {"query": "mail_mcp__imap_bulk_update_flags"},
    )
    assert any("Call this tool via callTool now" in hint for hint in state.mcp_guidance)
    assert any("message_ids" in hint for hint in state.mcp_guidance)


def test_analyze_discovery_misses_doc_only_tool_name_hits() -> None:
    """searchTool(short name) often returns docs that mention it, not the tool."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps(
        [
            {
                "toolName": "home_assistant__ha_call_service",
                "description": "Use ha_search() to find entity IDs first.",
            },
            {
                "toolName": "home_assistant__ha_get_entity",
                "description": "Related: ha_search() finds entities by name.",
            },
        ]
    )
    policy.analyze_discovery_tool_result(
        state,
        "searchTool",
        output,
        {"query": "ha_search", "domain": "smart-home"},
    )
    assert state.mcp_guidance
    assert any("did not return a tool named" in hint for hint in state.mcp_guidance)
    # Prefix inferred from returned hits — not hard-coded per domain.
    assert any("home_assistant__ha_search" in hint for hint in state.mcp_guidance)


def test_analyze_discovery_miss_infers_prefix_from_any_server() -> None:
    """Doc-only misses use the server prefix from whatever tools were returned."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps(
        [
            {
                "toolName": "mail_mcp__imap_get_message",
                "description": "Fetch one message. Prefer imap_search_messages first.",
            }
        ]
    )
    policy.analyze_discovery_tool_result(
        state,
        "searchTool",
        output,
        {"query": "imap_search_messages", "domain": "email"},
    )
    assert any("mail_mcp__imap_search_messages" in hint for hint in state.mcp_guidance)


def test_analyze_discovery_streak_stops_loops() -> None:
    """Repeated discovery without a concrete call injects a stop-looping hint."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps(
        [{"toolName": "home_assistant__ha_call_service", "description": "control"}]
    )
    policy.analyze_discovery_tool_result(
        state, "searchTool", output, {"query": "temperature"}
    )
    policy.analyze_discovery_tool_result(
        state, "searchTool", output, {"query": "get_state"}
    )
    assert state.discovery_streak >= 2
    assert any("Stop discovery loops" in hint for hint in state.mcp_guidance)
    assert any("home_assistant__" in hint for hint in state.mcp_guidance)
    assert not any("ha_search" in hint for hint in state.mcp_guidance)


def test_build_mcp_tool_adherence_hint_uses_catalog() -> None:
    """Next-step guidance cites cached MCP metadata instead of hard-coded args."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.cache_mcp_tool_catalog_entry(
        state,
        "mail_mcp__imap_bulk_update_flags",
        description="Bulk update message flags.",
        server_llm_context="Pass message_ids from search results.",
        parameters="Required: mailbox, message_ids, flags",
    )

    hint = policy.build_mcp_tool_adherence_hint(
        state,
        "mail_mcp__imap_bulk_update_flags",
        lead_in="Next plan step:",
    )

    assert "Adhere strictly to the MCP tool definition" in hint
    assert "Bulk update message flags." in hint
    assert "message_ids" in hint
    assert "UIDs" not in hint


def test_analyze_search_tool_result_injects_mcp_adherence() -> None:
    """Search/list results produce a factual summary and MCP adherence hint."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "mark all unread emails as read"
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_bulk_update_flags"},
        ],
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX", "unread_only": True},
        succeeded=True,
    )
    policy.cache_mcp_tool_catalog_entry(
        state,
        "mail_mcp__imap_bulk_update_flags",
        description="Update flags for message_ids.",
        parameters="Required: mailbox, message_ids, flags",
    )
    output = json.dumps(
        {
            "messages": [
                {"uid": 1, "flags": [r"\Recent", r"\Seen"]},
                {"uid": 2, "flags": [r"\Seen"]},
            ],
            "total": 2,
        }
    )

    policy.analyze_search_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "unread_only": True},
    )

    assert any("SEARCH RESULT" in hint for hint in state.mcp_guidance)
    assert any("Adhere strictly" in hint for hint in state.mcp_guidance)
    assert any("message_ids" in hint for hint in state.mcp_guidance)
    assert not any("UIDs" in hint for hint in state.mcp_guidance)


def test_analyze_search_tool_result_counts_items() -> None:
    """Search results steer the agent to the next tool via MCP metadata."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="mark unread",
        route="email",
        tool_steps=[
            {"toolName": "mail_mcp__imap_search_messages"},
            {"toolName": "mail_mcp__imap_bulk_update_flags"},
        ],
    )
    policy.record_plan_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        {"mailbox": "INBOX"},
        succeeded=True,
    )
    policy.cache_mcp_tool_catalog_entry(
        state,
        "mail_mcp__imap_bulk_update_flags",
        description="Set flags using message_ids from prior search output.",
    )
    output = json.dumps(
        {
            "messages": [
                {"uid": 42, "flags": [r"\Recent"]},
                {"uid": 43, "flags": [r"\Seen"]},
            ]
        }
    )

    policy.analyze_search_tool_result(
        state,
        "mail_mcp__imap_search_messages",
        output,
        {"mailbox": "INBOX", "unread_only": True},
    )

    assert any("returned 2 item(s)" in hint for hint in state.mcp_guidance)
    assert any("Adhere strictly" in hint for hint in state.mcp_guidance)
    assert any("message_ids" in hint for hint in state.mcp_guidance)


def test_analyze_search_tool_result_empty_ha_search_soft_recovery() -> None:
    """Empty multi-word reading search prefers the place token, not temperature."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "query": "jonathan temperature",
            "entities": [],
            "entity_total_matches": 0,
        }
    )

    unproductive = policy.analyze_search_tool_result(
        state,
        "home_assistant__ha_search",
        output,
        {"query": "jonathan temperature", "domain_filter": "sensor"},
    )

    assert unproductive is True
    assert state.empty_entity_search_attempts == 1
    assert state.mcp_guidance
    hint = state.mcp_guidance[0]
    assert "no matching entities" in hint
    assert "domain_filter=`sensor`" in hint
    assert "query=`jonathans`" in hint
    assert "query=`temperature`" not in hint
    assert "Do not answer yet" in hint
    assert "Answer the user from these results" not in hint


def test_analyze_search_tool_result_allows_multiple_empty_retries() -> None:
    """Empty entity searches get several progressive retries before giving up."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    output = json.dumps(
        {
            "success": True,
            "query": "outdoor aqi",
            "entities": [],
            "entity_total_matches": 0,
        }
    )
    args = {"query": "outdoor aqi", "domain_filter": "sensor"}

    for _ in range(3):
        assert (
            policy.analyze_search_tool_result(
                state, "home_assistant__ha_search", output, args
            )
            is True
        )

    assert state.empty_entity_search_attempts == 3
    # Latest hint should be the exhausted-retry message.
    assert any("Do not repeat the same" in hint for hint in state.mcp_guidance)


def test_analyze_entity_lookup_miss_nudges_comparable_search() -> None:
    """Missing entity_id reopens the plan and points at comparable ha_search."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    policy.initialize_loop_plan(
        state,
        goal="anything about outdoor aqi?",
        route="action",
        tool_steps=[
            {"toolName": "home_assistant__ha_get_state"},
            {"toolName": "home_assistant__ha_search"},
        ],
    )
    # Pretend the skill step was marked done from a "successful" MCP envelope.
    state.plan_step_statuses[0] = "done"
    state.plan_completed_tools.append("home_assistant__ha_get_state")

    output = json.dumps(
        {
            "success": False,
            "error": {
                "code": "ENTITY_NOT_FOUND",
                "message": "Entity not found",
            },
            "entity_id": "sensor.purpleair_san_jose_home_outdoor_5min_mean",
        }
    )
    unproductive = policy.analyze_entity_lookup_result(
        state,
        "home_assistant__ha_get_state",
        output,
        {
            "data": {
                "entity_id": "sensor.purpleair_san_jose_home_outdoor_5min_mean",
            }
        },
    )

    assert unproductive is True
    assert state.plan_step_statuses[0] == "needs_work"
    assert "home_assistant__ha_get_state" not in state.plan_completed_tools
    assert state.mcp_guidance
    hint = state.mcp_guidance[0]
    assert "ENTITY LOOKUP FAILED" in hint
    assert "domain_filter=`sensor`" in hint
    assert "ha_search" in hint
    assert "unit_of_measurement" in hint
    assert "Do not answer yet" in hint


def test_analyze_entity_lookup_unavailable_state_nudges_search() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "outdoor air quality"
    output = json.dumps(
        {
            "data": {
                "entity_id": "sensor.purpleair_airquality_a",
                "state": "unavailable",
            }
        }
    )
    assert (
        policy.analyze_entity_lookup_result(
            state,
            "home_assistant__ha_get_state",
            output,
            {"entity_id": "sensor.purpleair_airquality_a"},
        )
        is True
    )
    assert any("comparable" in hint for hint in state.mcp_guidance)


def test_analyze_entity_lookup_validation_failed_nudges_search() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "temperature in Jonathans room"
    output = json.dumps(
        {
            "success": False,
            "error": {
                "code": "VALIDATION_FAILED",
                "message": "Invalid entity_id",
            },
        }
    )
    assert (
        policy.analyze_entity_lookup_result(
            state,
            "home_assistant__ha_get_state",
            output,
            {"entity_id": "sensor.emilias_room_temperature"},
        )
        is True
    )
    assert any("ENTITY LOOKUP FAILED" in hint for hint in state.mcp_guidance)


def test_analyze_search_tool_result_counts_ha_search_entities() -> None:
    """ha_search entity lists are counted (not treated as empty)."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "query": "jonathan",
            "entities": [
                {
                    "entity_id": "sensor.jonathan_s_lights_energy",
                    "friendly_name": "Jonathan's Lights Energy",
                    "state": "2.82",
                },
                {
                    "entity_id": "sensor.jonathans_bedroom_9b3a_temperature",
                    "friendly_name": "Jonathans Bedroom-9b3a Temperature",
                    "state": "22.9",
                    "unit_of_measurement": "°C",
                    "device_class": "temperature",
                },
                {
                    "entity_id": "sensor.jonathans_bedroom_9b3a_voltage",
                    "friendly_name": "Jonathans Bedroom-9b3a Voltage",
                    "state": "2.76",
                    "unit_of_measurement": "V",
                    "device_class": "voltage",
                },
            ],
            "entity_total_matches": 3,
        }
    )

    unproductive = policy.analyze_search_tool_result(
        state,
        "home_assistant__ha_search",
        output,
        {"query": "jonathan", "domain_filter": "sensor"},
    )

    assert unproductive is False
    assert (
        state.confirmed_reading_entity_id == "sensor.jonathans_bedroom_9b3a_temperature"
    )
    assert any(
        "READING CANDIDATES (temperature)" in hint for hint in state.mcp_guidance
    )
    assert any("Confirmed" in hint for hint in state.mcp_guidance)
    assert any(
        "sensor.jonathans_bedroom_9b3a_temperature" in hint
        for hint in state.mcp_guidance
    )
    assert not any(
        "sensor.jonathan_s_lights_energy" in hint for hint in state.mcp_guidance
    )
    assert not any(
        "Answer the user from these results" in hint for hint in state.mcp_guidance
    )


def test_analyze_search_misses_reading_kind_keeps_searching() -> None:
    """Hits without the asked reading type are unproductive."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "query": "jonathan",
            "entities": [
                {
                    "entity_id": "sensor.jonathan_s_lights_energy",
                    "friendly_name": "Jonathan's Lights Energy",
                    "state": "2.82",
                }
            ],
            "entity_total_matches": 1,
        }
    )
    assert (
        policy.analyze_search_tool_result(
            state,
            "home_assistant__ha_search",
            output,
            {"query": "jonathan", "domain_filter": "sensor"},
        )
        is True
    )
    assert any("No temperature sensors" in hint for hint in state.mcp_guidance)
    # Prefer place token retry over paging all temperature sensors.
    assert any("query=`jonathans`" in hint for hint in state.mcp_guidance)


def test_analyze_search_place_page_without_temperature_paginates() -> None:
    """Humidity/rssi-first place pages should paginate, not give up."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "query": "jonathan",
            "entities": [
                {
                    "entity_id": "sensor.jonathans_bedroom_9b3a_humidity",
                    "friendly_name": "Jonathan humidity",
                    "state": "56",
                },
                {
                    "entity_id": "sensor.jonathans_bedroom_9b3a_rssi",
                    "friendly_name": "Jonathan rssi",
                    "state": "-66",
                },
            ],
            "entity_total_matches": 40,
            "has_more": True,
            "entity_next_offset": 10,
            "next_offset": 10,
        }
    )
    assert (
        policy.analyze_search_tool_result(
            state,
            "home_assistant__ha_search",
            output,
            {"query": "jonathan", "domain_filter": "sensor"},
        )
        is True
    )
    assert state.suppress_pagination is False
    assert any("More results available" in hint for hint in state.mcp_guidance)
    assert any("offset=`10`" in hint for hint in state.mcp_guidance)


def test_missing_reading_nudge_prefers_place_token() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    nudge = policy.build_missing_reading_nudge(state)
    assert "query=`temperature`" not in nudge
    assert "query=`jonathans`" in nudge


def test_analyze_search_wrong_place_temperature_stops_pagination() -> None:
    """Generic temperature pages that miss the place are unproductive."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "query": "temperature",
            "entities": [
                {
                    "entity_id": "sensor.attic_sensor_temperature",
                    "friendly_name": "Attic Sensor temperature",
                    "device_class": "temperature",
                    "state": "32.2",
                }
            ],
            "entity_total_matches": 40,
            "has_more": True,
            "next_offset": 10,
        }
    )
    assert (
        policy.analyze_search_tool_result(
            state,
            "home_assistant__ha_search",
            output,
            {"query": "temperature", "domain_filter": "sensor"},
        )
        is True
    )
    assert state.suppress_pagination is True
    assert any("none matching jonathans" in hint for hint in state.mcp_guidance)
    assert any("query=`jonathans`" in hint for hint in state.mcp_guidance)


def test_scrub_mismatched_reading_slots_clears_wrong_entity() -> None:
    policy = _load_loop_policy()
    scrubbed = policy.scrub_mismatched_reading_slots(
        {"value": "sensor.emilias_room_humidity", "query": "aqi"},
        "what is the outdoor air quality?",
    )
    assert scrubbed["value"] == ""
    assert scrubbed["query"] == "aqi"
    kept = policy.scrub_mismatched_reading_slots(
        {"value": "sensor.home_outdoor_aqi_5min_mean"},
        "what is the outdoor air quality?",
    )
    assert kept["value"] == "sensor.home_outdoor_aqi_5min_mean"


def test_scrub_mismatched_plan_entities_drops_wrong_entity_id() -> None:
    policy = _load_loop_policy()
    steps = policy.scrub_mismatched_plan_entities(
        [
            {
                "toolName": "home_assistant__ha_get_state",
                "arguments": {"entity_id": "sensor.emilias_room_humidity"},
            }
        ],
        "outdoor air quality",
    )
    assert steps is not None
    assert "entity_id" not in steps[0]["arguments"]


def test_honest_missing_reading_uses_an_for_aqi() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "outdoor air quality"
    assert "an aqi" in policy.honest_missing_reading_message(state)


def test_analyze_entity_lookup_rejects_wrong_place_temperature() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Jonathans room"
    output = json.dumps(
        {
            "success": True,
            "data": {
                "entity_id": "sensor.attic_sensor_temperature",
                "state": "32.2",
                "attributes": {
                    "unit_of_measurement": "°C",
                    "device_class": "temperature",
                    "friendly_name": "Attic Sensor temperature",
                },
            },
        }
    )
    assert (
        policy.analyze_entity_lookup_result(
            state,
            "home_assistant__ha_get_state",
            output,
            {"entity_id": "sensor.attic_sensor_temperature"},
        )
        is True
    )
    assert state.confirmed_reading_entity_id is None
    assert any("STATE PLACE MISMATCH" in hint for hint in state.mcp_guidance)


def test_analyze_entity_lookup_rejects_wrong_reading_type() -> None:
    """Voltage get_state is unproductive when the goal is temperature."""
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Emilias room"
    policy.initialize_loop_plan(
        state,
        goal=state.plan_goal,
        route="action",
        tool_steps=[{"toolName": "home_assistant__ha_get_state"}],
    )
    state.plan_step_statuses[0] = "done"
    state.plan_completed_tools.append("home_assistant__ha_get_state")

    output = json.dumps(
        {
            "success": True,
            "data": {
                "entity_id": "sensor.emilias_room_2733_voltage",
                "state": "2.83",
                "attributes": {
                    "unit_of_measurement": "V",
                    "device_class": "voltage",
                    "friendly_name": "Emilia's Room-2733 Voltage",
                },
            },
        }
    )
    assert (
        policy.analyze_entity_lookup_result(
            state,
            "home_assistant__ha_get_state",
            output,
            {"entity_id": "sensor.emilias_room_2733_voltage"},
        )
        is True
    )
    assert state.confirmed_reading_entity_id is None
    assert state.plan_step_statuses[0] == "needs_work"
    assert any("STATE MISMATCH" in hint for hint in state.mcp_guidance)


def test_analyze_entity_lookup_confirms_matching_temperature() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Emilias room"
    output = json.dumps(
        {
            "success": True,
            "data": {
                "entity_id": "sensor.emilias_room_2733_temperature",
                "state": "22.83",
                "attributes": {
                    "unit_of_measurement": "°C",
                    "device_class": "temperature",
                    "friendly_name": "Emilia's Room-2733 Temperature",
                },
            },
        }
    )
    assert (
        policy.analyze_entity_lookup_result(
            state,
            "home_assistant__ha_get_state",
            output,
            {"entity_id": "sensor.emilias_room_2733_temperature"},
        )
        is False
    )
    assert state.confirmed_reading_entity_id == "sensor.emilias_room_2733_temperature"


def test_should_retry_missing_reading_without_confirmed_state() -> None:
    policy = _load_loop_policy()
    state = policy.LoopState()
    state.plan_goal = "what is the temperature in Emilias room"
    claim = "The temperature in Emilia's room is 2.83°C."
    assert policy.needs_confirmed_reading(state, claim) is True
    assert (
        policy.should_retry_missing_reading(
            state,
            assistant_text=claim,
            iteration=0,
            max_iterations=6,
        )
        is True
    )
    assert "haven't confirmed" in policy.honest_missing_reading_message(state).lower()
    state.confirmed_reading_entity_id = "sensor.emilias_room_2733_temperature"
    assert policy.needs_confirmed_reading(state, claim) is False


def test_enrich_tool_output_adds_search_entities_recovery() -> None:
    """Unknown tools steer the model toward discovery."""
    policy = _load_loop_policy()
    output = policy.enrich_tool_output(
        "home_assistant__ha_search_entities",
        {},
        "Tool error: Unknown tool: 'ha_search_entities'",
    )

    assert "RECOVERY HINTS" in output
    assert "searchToolsForDomain" in output


def test_enrich_tool_output_adds_generic_large_result_hint() -> None:
    policy = _load_loop_policy()
    output = policy.enrich_tool_output(
        "mail_mcp_imap_search_messages",
        {},
        "Tool error: inbox too large to list",
    )

    assert "RECOVERY HINTS" in output
    assert "MCP parameters" in output


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

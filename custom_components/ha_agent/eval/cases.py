"""Built-in and promoted eval benchmark cases."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from .models import EVAL_TASKS, EvalCase
from .store import get_eval_store

_BUILTIN_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="light_off",
        task="action",
        user_text="turn off the dining room lights",
        exposed_entities=[
            {
                "entity_id": "light.dining",
                "name": "Dining",
                "state": "on",
                "area_name": "Dining room",
            }
        ],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "light",
            "service": "turn_off",
            "entity_id": "light.dining",
        },
        expected_text_contains=["off"],
        mock_mcp_responses=['{"success": true}'],
    ),
    EvalCase(
        id="cover_open",
        task="action",
        user_text="open the patio cover",
        exposed_entities=[],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "cover",
            "service": "open_cover",
            "entity_id": "cover.patio",
        },
        expected_text_contains=["open"],
        mock_mcp_responses=[
            '{"tools":[{"toolName":"home_assistant__ha_call_service"}]}',
            '{"success": true}',
        ],
        max_iterations=8,
    ),
    EvalCase(
        id="news_headlines",
        task="news",
        user_text="What's the news?",
        expected_tool="mcp_news__news_curate",
        expected_text_contains=["headline"],
        mock_mcp_responses=['{"headlines":["Example headline"]}'],
    ),
    EvalCase(
        id="email_unread",
        task="email",
        user_text="how many unread emails do I have",
        expected_tool="mail_mcp__imap_search_messages",
        expected_text_contains=["3"],
        mock_mcp_responses=['{"count": 3}'],
    ),
    EvalCase(
        id="chat_weather",
        task="chat",
        user_text="what is the weather like today",
        expected_tool=None,
        expected_text_contains=["weather"],
        mock_mcp_responses=[],
        max_iterations=4,
    ),
    EvalCase(
        id="classifier_movie_night",
        task="classifier",
        user_text="dim the living room lights for movie night",
        expected_playbook_route="movie_night",
        mock_mcp_responses=[],
        max_iterations=1,
    ),
    EvalCase(
        id="classifier_email_inbox",
        task="classifier",
        user_text="how many unread emails do I have in my inbox",
        expected_playbook_route="email",
        mock_mcp_responses=[],
        max_iterations=1,
    ),
    EvalCase(
        id="classifier_news_briefing",
        task="classifier",
        user_text="give me today's news briefing",
        expected_playbook_route="news",
        mock_mcp_responses=[],
        max_iterations=1,
    ),
)


def list_eval_cases(
    *,
    tasks: list[str] | None = None,
    custom_cases: list[EvalCase] | None = None,
) -> list[EvalCase]:
    """Return built-in cases plus optional promoted cases."""
    allowed = set(tasks or EVAL_TASKS)
    builtin = [case for case in _BUILTIN_CASES if case.task in allowed]
    promoted = [case for case in (custom_cases or []) if case.task in allowed]
    return [*builtin, *promoted]


def list_eval_cases_for_entry(
    hass: HomeAssistant,
    entry_id: str,
    *,
    tasks: list[str] | None = None,
) -> list[EvalCase]:
    """Return built-in and entry-specific promoted eval cases."""
    store = get_eval_store(hass, entry_id)
    custom_cases = store.list_custom_cases()
    return list_eval_cases(tasks=tasks, custom_cases=custom_cases)


def cases_for_task(task: str) -> list[EvalCase]:
    """Return built-in cases for one task route."""
    return [case for case in _BUILTIN_CASES if case.task == task]

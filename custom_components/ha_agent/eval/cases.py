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
        id="light_on",
        task="action",
        user_text="turn on the bedroom light",
        exposed_entities=[
            {
                "entity_id": "light.bedroom",
                "name": "Bedroom",
                "state": "off",
                "area_name": "Bedroom",
            }
        ],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.bedroom",
        },
        expected_text_contains=["on"],
        mock_mcp_responses=['{"success": true}'],
    ),
    EvalCase(
        id="climate_set_temp",
        task="action",
        user_text="set the living room to 21 degrees",
        exposed_entities=[
            {
                "entity_id": "climate.living",
                "name": "Living room",
                "state": "heat",
                "area_name": "Living room",
            }
        ],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "climate",
            "service": "set_temperature",
            "entity_id": "climate.living",
        },
        expected_text_contains=["21"],
        mock_mcp_responses=['{"success": true}'],
    ),
    EvalCase(
        id="media_pause",
        task="action",
        user_text="pause the kitchen speaker",
        exposed_entities=[
            {
                "entity_id": "media_player.kitchen",
                "name": "Kitchen speaker",
                "state": "playing",
                "area_name": "Kitchen",
            }
        ],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "media_player",
            "service": "media_pause",
            "entity_id": "media_player.kitchen",
        },
        expected_text_contains=["pause"],
        mock_mcp_responses=['{"success": true}'],
    ),
    EvalCase(
        id="scene_movie_night",
        task="action",
        user_text="activate the movie night scene",
        exposed_entities=[],
        expected_tool="home_assistant__ha_call_service",
        expected_tool_args={
            "domain": "scene",
            "service": "turn_on",
            "entity_id": "scene.movie_night",
        },
        expected_text_contains=["scene"],
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
        id="chat_greeting",
        task="chat",
        user_text="hello, how are you?",
        expected_tool=None,
        expected_text_contains=["hello"],
        mock_mcp_responses=[],
        max_iterations=4,
    ),
    EvalCase(
        id="chat_help",
        task="chat",
        user_text="what can you help me with around the house?",
        expected_tool=None,
        expected_text_contains=["help"],
        mock_mcp_responses=[],
        max_iterations=4,
    ),
    EvalCase(
        id="chat_entity_state",
        task="chat",
        user_text="is the dining room light on?",
        exposed_entities=[
            {
                "entity_id": "light.dining",
                "name": "Dining",
                "state": "on",
                "area_name": "Dining room",
            }
        ],
        expected_tool=None,
        expected_text_contains=["on"],
        mock_mcp_responses=[],
        max_iterations=4,
    ),
    EvalCase(
        id="chat_explain",
        task="chat",
        user_text="explain what a smart home assistant does in one sentence",
        expected_tool=None,
        expected_text_contains=["home"],
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

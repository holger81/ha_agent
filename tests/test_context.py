"""Unit tests for agent context building."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if name == "config_helpers":
        conv = types.ModuleType("homeassistant.components.conversation")
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = conv

    if name == "context":
        conv = types.ModuleType("homeassistant.components.conversation")
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = conv

    if name == "router":
        _load_module("config_helpers")
        _load_module("context")

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


context = _load_module("context")


def test_format_exposed_entities() -> None:
    """Exposed entities are formatted for the system prompt."""
    text = context.format_exposed_entities(
        [
            {
                "entity_id": "light.dining",
                "name": "Dining",
                "state": "on",
                "area_name": "Dining room",
            }
        ]
    )
    assert "light.dining" in text
    assert "Dining room" in text


def test_build_tool_context_adds_news_hint() -> None:
    """News queries get a direct news_curate tool hint."""
    tool_context = context.build_tool_context("What's the news?", [])
    assert "mcp_news__news_curate" in tool_context
    assert "callTool" in tool_context


def test_build_tool_context_adds_device_search_hint() -> None:
    """Device actions without exposed match get search hint."""
    tool_context = context.build_tool_context(
        "open the patio door",
        [{"entity_id": "light.kitchen", "name": "Kitchen"}],
    )
    assert "smart-home" in tool_context
    assert "searchToolsForDomain" in tool_context
    assert "ha_call_service" in tool_context


def test_build_tool_context_exposed_entities_are_shortcuts() -> None:
    """Exposed entities are labeled as shortcuts, not the full entity list."""
    tool_context = context.build_tool_context(
        "what is the kitchen temperature?",
        [{"entity_id": "sensor.kitchen_temp", "name": "Kitchen temperature"}],
    )
    assert "shortcuts" in tool_context.lower()
    assert "not a complete list" in tool_context.lower()
    assert "searchToolsForDomain" in tool_context
    assert "sensor.kitchen_temp" in tool_context


def test_build_tool_context_matched_device_allows_discovery() -> None:
    """Matched shortcuts still mention discovery when they may be insufficient."""
    tool_context = context.build_tool_context(
        "turn on the dining room lights",
        [
            {
                "entity_id": "light.dining_room_ceiling",
                "name": "Dining Room Ceiling Lights",
                "area_name": "Dining room",
            }
        ],
    )
    assert "shortcut" in tool_context.lower()
    assert "searchToolsForDomain" in tool_context


def test_route_playbook_device_mentions_discovery() -> None:
    """Device playbook prefers shortcuts but allows entity discovery."""
    router = _load_module("router")
    playbook = router.route_playbook(router.TaskRoute.HA_ACTION)
    assert "shortcut" in playbook.lower()
    assert "searchToolsForDomain" in playbook


def test_build_tool_context_adds_explicit_service_hint_for_match() -> None:
    """Matched exposed lights get domain/service/entity_id guidance."""
    tool_context = context.build_tool_context(
        "turn on the dining room lights",
        [
            {
                "entity_id": "light.dining_room_ceiling",
                "name": "Dining Room Ceiling Lights",
                "area_name": "Dining room",
            }
        ],
    )
    assert "light.dining_room_ceiling" in tool_context
    assert "domain light" in tool_context
    assert "service turn_on" in tool_context
    assert "ha_call_service" in tool_context
    assert "Do NOT call" in tool_context
    assert "ha_search_entities" in tool_context


def test_entity_matches_query() -> None:
    """Entity matching uses names and aliases."""
    entity = {
        "entity_id": "cover.patio",
        "name": "Patio door",
        "aliases": ["Jonathan patio"],
    }
    assert context.entity_matches_query(entity, "open Jonathan patio door")


def test_build_tool_context_adds_email_hint() -> None:
    """Email queries reference MCP discovery workflow."""
    tool_context = context.build_tool_context("do I have new emails", [])
    assert "domain email" in tool_context
    assert "searchToolsForDomain" in tool_context


def test_build_tool_context_adds_capability_hint() -> None:
    """Capability questions reference MCP session context."""
    tool_context = context.build_tool_context("what tools do you have access to?", [])
    assert "MCP SERVER INSTRUCTIONS" in tool_context


def test_build_system_message_includes_route_playbook() -> None:
    """Route playbooks are injected into the system message."""
    system_message = context.build_system_message(
        "You are helpful.",
        "Use tools.",
        route_playbook="EMAIL PLAYBOOK:\n1. Check inbox.",
    )
    assert "EMAIL PLAYBOOK" in system_message
    assert "You are helpful." in system_message
    assert "Use tools." in system_message


def test_build_system_message_includes_mcp_session_prompt() -> None:
    """System message includes MCP initialize instructions."""
    system_message = context.build_system_message(
        "You are helpful.",
        "Follow MCP instructions.",
        mcp_session_prompt=(
            "MCP SERVER INSTRUCTIONS:\nDiscover tools with searchToolsForDomain."
        ),
    )
    assert "MCP SERVER INSTRUCTIONS" in system_message
    assert "searchToolsForDomain" in system_message


def test_is_email_query() -> None:
    """Email intent detection works."""
    assert context.is_email_query("do I have new emails")
    assert not context.is_email_query("turn off the lights")


def test_is_email_query_keyword_override() -> None:
    """A keyword override replaces the default email matcher."""
    assert context.is_email_query("any postbox updates?", ["postbox"])
    assert not context.is_email_query("do I have new emails", ["postbox"])


def test_is_news_query_keyword_override() -> None:
    """A keyword override replaces the default news matcher."""
    assert context.is_news_query("what's the scoop?", ["scoop"])
    assert not context.is_news_query("what's the news?", ["scoop"])


def test_is_device_action_query_keyword_override() -> None:
    """A keyword override replaces the default device/camera matchers."""
    assert context.is_device_action_query("dim the lounge", ["dim"])
    assert not context.is_device_action_query("open the door", ["dim"])


def test_is_device_action_query_matches_camera_snapshot() -> None:
    """Camera snapshot requests count as homeassistant service actions."""
    assert context.is_device_action_query(
        "take a snapshot from my front door cam"
    )
    assert context.is_camera_action_query(
        "take a snapshot from my front door cam"
    )


def test_build_tool_context_camera_snapshot_suggests_entity_search() -> None:
    """Camera requests without a shortcut get search_entities + snapshot guidance."""
    tool_context = context.build_tool_context(
        "take a snapshot from my front door cam",
        [],
    )
    assert "ha_search_entities" in tool_context
    assert "camera" in tool_context
    assert "snapshot" in tool_context
    assert "front door" in tool_context.lower()


def test_build_tool_context_camera_match_uses_camera_entity() -> None:
    """Matched camera shortcuts get camera.snapshot guidance."""
    tool_context = context.build_tool_context(
        "take a snapshot from the front door cam",
        [
            {
                "entity_id": "camera.front_door",
                "name": "Front door cam",
                "area_name": "Front door",
            },
            {
                "entity_id": "light.porch",
                "name": "Porch light",
                "area_name": "Front door",
            },
        ],
    )
    assert "camera.front_door" in tool_context
    assert "service snapshot" in tool_context
    assert "domain camera" in tool_context


def test_is_device_action_query_matches_turn_them_back_off() -> None:
    """Pronoun phrases between turn and off still count as device actions."""
    assert context.is_device_action_query("turn them back off")


def test_build_tool_context_turn_them_back_off_uses_turn_off() -> None:
    """Follow-up off phrasing suggests turn_off, not turn_on."""
    history = [
        {
            "role": "assistant",
            "content": "Controlled: light.dining_room_ceiling.",
        }
    ]
    tool_context = context.build_tool_context(
        "turn them back off",
        [
            {
                "entity_id": "light.dining_room_ceiling",
                "name": "Dining Room Ceiling Lights",
                "area_name": "Dining room",
            }
        ],
        history=history,
    )
    assert "service turn_off" in tool_context


def test_build_tool_context_follow_up_hint_from_history() -> None:
    """Retry/pronoun follow-ups reuse entity ids from prior turns."""
    history = [
        {
            "role": "user",
            "content": "turn on the dining room lights",
        },
        {
            "role": "assistant",
            "content": (
                "The dining room lights have been turned on. "
                "Controlled: light.dining_room_ceiling."
            ),
        },
    ]
    tool_context = context.build_tool_context(
        "they are. try again",
        [],
        history=history,
    )
    assert "FOLLOW-UP DEVICE ACTION" in tool_context
    assert "light.dining_room_ceiling" in tool_context


def test_build_tool_context_turn_them_back_off_reuses_history_entity() -> None:
    """Pronoun off commands reuse the entity id from the previous turn."""
    history = [
        {"role": "user", "content": "turn on the dining room lights"},
        {
            "role": "assistant",
            "content": (
                "The dining room lights have been turned on. "
                "Controlled: light.dining_room_ceiling."
            ),
        },
    ]
    tool_context = context.build_tool_context(
        "turn them back off",
        [
            {
                "entity_id": "light.dining_room_ceiling",
                "name": "Dining Room Ceiling Lights",
            }
        ],
        history=history,
    )
    assert "service turn_off" in tool_context
    assert "light.dining_room_ceiling" in tool_context

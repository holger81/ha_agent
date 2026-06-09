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

    if name == "context":
        conv = types.ModuleType("homeassistant.components.conversation")
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = conv

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
    """News queries reference MCP discovery workflow."""
    tool_context = context.build_tool_context("What's the news?", [])
    assert "domain news" in tool_context
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

"""Unit tests for MCP tool execution."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_stubs() -> None:
    if "homeassistant.exceptions" not in sys.modules:
        ha_exc = types.ModuleType("homeassistant.exceptions")

        class HomeAssistantError(Exception):
            pass

        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant.exceptions"] = ha_exc


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    _ensure_ha_stubs()

    if name == "tools":
        _load_module("llm_client")

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tools = _load_module("tools")
llm_client = _load_module("llm_client")


def test_normalize_legacy_mcp_call_tool() -> None:
    """Legacy mcp_call_tool maps to MCP callTool."""
    call = llm_client.ToolCall(
        id="call_1",
        name="mcp_call_tool",
        arguments=json.dumps(
            {
                "toolName": "mail_mcp__imap_search_messages",
                "arguments": {"mailbox": "INBOX"},
            }
        ),
    )

    tool_name, tool_args = tools._normalize_tool_call(call)

    assert tool_name == "callTool"
    assert tool_args["toolName"] == "mail_mcp__imap_search_messages"
    assert tool_args["arguments"]["mailbox"] == "INBOX"


def test_normalize_session_tool_call() -> None:
    """Session tools are passed through to MCP tools/call."""
    call = llm_client.ToolCall(
        id="call_2",
        name="searchToolsForDomain",
        arguments=json.dumps({"domain": "email", "query": "imap"}),
    )

    tool_name, tool_args = tools._normalize_tool_call(call)

    assert tool_name == "searchToolsForDomain"
    assert tool_args["domain"] == "email"


def test_normalize_flat_ha_call_service_args() -> None:
    """Flat callTool payloads keep service fields instead of dropping them."""
    call = llm_client.ToolCall(
        id="call_3",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "home_assistant__ha_call_service",
                "entity_id": "light.dining",
                "service": "turn_on",
            }
        ),
    )

    tool_name, tool_args = tools._normalize_tool_call(call)

    assert tool_name == "callTool"
    assert tool_args["toolName"] == "home_assistant__ha_call_service"
    assert tool_args["arguments"]["entity_id"] == "light.dining"
    assert tool_args["arguments"]["service"] == "turn_on"
    assert tool_args["arguments"]["domain"] == "light"


def test_normalize_ha_call_service_service_aliases() -> None:
    """Common LLM service spellings map to homeassistant service ids."""
    call = llm_client.ToolCall(
        id="call_4b",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "entity_id": "light.dining",
                    "service": "turn off",
                },
            }
        ),
    )

    _tool_name, tool_args = tools._normalize_tool_call(call)

    assert tool_args["arguments"]["service"] == "turn_off"
    assert tool_args["arguments"]["domain"] == "light"


def test_normalize_ha_call_service_infers_domain() -> None:
    """ha_call_service calls missing domain are repaired from entity_id."""
    call = llm_client.ToolCall(
        id="call_4",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "entity_id": "light.dining_room_ceiling",
                    "service": "turn_on",
                },
            }
        ),
    )

    _, tool_args = tools._normalize_tool_call(call)

    assert tool_args["arguments"]["domain"] == "light"


def test_normalize_ha_call_service_resolves_display_name() -> None:
    """Display names from the model are mapped to real entity ids."""
    call = llm_client.ToolCall(
        id="call_5",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "entity_id": "Dining Room Ceiling Lights",
                    "service": "turn_off",
                },
            }
        ),
    )
    exposed = [
        {
            "entity_id": "light.dining_room_ceiling",
            "name": "Dining Room Ceiling Lights",
        }
    ]

    _, tool_args = tools._normalize_tool_call(call, exposed_entities=exposed)

    assert tool_args["arguments"]["entity_id"] == "light.dining_room_ceiling"
    assert tool_args["arguments"]["domain"] == "light"


def test_normalize_doubled_mcp_tool_prefix() -> None:
    """Duplicate MCP server prefixes are collapsed before callTool."""
    call = llm_client.ToolCall(
        id="call_news",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "mcp-news__mcp_news__news_curate",
                "arguments": {"limit": 5},
            }
        ),
    )

    tool_name, tool_args = tools._normalize_tool_call(call)

    assert tool_name == "callTool"
    assert tool_args["toolName"] == "mcp_news__news_curate"
    assert tool_args["arguments"]["limit"] == 5


def test_memory_assistant_text_appends_controlled_entities() -> None:
    """Conversation memory keeps entity ids for pronoun follow-ups."""
    text = tools.memory_assistant_text(
        "The dining room lights have been turned on.",
        ["light.dining_room_ceiling"],
    )
    assert "light.dining_room_ceiling" in text

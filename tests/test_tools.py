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

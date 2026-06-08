"""Tool execution for the agent loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError

from .llm_client import ToolCall

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse tool call arguments JSON."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_tool_call(call: ToolCall) -> tuple[str, dict[str, Any]]:
    """Map LLM tool calls to MCP tools/call name and arguments."""
    args = parse_tool_arguments(call.arguments)
    tool_name = call.name

    if tool_name == "mcp_call_tool":
        tool_name = "callTool"

    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a flat object")

    if tool_name == "callTool" and "toolName" in args and "arguments" not in args:
        return tool_name, {
            "toolName": args["toolName"],
            "arguments": {},
        }

    return tool_name, args


async def execute_tool(
    mcp_client: McpProxyClient,
    call: ToolCall,
) -> str:
    """Execute a single LLM tool call via MCP tools/call."""
    try:
        tool_name, tool_args = _normalize_tool_call(call)
    except ValueError as err:
        return str(err)

    try:
        result = await mcp_client.call_tool(tool_name, tool_args)
    except HomeAssistantError as err:
        return f"Tool error: {err}"
    except Exception as err:
        return f"Tool error: {err}"

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def tool_result_message(call: ToolCall, output: str) -> dict[str, str]:
    """Build an OpenAI tool result message."""
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": output,
    }

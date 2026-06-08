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


async def execute_tool(
    mcp_client: McpProxyClient,
    call: ToolCall,
) -> str:
    """Execute a single LLM tool call."""
    if call.name != "mcp_call_tool":
        return f"Unsupported tool: {call.name}"

    args = parse_tool_arguments(call.arguments)
    tool_name = args.get("toolName")
    if not tool_name:
        return "Missing toolName in mcp_call_tool arguments"

    tool_args = args.get("arguments")
    if tool_args is None:
        tool_args = {}
    if not isinstance(tool_args, dict):
        return "arguments must be a flat object"

    try:
        result = await mcp_client.call_tool(str(tool_name), tool_args)
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

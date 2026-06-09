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


def _domain_from_entity_id(entity_id: Any) -> str | None:
    """Infer a Home Assistant domain from an entity id."""
    if isinstance(entity_id, str) and "." in entity_id:
        return entity_id.split(".", 1)[0]
    if isinstance(entity_id, list) and entity_id:
        first = entity_id[0]
        if isinstance(first, str) and "." in first:
            return first.split(".", 1)[0]
    return None


def _normalize_ha_call_service_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fill missing ha_call_service fields the local LLM often omits."""
    normalized = dict(arguments)
    if not normalized.get("domain") and (
        domain := _domain_from_entity_id(normalized.get("entity_id"))
    ):
        normalized["domain"] = domain
    return normalized


def _normalize_call_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize callTool payloads before sending them to MCP."""
    upstream = payload.get("toolName")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    if isinstance(upstream, str) and upstream.endswith("ha_call_service"):
        arguments = _normalize_ha_call_service_arguments(arguments)

    return {"toolName": upstream, "arguments": arguments}


def _normalize_tool_call(call: ToolCall) -> tuple[str, dict[str, Any]]:
    """Map LLM tool calls to MCP tools/call name and arguments."""
    args = parse_tool_arguments(call.arguments)
    tool_name = call.name

    if tool_name == "mcp_call_tool":
        tool_name = "callTool"

    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a flat object")

    if tool_name == "callTool" and "toolName" in args:
        upstream = args["toolName"]
        if isinstance(args.get("arguments"), dict):
            nested = dict(args["arguments"])
        else:
            nested = {
                key: value
                for key, value in args.items()
                if key != "toolName"
            }
        return tool_name, _normalize_call_tool_payload(
            {"toolName": upstream, "arguments": nested}
        )

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

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


def _looks_like_entity_id(value: str) -> bool:
    """Return True when text looks like a homeassistant entity id."""
    return "." in value and " " not in value


def _resolve_entity_id(
    value: Any,
    exposed_entities: list[dict[str, Any]] | None,
) -> Any:
    """Map display names to entity_id values the MCP service accepts."""
    if isinstance(value, list):
        resolved = [
            _resolve_entity_id(item, exposed_entities)
            for item in value
            if item is not None
        ]
        return resolved[0] if len(resolved) == 1 else resolved

    if not isinstance(value, str) or not value.strip():
        return value

    candidate = value.strip()
    if _looks_like_entity_id(candidate):
        return candidate

    if not exposed_entities:
        return candidate

    lowered = candidate.lower()
    for entity in exposed_entities:
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str):
            continue
        if lowered == entity_id.lower():
            return entity_id
        name = entity.get("name")
        if isinstance(name, str) and lowered == name.lower():
            return entity_id
        area = entity.get("area_name")
        if isinstance(area, str) and lowered == area.lower():
            return entity_id
        aliases = entity.get("aliases")
        if isinstance(aliases, list) and any(
            isinstance(alias, str) and lowered == alias.lower() for alias in aliases
        ):
            return entity_id
    return candidate


def _domain_from_entity_id(entity_id: Any) -> str | None:
    """Infer a Home Assistant domain from an entity id."""
    if isinstance(entity_id, str) and "." in entity_id:
        return entity_id.split(".", 1)[0]
    if isinstance(entity_id, list) and entity_id:
        first = entity_id[0]
        if isinstance(first, str) and "." in first:
            return first.split(".", 1)[0]
    return None


def _normalize_service_name(service: Any) -> Any:
    """Map common LLM service spellings to homeassistant service ids."""
    if not isinstance(service, str):
        return service
    key = service.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "turn on": "turn_on",
        "turn off": "turn_off",
        "open cover": "open_cover",
        "close cover": "close_cover",
        "open": "open_cover",
        "close": "close_cover",
    }
    if key in aliases:
        return aliases[key]
    if " " not in key:
        return service.strip()
    return key.replace(" ", "_")


def _normalize_ha_call_service_arguments(
    arguments: dict[str, Any],
    *,
    exposed_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fill missing ha_call_service fields the local LLM often omits."""
    normalized = dict(arguments)
    if "service" in normalized:
        normalized["service"] = _normalize_service_name(normalized["service"])
    if "entity_id" in normalized:
        normalized["entity_id"] = _resolve_entity_id(
            normalized["entity_id"],
            exposed_entities,
        )
    if not normalized.get("domain") and (
        domain := _domain_from_entity_id(normalized.get("entity_id"))
    ):
        normalized["domain"] = domain
    return normalized


def _normalize_call_tool_payload(
    payload: dict[str, Any],
    *,
    exposed_entities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize callTool payloads before sending them to MCP."""
    upstream = payload.get("toolName")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}

    if isinstance(upstream, str) and upstream.endswith("ha_call_service"):
        arguments = _normalize_ha_call_service_arguments(
            arguments,
            exposed_entities=exposed_entities,
        )

    return {"toolName": upstream, "arguments": arguments}


def _normalize_tool_call(
    call: ToolCall,
    *,
    exposed_entities: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
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
            {"toolName": upstream, "arguments": nested},
            exposed_entities=exposed_entities,
        )

    return tool_name, args


def ha_service_entity_id(
    call: ToolCall,
    *,
    exposed_entities: list[dict[str, Any]] | None = None,
) -> str | None:
    """Return the entity_id from a normalized ha_call_service tool call."""
    try:
        tool_name, tool_args = _normalize_tool_call(
            call,
            exposed_entities=exposed_entities,
        )
    except ValueError:
        return None
    if tool_name != "callTool":
        return None
    upstream = tool_args.get("toolName")
    if not isinstance(upstream, str) or not upstream.endswith("ha_call_service"):
        return None
    arguments = tool_args.get("arguments")
    if not isinstance(arguments, dict):
        return None
    entity_id = arguments.get("entity_id")
    if isinstance(entity_id, str) and _looks_like_entity_id(entity_id):
        return entity_id
    return None


def memory_assistant_text(text: str, entity_ids: list[str]) -> str:
    """Append controlled entity ids for follow-up turns in conversation memory."""
    cleaned = text.strip()
    unique_ids = list(dict.fromkeys(entity_ids))
    if not unique_ids:
        return cleaned
    suffix = " Controlled: " + ", ".join(unique_ids) + "."
    if not cleaned:
        return suffix.strip()
    return cleaned + suffix


async def execute_tool(
    mcp_client: McpProxyClient,
    call: ToolCall,
    *,
    exposed_entities: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a single LLM tool call via MCP tools/call."""
    try:
        tool_name, tool_args = _normalize_tool_call(
            call,
            exposed_entities=exposed_entities,
        )
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

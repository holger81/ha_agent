"""Tool execution for the agent loop."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError

from .llm_client import ToolCall

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

_DISCOVERY_TOOL = re.compile(
    r"(searchToolsForDomain|searchTool|tools/list|tools_list)",
    re.IGNORECASE,
)
_TOOL_OUTPUT_MAX_CHARS = 10_000
_DISCOVERY_OUTPUT_MAX_CHARS = 6_000
_DISCOVERY_MAX_TOOLS = 40


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
        "snapshot": "snapshot",
        "take snapshot": "snapshot",
        "take photo": "snapshot",
        "take picture": "snapshot",
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


def _normalize_upstream_tool_name(name: str) -> str:
    """Fix common LLM mistakes in MCP proxy tool names."""
    cleaned = name.strip()
    if not cleaned:
        return cleaned

    if "__" not in cleaned:
        return cleaned.replace("-", "_")

    parts = [part.replace("-", "_") for part in cleaned.split("__") if part]
    while len(parts) >= 3 and parts[0] == parts[1]:
        parts = [parts[0], *parts[2:]]
    return "__".join(parts)


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

    if isinstance(upstream, str):
        upstream = _normalize_upstream_tool_name(upstream)

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
        if isinstance(upstream, str):
            upstream = _normalize_upstream_tool_name(upstream)
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

    if tool_name not in {"callTool", "searchTool", "searchToolsForDomain"}:
        tool_name = _normalize_upstream_tool_name(tool_name)

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


def is_discovery_tool_name(tool_name: str) -> bool:
    """Return True for MCP discovery/list tools."""
    return bool(_DISCOVERY_TOOL.search(tool_name or ""))


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return (
        f"{text[: max_chars - 80]}\n\n"
        f"[truncated — {len(text)} chars total; use narrower tool arguments]"
    )


def _tool_entry_name(entry: dict[str, Any]) -> str:
    for key in ("toolName", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _compact_tool_entry(entry: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    if name := _tool_entry_name(entry):
        compact["toolName"] = name
    description = entry.get("description")
    if isinstance(description, str) and description.strip():
        compact["description"] = description.strip()[:240]
    context = entry.get("serverLlmContext")
    if isinstance(context, str) and context.strip():
        compact["serverLlmContext"] = context.strip()[:400]
    domain = entry.get("domain")
    if isinstance(domain, str) and domain.strip():
        compact["domain"] = domain.strip()
    return compact


def _parse_discovery_payload(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)], meta
    if not isinstance(data, dict):
        return [], meta

    for key in ("mode", "domain", "hasMore", "offset", "limit", "total"):
        if key in data:
            meta[key] = data[key]

    for key in ("tools", "results", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)], meta

    if _tool_entry_name(data):
        return [data], meta
    return [], meta


def compact_discovery_tool_output(
    output: str,
    *,
    max_tools: int = _DISCOVERY_MAX_TOOLS,
    max_chars: int = _DISCOVERY_OUTPUT_MAX_CHARS,
) -> str:
    """Shrink discovery tool JSON so tool schemas do not blow the LLM context."""
    if output.startswith("Tool error:"):
        return output
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return _truncate_text(output, max_chars=max_chars)

    entries, meta = _parse_discovery_payload(data)
    if not entries:
        return _truncate_text(output, max_chars=max_chars)

    shown = max(1, max_tools)
    while shown > 0:
        compacted = [
            item
            for item in (_compact_tool_entry(entry) for entry in entries[:shown])
            if item
        ]
        result: dict[str, Any] = dict(meta)
        if compacted:
            result["tools"] = compacted
        if len(entries) > shown:
            result["truncated"] = True
            result["shown"] = shown
            result["total"] = meta.get("total", len(entries))
            result["note"] = (
                "Tool list shortened for context. Pick the best toolName and "
                "call callTool; use searchTool with a narrower query if needed."
            )
        text = json.dumps(result, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        shown = max(0, shown // 2)

    return _truncate_text(output, max_chars=max_chars)


def compact_tool_output(tool_name: str, output: str) -> str:
    """Bound tool result size before it is appended to the LLM conversation."""
    if output.startswith("Tool error:"):
        return output
    if is_discovery_tool_name(tool_name):
        return compact_discovery_tool_output(output)
    return _truncate_text(output, max_chars=_TOOL_OUTPUT_MAX_CHARS)

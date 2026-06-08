"""Parse tool calls embedded in model text (Gemma / LFM / llama.cpp templates)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_BLOCK = re.compile(
    r"<\|tool_call\|>(.*?)<?/?tool_call\|>",
    re.DOTALL | re.IGNORECASE,
)
_DIRECT_CALL = re.compile(
    r"^call:(?P<name>[a-zA-Z0-9_]+)\s*\{\s*arguments:\s*(?P<args>\{.*\})\s*\}\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedToolCall:
    """Tool call parsed from embedded model text."""

    id: str
    name: str
    arguments: str


def _parse_js_like_object(raw: str) -> dict[str, Any]:
    """Parse a loosely formatted JSON/JS object from model output."""
    text = raw.strip()
    if not text.startswith("{"):
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)
    try:
        parsed = json.loads(fixed)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _direct_mcp_tool_call(tool_name: str, arguments: dict[str, Any]) -> ParsedToolCall:
    """Build a direct MCP session tool call."""
    return ParsedToolCall(
        id="call_embedded",
        name=tool_name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )


def _legacy_call_tool(tool_name: str, arguments: dict[str, Any]) -> ParsedToolCall:
    """Map a legacy upstream tool invocation to MCP callTool."""
    return ParsedToolCall(
        id="call_embedded",
        name="callTool",
        arguments=json.dumps(
            {"toolName": tool_name, "arguments": arguments},
            ensure_ascii=False,
        ),
    )


def _parse_tool_call_block(block: str, *, call_id: str) -> ParsedToolCall | None:
    """Parse one <|tool_call|>...<|tool_call|> block."""
    text = block.strip()
    if not text:
        return None

    if direct := _DIRECT_CALL.match(text):
        tool_name = direct.group("name")
        args = _parse_js_like_object(direct.group("args"))
        if tool_name in {"callTool", "searchTool", "searchToolsForDomain"}:
            call = _direct_mcp_tool_call(tool_name, args)
        elif "__" in tool_name:
            call = _legacy_call_tool(tool_name, args)
        else:
            call = _direct_mcp_tool_call(tool_name, args)
        return ParsedToolCall(id=call_id, name=call.name, arguments=call.arguments)

    if text.startswith("{"):
        payload = _parse_js_like_object(text)
        if not payload:
            return None
        name = payload.get("name") or payload.get("toolName")
        if not name:
            return None
        if name in {"mcp_call_tool", "callTool"}:
            arguments = payload.get("arguments") or payload
            if isinstance(arguments, dict):
                return ParsedToolCall(
                    id=call_id,
                    name="callTool",
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
        if isinstance(name, str) and "__" in name:
            args = payload.get("arguments")
            if not isinstance(args, dict):
                args = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"name", "toolName", "arguments"}
                }
            call = _legacy_call_tool(name, args if isinstance(args, dict) else {})
            return ParsedToolCall(id=call_id, name=call.name, arguments=call.arguments)
        return ParsedToolCall(
            id=call_id,
            name=str(name),
            arguments=json.dumps(
                payload.get("arguments") or payload,
                ensure_ascii=False,
            ),
        )

    return None


def parse_embedded_tool_calls(content: str | None) -> list[ParsedToolCall]:
    """Extract tool calls written as text instead of API tool_calls."""
    if not content or "<|tool_call|>" not in content.lower():
        return []

    calls: list[ParsedToolCall] = []
    for index, match in enumerate(_TOOL_CALL_BLOCK.finditer(content)):
        call = _parse_tool_call_block(match.group(1), call_id=f"call_embedded_{index}")
        if call:
            calls.append(call)
    return calls


def strip_embedded_tool_markup(content: str | None) -> str:
    """Remove embedded tool-call markup from assistant text."""
    if not content:
        return ""
    return _TOOL_CALL_BLOCK.sub("", content).strip()


def is_tool_call_only_text(content: str | None) -> bool:
    """Return True when content is only embedded tool-call markup."""
    if not content:
        return False
    return bool(parse_embedded_tool_calls(content)) and not strip_embedded_tool_markup(
        content
    )

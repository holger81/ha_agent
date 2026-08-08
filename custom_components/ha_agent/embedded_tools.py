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
_COMPACT_CALL = re.compile(
    r"^call:(?P<name>[a-zA-Z0-9_]+)\{(?P<body>.*)\}\s*$",
    re.DOTALL | re.IGNORECASE,
)
_GEMMA_STRING_QUOTE = re.compile(r'<\|"\|>')
# Small models often emit: [searchToolsForDomain domain="smart-home", query="..."]
# or [home_assistant__ha_call_service service="turn_off", entity_id="light.x"]
_BRACKET_TOOL_NAMES = (
    r"searchToolsForDomain|searchTool|callTool|"
    r"[a-z][a-z0-9_]*(?:__[a-z0-9_]+)+"
)
_BRACKET_TOOL_CALL = re.compile(
    rf"\[(?P<name>{_BRACKET_TOOL_NAMES})(?P<body>[^\]]*)\]",
    re.IGNORECASE,
)
_BRACKET_KV = re.compile(
    r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?:
        "(?P<dq>[^"]*)"
        |'(?P<sq>[^']*)'
        |(?P<bare>[^\s,\]]+)
    )
    """,
    re.VERBOSE,
)
_TRAILING_TOOL_JUNK = re.compile(
    r"(?:\]+)?(?:\s*<\|/?tool_call(?:_end)?\|>)*\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedToolCall:
    """Tool call parsed from embedded model text."""

    id: str
    name: str
    arguments: str


def _normalize_gemma_tokens(text: str) -> str:
    """Normalize Gemma-specific string tokens in tool-call text."""
    return _GEMMA_STRING_QUOTE.sub('"', text)


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


def _parse_compact_call_body(body: str) -> dict[str, Any]:
    """Parse call:tool{field:value,...} bodies from Gemma templates."""
    normalized = _normalize_gemma_tokens(body.strip())
    if normalized.startswith("arguments:"):
        return _parse_js_like_object(normalized[len("arguments:") :].strip())
    if not normalized.startswith("{"):
        normalized = f"{{{normalized}}}"
    return _parse_js_like_object(normalized)


def _parse_bracket_kv_body(body: str) -> dict[str, Any]:
    """Parse key="value" pairs from bracket-style tool markup."""
    cleaned = _TRAILING_TOOL_JUNK.sub("", body.strip())
    args: dict[str, Any] = {}
    for match in _BRACKET_KV.finditer(cleaned):
        key = match.group("key")
        value = match.group("dq")
        if value is None:
            value = match.group("sq")
        if value is None:
            value = (match.group("bare") or "").strip().rstrip("]")
        if key:
            args[key] = value
    return args


def _tool_call_from_name_and_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    call_id: str,
) -> ParsedToolCall:
    """Build a parsed tool call for session or upstream MCP tools."""
    if tool_name in {"callTool", "searchTool", "searchToolsForDomain"}:
        call = _direct_mcp_tool_call(tool_name, args)
    elif "__" in tool_name:
        call = _legacy_call_tool(tool_name, args)
    else:
        call = _direct_mcp_tool_call(tool_name, args)
    return ParsedToolCall(id=call_id, name=call.name, arguments=call.arguments)


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
        return _tool_call_from_name_and_args(tool_name, args, call_id=call_id)

    if compact := _COMPACT_CALL.match(text):
        tool_name = compact.group("name")
        args = _parse_compact_call_body(compact.group("body"))
        if args:
            return _tool_call_from_name_and_args(tool_name, args, call_id=call_id)

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
    if not content:
        return []

    calls: list[ParsedToolCall] = []
    lowered = content.lower()
    if "<|tool_call|>" in lowered:
        for index, match in enumerate(_TOOL_CALL_BLOCK.finditer(content)):
            call = _parse_tool_call_block(
                match.group(1), call_id=f"call_embedded_{index}"
            )
            if call:
                calls.append(call)

    # Prefer classic markup when both forms appear; otherwise recover LFM
    # bracket-style invocations that never used <|tool_call|> wrappers.
    if calls:
        return calls

    for index, match in enumerate(_BRACKET_TOOL_CALL.finditer(content)):
        name = match.group("name")
        args = _parse_bracket_kv_body(match.group("body") or "")
        if not name:
            continue
        # Require at least one argument for bare session tools to avoid
        # treating prose like "[searchTool]" mentions as calls.
        if not args and name.lower() in {
            "searchtoolsfordomain",
            "searchtool",
            "calltool",
        }:
            continue
        calls.append(
            _tool_call_from_name_and_args(
                name,
                args,
                call_id=f"call_bracket_{index}",
            )
        )
    return calls


def strip_embedded_tool_markup(content: str | None) -> str:
    """Remove embedded tool-call markup from assistant text."""
    if not content:
        return ""
    text = _TOOL_CALL_BLOCK.sub("", content)
    text = _BRACKET_TOOL_CALL.sub("", text)
    text = re.sub(r"<\|/?tool_call(?:_end)?\|>", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_tool_call_only_text(content: str | None) -> bool:
    """Return True when content is only embedded tool-call markup."""
    if not content:
        return False
    return bool(parse_embedded_tool_calls(content)) and not strip_embedded_tool_markup(
        content
    )


_TOOL_CALL_MARKER = "<|tool_call|>"


def safe_stream_display_text(text: str) -> str:
    """Return assistant text safe to show while a stream is still in progress."""
    if not text:
        return ""

    holdback = len(text)
    marker = _TOOL_CALL_MARKER
    for prefix_len in range(1, len(marker)):
        suffix = text[-prefix_len:]
        if suffix.startswith("<") and marker.lower().startswith(suffix.lower()):
            holdback = len(text) - prefix_len
            break

    candidate = text[:holdback]
    tool_start = candidate.lower().find(_TOOL_CALL_MARKER.lower())
    if tool_start != -1:
        candidate = candidate[:tool_start]

    # Hold back a trailing open bracket tool call while streaming.
    open_bracket = candidate.rfind("[")
    if open_bracket != -1 and "]" not in candidate[open_bracket:]:
        maybe = candidate[open_bracket + 1 :]
        if re.match(rf"(?:{_BRACKET_TOOL_NAMES})\b", maybe, re.IGNORECASE):
            candidate = candidate[:open_bracket]

    return strip_embedded_tool_markup(candidate)

"""Helpers for MCP session state (initialize + tools/list)."""

from __future__ import annotations

from typing import Any

FALLBACK_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "searchToolsForDomain",
        "description": (
            "Discover tools in one MCP proxy domain. Use before callTool when "
            "the needed upstream tool is not listed at session level."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "query": {"type": "string"},
                "listAll": {"type": "boolean"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["domain"],
        },
    },
    {
        "name": "searchTool",
        "description": "Discover MCP tools by keyword across domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "domain": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "callTool",
        "description": (
            "Execute an upstream MCP tool using toolName from discovery."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "toolName": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["toolName"],
        },
    },
]


def server_info_from_init(init_result: dict[str, Any]) -> dict[str, Any]:
    """Return server metadata from an initialize result."""
    server = init_result.get("serverInfo") or init_result.get("server")
    return server if isinstance(server, dict) else {}


def server_info_summary(init_result: dict[str, Any]) -> str:
    """Format MCP server metadata for the system prompt."""
    server = server_info_from_init(init_result)
    name = str(server.get("name") or "MCP server")
    version = server.get("version")
    if version:
        return f"{name} v{version}"
    return name


def format_mcp_session_prompt(
    *,
    instructions: str,
    init_result: dict[str, Any],
    session_tools: list[dict[str, Any]],
) -> str:
    """Build MCP protocol context for the LLM system prompt."""
    parts: list[str] = []

    summary = server_info_summary(init_result)
    if summary:
        parts.append(f"MCP SERVER: {summary}")

    if instructions.strip():
        parts.append(f"MCP SERVER INSTRUCTIONS:\n{instructions.strip()}")

    if session_tools:
        lines = [
            f"- {tool['name']}: {(tool.get('description') or '').strip()}"
            for tool in session_tools
            if tool.get("name")
        ]
        if lines:
            parts.append(
                "MCP SESSION TOOLS (also available as function tools):\n"
                + "\n".join(lines)
            )

    return "\n\n".join(parts)


def mcp_tool_to_openai_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one MCP tools/list entry to an OpenAI function schema."""
    name = tool.get("name")
    if not name:
        raise ValueError("MCP tool is missing name")

    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(tool.get("description") or "").strip(),
            "parameters": schema,
        },
    }


def mcp_tools_to_openai_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tools/list entries to OpenAI-compatible tool schemas."""
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if not tool.get("name"):
            continue
        schemas.append(mcp_tool_to_openai_schema(tool))
    return schemas

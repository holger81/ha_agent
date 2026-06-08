"""Unit tests for MCP session helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mcp_session = _load_module("mcp_session")


def test_format_mcp_session_prompt_includes_instructions() -> None:
    """Initialize instructions are injected into the system prompt."""
    prompt = mcp_session.format_mcp_session_prompt(
        instructions="Discover tools with searchToolsForDomain.",
        init_result={
            "serverInfo": {"name": "mcp-proxy", "version": "0.1.0"},
        },
        session_tools=[
            {
                "name": "callTool",
                "description": "Execute upstream MCP tools.",
            }
        ],
    )

    assert "MCP SERVER: mcp-proxy v0.1.0" in prompt
    assert "MCP SERVER INSTRUCTIONS" in prompt
    assert "searchToolsForDomain" in prompt
    assert "MCP SESSION TOOLS" in prompt
    assert "callTool" in prompt


def test_mcp_tools_to_openai_schemas() -> None:
    """MCP tools/list entries convert to OpenAI function schemas."""
    schemas = mcp_session.mcp_tools_to_openai_schemas(
        [
            {
                "name": "searchTool",
                "description": "Search tools",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )

    assert schemas[0]["function"]["name"] == "searchTool"
    assert schemas[0]["function"]["parameters"]["required"] == ["query"]

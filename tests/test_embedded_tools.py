"""Unit tests for embedded tool-call parsing."""

from __future__ import annotations

import importlib.util
import json
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


embedded_tools = _load_module("embedded_tools")


def test_parse_compact_gemma_tool_call_from_screenshot() -> None:
    """Gemma compact call:TOOL{query:<|"|>...} blocks are parsed."""
    content = (
        '<|tool_call|>call:home_assistant__ha_search_entities'
        '{query:<|"|>camera snapshot<|"|>}<tool_call|>'
    )
    calls = embedded_tools.parse_embedded_tool_calls(content)

    assert len(calls) == 1
    assert calls[0].name == "callTool"
    args = json.loads(calls[0].arguments)
    assert args["toolName"] == "home_assistant__ha_search_entities"
    assert args["arguments"]["query"] == "camera snapshot"


def test_parse_direct_mcp_tool_call_from_screenshot() -> None:
    """Gemma-style call:TOOL{arguments:{...}} blocks are parsed."""
    content = (
        '<|tool_call|>call:home_assistant__ha_search_entities'
        '{arguments: {domain_filter:"sensor",query:"email"}}<tool_call|>'
    )
    calls = embedded_tools.parse_embedded_tool_calls(content)

    assert len(calls) == 1
    assert calls[0].name == "callTool"
    args = json.loads(calls[0].arguments)
    assert args["toolName"] == "home_assistant__ha_search_entities"
    assert args["arguments"]["query"] == "email"


def test_strip_embedded_tool_markup() -> None:
    """Tool-call markup is removed from assistant text."""
    content = "Hello <|tool_call|>call:test{arguments:{}}<tool_call|> world"
    assert embedded_tools.strip_embedded_tool_markup(content) == "Hello  world"


def test_is_tool_call_only_text() -> None:
    """Detect responses that contain only tool-call markup."""
    content = '<|tool_call|>call:test{arguments:{}}<tool_call|>'
    assert embedded_tools.is_tool_call_only_text(content)
    assert not embedded_tools.is_tool_call_only_text("Here you go.")

"""Tests for loop tool pruning."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    deps = {"tool_pruning": ["mcp_session"], "mcp_session": []}
    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)
    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tool_pruning = _load("tool_pruning")


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_prune_keeps_discovery_and_preferred_tools() -> None:
    full = [
        _tool("mcp_a__searchToolsForDomain"),
        _tool("mcp_a__searchTool"),
        _tool("mcp_a__callTool"),
        _tool("mcp_news__news_curate"),
        _tool("mcp_mail__imap_search"),
        _tool("home_assistant__ha_call_service"),
    ]
    pruned = tool_pruning.prune_loop_tools(
        full,
        preferred_names=["mcp_news__news_curate"],
        max_tools=4,
    )
    names = [tool_pruning._tool_schema_name(tool) for tool in pruned]
    assert "mcp_a__searchToolsForDomain" in names
    assert "mcp_news__news_curate" in names
    assert len(pruned) <= 4


def test_prune_returns_full_catalog_when_requested() -> None:
    full = [_tool(f"tool_{index}") for index in range(12)]
    assert tool_pruning.prune_loop_tools(
        full,
        max_tools=4,
        include_full_catalog=True,
    ) == full

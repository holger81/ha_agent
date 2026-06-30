"""Prune MCP tool schemas sent to the loop LLM."""

from __future__ import annotations

from typing import Any

from .mcp_session import FALLBACK_MCP_TOOLS

_DISCOVERY_NAMES = frozenset(
    tool["name"].lower() for tool in FALLBACK_MCP_TOOLS if tool.get("name")
)


def _tool_schema_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            return name
    name = tool.get("name")
    return str(name or "")


def _names_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tail = left.split("__")[-1]
    right_tail = right.split("__")[-1]
    return left_tail == right_tail or left.endswith(right) or right.endswith(left)


def _is_discovery_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in _DISCOVERY_NAMES:
        return True
    return lowered.endswith("__searchtoolsfordomain") or lowered.endswith(
        "__searchtool"
    )


def prune_loop_tools(
    full_tools: list[dict[str, Any]],
    *,
    preferred_names: list[str] | None = None,
    max_tools: int,
    include_full_catalog: bool = False,
) -> list[dict[str, Any]]:
    """Return a smaller tool list for one loop iteration."""
    if include_full_catalog or max_tools <= 0 or len(full_tools) <= max_tools:
        return list(full_tools)

    preferred = [name for name in (preferred_names or []) if name]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_tool(tool: dict[str, Any]) -> None:
        name = _tool_schema_name(tool)
        if not name or name in seen:
            return
        selected.append(tool)
        seen.add(name)

    for tool in full_tools:
        name = _tool_schema_name(tool)
        if _is_discovery_name(name):
            add_tool(tool)

    for want in preferred:
        if len(selected) >= max_tools:
            break
        for tool in full_tools:
            name = _tool_schema_name(tool)
            if _names_match(name, want):
                add_tool(tool)
                break

    if len(selected) < max_tools:
        for tool in full_tools:
            if len(selected) >= max_tools:
                break
            add_tool(tool)

    return selected[:max_tools]

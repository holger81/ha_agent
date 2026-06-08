"""Unit tests for agent context building."""

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

    if name == "context":
        conv = types.ModuleType("homeassistant.components.conversation")
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = conv

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


context = _load_module("context")


def test_format_exposed_entities() -> None:
    """Exposed entities are formatted for the system prompt."""
    text = context.format_exposed_entities(
        [
            {
                "entity_id": "light.dining",
                "name": "Dining",
                "state": "on",
                "area_name": "Dining room",
            }
        ]
    )
    assert "light.dining" in text
    assert "Dining room" in text


def test_build_tool_context_adds_news_hint() -> None:
    """News queries receive MCP news hint."""
    tool_context = context.build_tool_context("What's the news?", [])
    assert "news_curate" in tool_context


def test_build_tool_context_adds_device_search_hint() -> None:
    """Device actions without exposed match get search hint."""
    tool_context = context.build_tool_context(
        "open the patio door",
        [{"entity_id": "light.kitchen", "name": "Kitchen"}],
    )
    assert "ha_search_entities" in tool_context


def test_entity_matches_query() -> None:
    """Entity matching uses names and aliases."""
    entity = {
        "entity_id": "cover.patio",
        "name": "Patio door",
        "aliases": ["Jonathan patio"],
    }
    assert context.entity_matches_query(entity, "open Jonathan patio door")


def test_build_tool_context_adds_email_hint() -> None:
    """Email queries receive MCP mail tool hint."""
    tool_context = context.build_tool_context("do I have new emails", [])
    assert "imap_mailbox_status" in tool_context


def test_is_email_query() -> None:
    """Email intent detection works."""
    assert context.is_email_query("do I have new emails")
    assert not context.is_email_query("turn off the lights")

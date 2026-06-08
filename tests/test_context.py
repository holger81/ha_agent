"""Unit tests for context builder."""

from __future__ import annotations

from ha_agent.context import (
    build_tool_context,
    entity_matches_query,
    format_exposed_entities,
    is_device_action_query,
    is_news_query,
    parse_exposed_entities,
)


def test_parse_exposed_entities_from_json_string() -> None:
    """JSON string payloads are parsed into entity dicts."""
    raw = '[{"entity_id":"light.kitchen","name":"Kitchen"}]'
    entities = parse_exposed_entities(raw)
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "light.kitchen"


def test_format_exposed_entities() -> None:
    """Entities are formatted for the system prompt."""
    text = format_exposed_entities(
        [{"entity_id": "light.dining", "name": "Dining", "state": "on"}]
    )
    assert "light.dining" in text
    assert "Dining" in text


def test_entity_matches_query() -> None:
    """Entity matching uses entity_id and name tokens."""
    entity = {"entity_id": "cover.patio", "name": "Patio Door"}
    assert entity_matches_query(entity, "open the patio door")


def test_build_tool_context_includes_news_hint() -> None:
    """News queries include MCP news tool guidance."""
    context = build_tool_context("what is the news?", [])
    assert "mcp_news__news_curate" in context


def test_is_news_query() -> None:
    """News intent detection works."""
    assert is_news_query("Give me the headlines")
    assert not is_news_query("turn off the lights")


def test_is_device_action_query() -> None:
    """Device action intent detection works."""
    assert is_device_action_query("turn off the kitchen light")
    assert not is_device_action_query("what time is it")

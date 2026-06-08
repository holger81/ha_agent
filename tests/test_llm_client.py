"""Unit tests for LLM client parsing."""

from __future__ import annotations

from unittest.mock import MagicMock

from ha_agent.config import LlmBackend
from ha_agent.llm_client import LlmClient


def test_parse_completion_with_tool_calls() -> None:
    """Tool calls are parsed from chat completion JSON."""
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "mcp_call_tool",
                                "arguments": '{"toolName":"mcp_news__news_curate"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    client = LlmClient(MagicMock())
    result = client._parse_completion(data)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "mcp_call_tool"
    assert result.tool_calls[0].id == "call_1"


def test_parse_completion_text_only() -> None:
    """Plain text responses are parsed."""
    data = {
        "choices": [
            {
                "message": {
                    "content": "Hello there.",
                    "tool_calls": [],
                }
            }
        ]
    }
    client = LlmClient(MagicMock())
    result = client._parse_completion(data)

    assert result.content == "Hello there."
    assert result.tool_calls == []


def test_llm_backend_defaults() -> None:
    """LlmBackend exposes sensible defaults."""
    backend = LlmBackend(base_url="http://localhost/v1", model="test")
    assert backend.max_tokens == 4096
    assert backend.temperature == 0.3

"""Unit tests for LLM client parsing."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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

    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")

        class HomeAssistantError(Exception):
            pass

        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc

    for dep in ("const", "config_helpers", "thinking"):
        if dep != name and f"ha_agent.{dep}" not in sys.modules:
            _load_module(dep)

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


llm_client = _load_module("llm_client")
config_helpers = _load_module("config_helpers")


def test_parse_completion_with_embedded_tool_call() -> None:
    """Embedded Gemma-style tool calls in content are parsed."""
    data = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<|tool_call|>call:home_assistant__ha_search_entities'
                        '{arguments: {query:"email"}}<tool_call|>'
                    ),
                    "tool_calls": [],
                }
            }
        ]
    }
    client = llm_client.LlmClient(MagicMock())
    result = client._parse_completion(data)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "callTool"
    assert result.content is None


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
                                "name": "callTool",
                                "arguments": (
                                    '{"toolName":"mcp_news__news_curate","arguments":{}}'
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }
    client = llm_client.LlmClient(MagicMock())
    result = client._parse_completion(data)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "callTool"
    assert "news_curate" in result.tool_calls[0].arguments


@pytest.mark.asyncio
async def test_chat_parses_response() -> None:
    """chat() returns parsed assistant content."""
    payload = {
        "choices": [{"message": {"content": "Hello there", "tool_calls": []}}]
    }
    response = AsyncMock()
    response.status = 200
    response.text = AsyncMock(return_value=json.dumps(payload))

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.post = MagicMock(return_value=context)

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test-model",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    client = llm_client.LlmClient(session)
    result = await client.chat([{"role": "user", "content": "Hi"}], backend)

    assert result.content == "Hello there"
    assert not result.tool_calls


@pytest.mark.asyncio
async def test_list_models_returns_sorted_ids() -> None:
    """list_models() parses OpenAI-compatible /models responses."""
    payload = {
        "data": [
            {"id": "model-b"},
            {"id": "model-a"},
            {"id": ""},
            {"id": "model-c"},
        ]
    }
    response = AsyncMock()
    response.status = 200
    response.text = AsyncMock(return_value=json.dumps(payload))

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.get = MagicMock(return_value=context)

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test-model",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    client = llm_client.LlmClient(session)
    models = await client.list_models(backend)

    assert models == ["model-a", "model-b", "model-c"]


def test_stream_timeout_uses_idle_limit_without_total_cap() -> None:
    """Streaming should not cap total duration; only idle time between chunks."""
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test-model",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=180,
        thinking_level="off",
    )
    timeout = llm_client.LlmClient._stream_timeout(backend)

    assert timeout.total is None
    assert timeout.sock_read == 180
    assert timeout.connect == 30


def test_parse_sse_reasoning_content() -> None:
    """Streaming parser separates reasoning_content from content."""

    async def _run() -> None:
        client = llm_client.LlmClient(MagicMock())
        session = llm_client.StreamChatSession()

        class FakeResponse:
            def __init__(self) -> None:
                self.content = self

            async def iter_any(self):
                payload = {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "Let me think",
                                "content": "Hello",
                            }
                        }
                    ]
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()

        response = FakeResponse()
        chunks: list[llm_client.StreamChunk] = []
        async for chunk in client._iter_sse_deltas(response, session):
            chunks.append(chunk)

        assert [
            chunk.reasoning_content for chunk in chunks if chunk.reasoning_content
        ] == ["Let me think"]
        assert [chunk.content for chunk in chunks if chunk.content] == ["Hello"]
        assert session.reasoning_content == "Let me think"
        assert session.content == "Hello"

    import asyncio

    asyncio.run(_run())


def test_payload_includes_thinking_for_medium() -> None:
    """Chat payloads include reasoning parameters when thinking is enabled."""
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test-model",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="medium",
    )
    client = llm_client.LlmClient(MagicMock())
    payload = client._payload(
        [{"role": "user", "content": "Hi"}],
        backend,
        None,
        stream=False,
    )

    assert payload["reasoning_effort"] == "medium"
    assert payload["reasoning_format"] == "deepseek"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_payload_omits_thinking_when_off() -> None:
    """Off thinking level disables reasoning in the payload."""
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test-model",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    client = llm_client.LlmClient(MagicMock())
    payload = client._payload(
        [{"role": "user", "content": "Hi"}],
        backend,
        None,
        stream=False,
    )

    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"]["enable_thinking"] is False

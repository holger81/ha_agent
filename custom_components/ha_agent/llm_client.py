"""OpenAI-compatible chat client for llama.cpp / local LLM servers."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .config_helpers import LlmBackend
from .const import LOGGER
from .embedded_tools import parse_embedded_tool_calls, strip_embedded_tool_markup
from .thinking import apply_thinking_to_payload


def stream_text_delta(buffer: str, piece: str) -> tuple[str, str]:
    """Append a streamed text fragment, handling cumulative providers.

    Some LLM servers send the full text-so-far in each SSE delta instead of
    only the new suffix. Returns ``(new_buffer, delta_to_emit)``.
    """
    if not piece:
        return buffer, ""
    if buffer and piece.startswith(buffer):
        return piece, piece[len(buffer) :]
    if buffer and buffer.endswith(piece):
        return buffer, ""
    return buffer + piece, piece


MCP_CALL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mcp_call_tool",
        "description": (
            "Call an MCP tool by name with flat arguments. "
            "Use for Home Assistant actions, news, mail, and other MCP tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "toolName": {
                    "type": "string",
                    "description": "Exact MCP tool name, e.g. mcp_news__news_curate",
                },
                "arguments": {
                    "type": "object",
                    "description": "Flat tool arguments object",
                },
            },
            "required": ["toolName"],
        },
    },
}


@dataclass(slots=True)
class ToolCall:
    """Parsed tool call from an LLM response."""

    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class ChatResult:
    """Non-streaming chat completion result."""

    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class StreamChunk:
    """One streamed delta from the LLM server."""

    content: str = ""
    reasoning_content: str = ""


@dataclass(slots=True)
class StreamChatSession:
    """Mutable stream state populated while iterating chat_stream."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    _tool_call_builders: dict[int, dict[str, str]] = field(
        default_factory=dict,
        repr=False,
    )


class LlmClient:
    """Async OpenAI-compatible chat client."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the client."""
        self._session = session
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self, backend: LlmBackend) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        return headers

    @staticmethod
    def _stream_timeout(backend: LlmBackend) -> aiohttp.ClientTimeout:
        """Idle timeout between chunks plus an absolute cap on stream duration."""
        connect = min(30, backend.timeout)
        total = max(backend.timeout * 2, backend.timeout + 60)
        return aiohttp.ClientTimeout(
            total=total,
            connect=connect,
            sock_read=backend.timeout,
        )

    def _payload(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": backend.model,
            "messages": messages,
            "max_tokens": backend.max_tokens,
            "temperature": backend.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        apply_thinking_to_payload(payload, backend.thinking_level)
        return payload

    async def check_connection(self, backend: LlmBackend) -> None:
        """Verify the LLM server is reachable."""
        await self.list_models(backend)

    async def list_models(self, backend: LlmBackend) -> list[str]:
        """Return model ids from the OpenAI-compatible /models endpoint."""
        url = f"{backend.base_url}/models"
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with self._session.get(
                url,
                headers=self._headers(backend),
                timeout=timeout,
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise HomeAssistantError(
                        f"LLM models failed (HTTP {response.status}): {body[:300]}"
                    )
                data = json.loads(body)
        except TimeoutError as err:
            raise HomeAssistantError("LLM server timed out") from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot connect to LLM server: {err}") from err
        except json.JSONDecodeError as err:
            raise HomeAssistantError("LLM returned invalid models JSON") from err

        models: list[str] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
        if not models:
            raise HomeAssistantError("LLM server returned no models")
        return sorted(models)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
        tools: list[dict[str, Any]] | None = None,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Run a non-streaming chat completion."""
        url = f"{backend.base_url}/chat/completions"
        payload = self._payload(
            messages,
            backend,
            tools,
            stream=False,
            response_format=response_format,
        )
        timeout = aiohttp.ClientTimeout(total=backend.timeout)
        started = time.perf_counter()

        try:
            async with self._session.post(
                url,
                json=payload,
                headers=self._headers(backend),
                timeout=timeout,
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise HomeAssistantError(
                        f"LLM chat failed (HTTP {response.status}): {body[:300]}"
                    )
                data = json.loads(body)
        except TimeoutError as err:
            raise HomeAssistantError("LLM chat timed out") from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"LLM chat request failed: {err}") from err
        except json.JSONDecodeError as err:
            raise HomeAssistantError("LLM returned invalid JSON") from err

        result = self._parse_completion(data)
        result.latency_ms = (time.perf_counter() - started) * 1000
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        if usage:
            result.prompt_tokens = _optional_int(usage.get("prompt_tokens"))
            result.completion_tokens = _optional_int(usage.get("completion_tokens"))
        return result

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
        tools: list[dict[str, Any]] | None = None,
        session: StreamChatSession | None = None,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream assistant text deltas from a chat completion."""
        url = f"{backend.base_url}/chat/completions"
        payload = self._payload(
            messages,
            backend,
            tools,
            stream=True,
            response_format=response_format,
        )
        timeout = self._stream_timeout(backend)
        stream_session = session or StreamChatSession()

        try:
            async with self._session.post(
                url,
                json=payload,
                headers=self._headers(backend),
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise HomeAssistantError(
                        f"LLM stream failed (HTTP {response.status}): {body[:300]}"
                    )
                async for chunk in self._iter_sse_deltas(
                    response,
                    stream_session,
                ):
                    if chunk.content or chunk.reasoning_content:
                        yield chunk
                stream_session.tool_calls = self._finalize_stream_tool_calls(
                    stream_session._tool_call_builders
                )
                if session is not None:
                    session.content = stream_session.content
                    session.tool_calls = stream_session.tool_calls
        except TimeoutError as err:
            raise HomeAssistantError(
                "LLM stream timed out waiting for the server. "
                f"Increase the LLM timeout (currently {backend.timeout}s) "
                "in HA Agent settings for large models or tool-heavy prompts."
            ) from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"LLM stream request failed: {err}") from err

    async def _iter_sse_deltas(
        self,
        response: aiohttp.ClientResponse,
        session: StreamChatSession,
    ) -> AsyncIterator[StreamChunk]:
        """Parse OpenAI-style SSE chunks."""
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    LOGGER.debug("Skipping invalid SSE chunk: %s", data_str[:80])
                    continue
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    content_part = ""
                    reasoning_part = ""
                    if reasoning:
                        session.reasoning_content, reasoning_part = stream_text_delta(
                            session.reasoning_content,
                            reasoning,
                        )
                    if content:
                        session.content, content_part = stream_text_delta(
                            session.content,
                            content,
                        )
                    if content_part or reasoning_part:
                        yield StreamChunk(
                            content=content_part,
                            reasoning_content=reasoning_part,
                        )
                    for tool_delta in delta.get("tool_calls") or []:
                        self._merge_stream_tool_delta(
                            session._tool_call_builders,
                            tool_delta,
                        )

    def _merge_stream_tool_delta(
        self,
        builders: dict[int, dict[str, str]],
        tool_delta: dict[str, Any],
    ) -> None:
        """Accumulate streamed tool-call fragments."""
        index = int(tool_delta.get("index", 0))
        entry = builders.setdefault(
            index,
            {"id": "", "name": "", "arguments": ""},
        )
        if call_id := tool_delta.get("id"):
            entry["id"] = call_id
        function = tool_delta.get("function") or {}
        if name := function.get("name"):
            entry["name"] = name
        if arguments := function.get("arguments"):
            entry["arguments"] += arguments

    def _finalize_stream_tool_calls(
        self,
        builders: dict[int, dict[str, str]],
    ) -> list[ToolCall]:
        """Convert accumulated streamed tool-call fragments to ToolCall objects."""
        tool_calls: list[ToolCall] = []
        for index in sorted(builders):
            entry = builders[index]
            if not entry.get("name"):
                continue
            tool_calls.append(
                ToolCall(
                    id=entry.get("id") or f"call_{index}",
                    name=entry["name"],
                    arguments=entry.get("arguments") or "{}",
                )
            )
        return tool_calls

    def _parse_completion(self, data: dict[str, Any]) -> ChatResult:
        """Parse a chat completion response."""
        choices = data.get("choices") or []
        if not choices:
            return ChatResult(content="", tool_calls=[], assistant_message={})

        message = choices[0].get("message") or {}
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []

        for index, call in enumerate(raw_tool_calls):
            function = call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "{}",
                )
            )

        if not tool_calls and content:
            for index, embedded in enumerate(parse_embedded_tool_calls(content)):
                tool_calls.append(
                    ToolCall(
                        id=embedded.id or f"call_embedded_{index}",
                        name=embedded.name,
                        arguments=embedded.arguments,
                    )
                )
            if tool_calls:
                content = strip_embedded_tool_markup(content) or None

        assistant_message = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ]

        return ChatResult(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

"""OpenAI-compatible chat client for llama.cpp / local LLM servers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .config_helpers import LlmBackend
from .const import LOGGER
from .embedded_tools import parse_embedded_tool_calls, strip_embedded_tool_markup

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
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)


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

    def _payload(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
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
        return payload

    async def check_connection(self, backend: LlmBackend) -> None:
        """Verify the LLM server is reachable."""
        url = f"{backend.base_url}/models"
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with self._session.get(
                url,
                headers=self._headers(backend),
                timeout=timeout,
            ) as response:
                if response.status >= 500:
                    raise HomeAssistantError(
                        f"LLM server returned HTTP {response.status}"
                    )
        except TimeoutError as err:
            raise HomeAssistantError("LLM server timed out") from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot connect to LLM server: {err}") from err

    async def chat(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        """Run a non-streaming chat completion."""
        url = f"{backend.base_url}/chat/completions"
        payload = self._payload(messages, backend, tools, stream=False)
        timeout = aiohttp.ClientTimeout(total=backend.timeout)

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

        return self._parse_completion(data)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        backend: LlmBackend,
    ) -> AsyncIterator[str]:
        """Stream assistant text deltas from a chat completion."""
        url = f"{backend.base_url}/chat/completions"
        payload = self._payload(messages, backend, tools=None, stream=True)
        timeout = aiohttp.ClientTimeout(total=backend.timeout)

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
                async for delta in self._iter_sse_deltas(response):
                    if delta:
                        yield delta
        except TimeoutError as err:
            raise HomeAssistantError("LLM stream timed out") from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"LLM stream request failed: {err}") from err

    async def _iter_sse_deltas(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[str]:
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
                    if content:
                        yield content

    def _parse_completion(self, data: dict[str, Any]) -> ChatResult:
        """Parse a chat completion response."""
        choices = data.get("choices") or []
        if not choices:
            return ChatResult(content="", tool_calls=[], assistant_message={})

        message = choices[0].get("message") or {}
        content = message.get("content")
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
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )

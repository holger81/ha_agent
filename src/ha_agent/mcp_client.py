"""HTTP client for MCP Proxy (streamable HTTP / JSON-RPC)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from .config import McpConfig
from .exceptions import HaAgentError

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpProxyClient:
    """Async MCP Proxy client using JSON-RPC over HTTP."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: McpConfig,
        *,
        client_name: str = "ha_agent",
        client_version: str = "0.1.0",
    ) -> None:
        """Initialize the client."""
        self._session = session
        self._config = config
        self._client_name = client_name
        self._client_version = client_version
        self._session_id: str | None = None
        self._request_id = 0
        self._initialized = False

    @property
    def url(self) -> str:
        """Return the MCP endpoint URL."""
        return self._config.url

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def check_health(self) -> None:
        """Verify MCP Proxy health endpoint."""
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with self._session.get(
                self._config.resolved_health_url(),
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                if response.status >= 500:
                    raise HaAgentError(
                        f"MCP health check failed (HTTP {response.status})"
                    )
        except TimeoutError as err:
            raise HaAgentError("MCP health check timed out") from err
        except aiohttp.ClientError as err:
            raise HaAgentError(f"Cannot reach MCP Proxy: {err}") from err

    async def initialize(self) -> None:
        """Initialize MCP session."""
        if self._initialized:
            return

        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
            },
        )
        if not result:
            raise HaAgentError("MCP initialize returned empty result")

        await self._rpc("notifications/initialized", None, notification=True)
        self._initialized = True

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call an upstream MCP tool via the proxy callTool wrapper."""
        await self.initialize()
        result = await self._rpc(
            "tools/call",
            {
                "name": "callTool",
                "arguments": {
                    "toolName": tool_name,
                    "arguments": arguments or {},
                },
            },
        )
        return self._extract_tool_result(result)

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        notification: bool = False,
    ) -> Any:
        """Send a JSON-RPC request to the MCP endpoint."""
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if not notification:
            body["id"] = self._next_id()
        if params is not None:
            body["params"] = params

        timeout = aiohttp.ClientTimeout(total=self._config.timeout)
        try:
            async with self._session.post(
                self._config.url,
                json=body,
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                session_header = response.headers.get("Mcp-Session-Id")
                if session_header:
                    self._session_id = session_header

                content_type = response.headers.get("Content-Type", "")
                raw = await response.text()
                if response.status >= 400:
                    raise HaAgentError(
                        f"MCP {method} failed (HTTP {response.status}): {raw[:300]}"
                    )

                if "text/event-stream" in content_type:
                    return self._parse_sse_json(raw)

                if not raw.strip():
                    return None

                data = json.loads(raw)
        except TimeoutError as err:
            raise HaAgentError(f"MCP {method} timed out") from err
        except aiohttp.ClientError as err:
            raise HaAgentError(f"MCP {method} request failed: {err}") from err
        except json.JSONDecodeError as err:
            raise HaAgentError(f"MCP {method} returned invalid JSON") from err

        if error := data.get("error"):
            message = error.get("message") or str(error)
            raise HaAgentError(f"MCP error: {message}")

        return data.get("result")

    def _parse_sse_json(self, raw: str) -> Any:
        """Extract the last JSON result from an SSE response."""
        last_result: Any = None
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "result" in data:
                last_result = data["result"]
            elif "error" in data:
                message = data["error"].get("message") or str(data["error"])
                raise HaAgentError(f"MCP error: {message}")
        return last_result

    def _extract_tool_result(self, result: Any) -> str:
        """Normalize MCP tool results to a string for the LLM."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            if result.get("isError"):
                content = result.get("content") or []
                return self._content_blocks_to_text(content) or "Tool returned an error"
            content = result.get("content")
            if content is not None:
                text = self._content_blocks_to_text(content)
                if text:
                    return text
            return json.dumps(result, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _content_blocks_to_text(content: Any) -> str:
        """Convert MCP content blocks to plain text."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if text := block.get("text"):
                    parts.append(str(text))
                elif data := block.get("data"):
                    parts.append(str(data))
        return "\n".join(parts)

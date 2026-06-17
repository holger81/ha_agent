"""HTTP client for MCP Proxy (streamable HTTP / JSON-RPC)."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .config_helpers import McpConfig
from .const import LOGGER, MCP_SESSION_TOOLS_TTL_SECONDS, MCP_TOOLS_LIST_MAX_PAGES
from .mcp_errors import friendly_mcp_http_error, friendly_mcp_json_error
from .mcp_session import (
    FALLBACK_MCP_TOOLS,
    format_mcp_session_prompt,
    mcp_tools_to_openai_schemas,
)

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpProxyClient:
    """Async MCP Proxy client using JSON-RPC over HTTP."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: McpConfig,
    ) -> None:
        """Initialize the client."""
        self._session = session
        self._config = config
        self._session_id: str | None = None
        self._request_id = 0
        self._initialized = False
        self._init_result: dict[str, Any] = {}
        self._instructions = ""
        self._session_tools: list[dict[str, Any]] = []
        self._session_tools_cached_at = 0.0

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
                self._config.health_url,
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise HomeAssistantError(
                        friendly_mcp_http_error(
                            method="health",
                            status=response.status,
                            body=body,
                        )
                    )
        except TimeoutError as err:
            raise HomeAssistantError(
                "MCP Proxy timed out. Try again or increase the MCP timeout."
            ) from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"Cannot reach MCP Proxy: {err}") from err

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session and load protocol instructions."""
        if self._initialized:
            return self._init_result

        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ha_agent", "version": "0.1.0"},
            },
        )
        if not isinstance(result, dict):
            raise HomeAssistantError("MCP initialize returned empty result")

        self._init_result = result
        self._instructions = str(result.get("instructions") or "").strip()

        await self._rpc("notifications/initialized", None, notification=True)
        self._initialized = True
        await self._load_session_tools(force_refresh=True)
        return self._init_result

    async def ensure_session(self) -> None:
        """Ensure MCP initialize and tools/list have completed."""
        await self.initialize()
        await self._load_session_tools()

    async def get_session_prompt(self) -> str:
        """Return MCP initialize instructions and session tool summary."""
        await self.ensure_session()
        return format_mcp_session_prompt(
            instructions=self._instructions,
            init_result=self._init_result,
            session_tools=self._session_tools,
        )

    async def get_llm_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible schemas from MCP tools/list."""
        await self.ensure_session()
        tools = self._session_tools or FALLBACK_MCP_TOOLS
        return mcp_tools_to_openai_schemas(tools)

    async def _load_session_tools(self, *, force_refresh: bool = False) -> None:
        """Fetch session-level tools via MCP tools/list."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._session_tools
            and now - self._session_tools_cached_at < MCP_SESSION_TOOLS_TTL_SECONDS
        ):
            return

        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        page = 0

        while page < MCP_TOOLS_LIST_MAX_PAGES:
            page += 1
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor

            result = await self._rpc("tools/list", params or None)
            if not isinstance(result, dict):
                break

            page_tools = result.get("tools") or []
            if isinstance(page_tools, list):
                tools.extend(entry for entry in page_tools if isinstance(entry, dict))

            cursor = result.get("nextCursor")
            if not cursor:
                break

        if page >= MCP_TOOLS_LIST_MAX_PAGES and cursor:
            LOGGER.warning(
                "MCP tools/list stopped after %s pages; more tools may be unavailable",
                MCP_TOOLS_LIST_MAX_PAGES,
            )

        if tools:
            self._session_tools = tools
        elif not self._session_tools:
            LOGGER.warning(
                "MCP tools/list returned no tools; using discovery fallback set"
            )
            self._session_tools = list(FALLBACK_MCP_TOOLS)

        self._session_tools_cached_at = now

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call an MCP tool via tools/call using the protocol tool name."""
        await self.ensure_session()
        result = await self._rpc(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments or {},
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
                    raise HomeAssistantError(
                        friendly_mcp_http_error(
                            method=method,
                            status=response.status,
                            body=raw,
                        )
                    )

                if "text/event-stream" in content_type:
                    return self._parse_sse_json(raw)

                if not raw.strip():
                    return None

                data = json.loads(raw)
        except TimeoutError as err:
            raise HomeAssistantError(
                "MCP Proxy timed out. Try again or increase the MCP timeout."
            ) from err
        except aiohttp.ClientError as err:
            raise HomeAssistantError(f"MCP {method} request failed: {err}") from err
        except json.JSONDecodeError as err:
            raise HomeAssistantError(f"MCP {method} returned invalid JSON") from err

        if error := data.get("error"):
            message = error.get("message") or str(error)
            raise HomeAssistantError(friendly_mcp_json_error(message))

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
                raise HomeAssistantError(friendly_mcp_json_error(message))
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


def derive_health_url_from_mcp(mcp_url: str) -> str:
    """Return default health URL for an MCP endpoint."""
    parsed = urlparse(mcp_url.rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}/api/health"

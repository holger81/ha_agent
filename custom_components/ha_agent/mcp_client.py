"""HTTP client for MCP Proxy (streamable HTTP / JSON-RPC)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp
from homeassistant.exceptions import HomeAssistantError

from .config_helpers import McpConfig
from .const import LOGGER, MCP_SESSION_TOOLS_TTL_SECONDS, MCP_TOOLS_LIST_MAX_PAGES
from .mcp_errors import (
    MCP_SESSION_EXPIRED_RETRY_MESSAGE,
    friendly_mcp_http_error,
    friendly_mcp_json_error,
    is_mcp_session_expired_status,
)
from .mcp_session import (
    FALLBACK_MCP_TOOLS,
    format_mcp_session_prompt,
    mcp_tools_to_openai_schemas,
)
from .sse import iter_sse_data_lines
from .tools import classify_tool_output

MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC methods that are safe to replay after the session was re-created.
# ``tools/call`` is deliberately excluded: we cannot tell whether the server
# executed it before dropping the session, so the caller must retry.
MCP_IDEMPOTENT_METHODS: frozenset[str] = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "prompts/list",
        "resources/list",
        "resources/templates/list",
    }
)


class McpSessionExpired(Exception):
    """Internal: the server rejected our Mcp-Session-Id (session terminated)."""

    def __init__(
        self,
        method: str,
        status: int,
        body: str,
        *,
        session_id: str | None,
    ) -> None:
        super().__init__(f"MCP session expired during {method} (HTTP {status})")
        self.method = method
        self.status = status
        self.body = body
        self.session_id = session_id


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
        self._init_lock = asyncio.Lock()

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
        """Initialize the MCP session and load protocol instructions.

        Serialized with a lock so concurrent turns share one handshake and a
        session re-initialization never races a second ``initialize``.
        """
        if self._initialized:
            return self._init_result

        async with self._init_lock:
            if self._initialized:
                return self._init_result

            # A stale session id must not be sent with a fresh initialize.
            self._session_id = None
            result = await self._rpc(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ha_agent", "version": "0.1.0"},
                },
                reinit=False,
            )
            if not isinstance(result, dict):
                raise HomeAssistantError("MCP initialize returned empty result")

            self._init_result = result
            self._instructions = str(result.get("instructions") or "").strip()

            await self._rpc(
                "notifications/initialized",
                None,
                notification=True,
                reinit=False,
            )
            self._initialized = True
            await self._load_session_tools(force_refresh=True, reinit=False)
            return self._init_result

    def _reset_session(self) -> None:
        """Forget the server session so the next call re-runs initialize."""
        self._session_id = None
        self._initialized = False
        self._init_result = {}
        self._instructions = ""
        self._session_tools = []
        self._session_tools_cached_at = 0.0

    async def _recover_expired_session(self, err: McpSessionExpired) -> None:
        """Drop the dead session and run a fresh initialize handshake."""
        LOGGER.info(
            "MCP session expired during %s (HTTP %s); re-initializing",
            err.method,
            err.status,
        )
        # Only tear down if nobody re-initialized in the meantime; otherwise
        # a concurrent recovery would wipe the fresh session.
        if self._session_id == err.session_id:
            self._reset_session()
        await self.initialize()

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

    async def _load_session_tools(
        self,
        *,
        force_refresh: bool = False,
        reinit: bool = True,
    ) -> None:
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

            result = await self._rpc("tools/list", params or None, reinit=reinit)
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
        reinit: bool = True,
    ) -> Any:
        """Send a JSON-RPC request to the MCP endpoint.

        When the server reports that our session is gone (404, or 400 while a
        session id was sent) the session is re-initialized. Idempotent methods
        are then replayed once; ``tools/call`` is not replayed because the
        server may already have executed it — a clear error asks the caller to
        retry instead.
        """
        try:
            return await self._rpc_once(method, params, notification=notification)
        except McpSessionExpired as err:
            if not reinit:
                raise HomeAssistantError(
                    friendly_mcp_http_error(
                        method=method,
                        status=err.status,
                        body=err.body,
                        session_expired=True,
                    )
                ) from err
            await self._recover_expired_session(err)
            if method in MCP_IDEMPOTENT_METHODS:
                return await self._rpc(
                    method,
                    params,
                    notification=notification,
                    reinit=False,
                )
            raise HomeAssistantError(MCP_SESSION_EXPIRED_RETRY_MESSAGE) from err

    async def _rpc_once(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        notification: bool,
    ) -> Any:
        """Send one JSON-RPC request and parse its JSON or SSE response."""
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        request_id: int | None = None
        if not notification:
            request_id = self._next_id()
            body["id"] = request_id
        if params is not None:
            body["params"] = params

        sent_session_id = self._session_id
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
                if response.status >= 400:
                    raw = await response.text()
                    if is_mcp_session_expired_status(
                        response.status, had_session=sent_session_id is not None
                    ):
                        raise McpSessionExpired(
                            method,
                            response.status,
                            raw,
                            session_id=sent_session_id,
                        )
                    raise HomeAssistantError(
                        friendly_mcp_http_error(
                            method=method,
                            status=response.status,
                            body=raw,
                        )
                    )

                if "text/event-stream" in content_type:
                    # Stream until our reply arrives; the server may keep the
                    # SSE connection open long after that.
                    return await self._read_sse_result(response, request_id)

                raw = await response.text()
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

        return self._result_from_json(method, data, request_id)

    def _result_from_json(self, method: str, data: Any, request_id: int | None) -> Any:
        """Extract ``result`` from a JSON-RPC response or batch for our id."""
        message: dict[str, Any] | None = None
        if isinstance(data, dict):
            message = data
        elif isinstance(data, list):
            # Batch: prefer the entry addressed to us, else the first dict.
            for item in data:
                if isinstance(item, dict) and _rpc_id_matches(item, request_id):
                    message = item
                    break
            else:
                message = next((item for item in data if isinstance(item, dict)), None)
        if message is None:
            if data is None or data == []:
                return None
            raise HomeAssistantError(f"MCP {method} returned unexpected JSON")
        if "error" in message and message["error"] is not None:
            raise HomeAssistantError(
                friendly_mcp_json_error(_rpc_error_message(message["error"]))
            )
        return message.get("result")

    async def _read_sse_result(
        self,
        response: aiohttp.ClientResponse,
        request_id: int | None,
    ) -> Any:
        """Return the JSON-RPC result for ``request_id`` from an SSE body.

        Reads incrementally and returns at the first message whose ``id``
        matches ours; other messages (server notifications or requests) are
        skipped. Messages without an id are kept as a fallback for servers
        that omit it.
        """
        fallback: dict[str, Any] | None = None
        async for payload in iter_sse_data_lines(response):
            text = payload.strip()
            if not text or text == "[DONE]":
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            messages = data if isinstance(data, list) else [data]
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if "result" not in message and "error" not in message:
                    continue
                if _rpc_id_matches(message, request_id):
                    return self._raise_or_result(message)
                if "id" not in message or message.get("id") is None:
                    fallback = message
        if fallback is not None:
            return self._raise_or_result(fallback)
        return None

    @staticmethod
    def _raise_or_result(message: dict[str, Any]) -> Any:
        error = message.get("error")
        if error is not None:
            raise HomeAssistantError(friendly_mcp_json_error(_rpc_error_message(error)))
        return message.get("result")

    def _extract_tool_result(self, result: Any) -> str:
        """Normalize MCP tool results to a string for the LLM."""
        if result is None:
            return ""
        if isinstance(result, str):
            return classify_tool_output(result)

        if isinstance(result, dict):
            if result.get("isError"):
                content = result.get("content") or []
                text = self._content_blocks_to_text(content) or "Tool returned an error"
                return classify_tool_output(text)
            content = result.get("content")
            if content is not None:
                text = self._content_blocks_to_text(content)
                if text:
                    return classify_tool_output(text)
            return json.dumps(result, ensure_ascii=False)

        return classify_tool_output(json.dumps(result, ensure_ascii=False))

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


def _rpc_id_matches(message: dict[str, Any], request_id: int | None) -> bool:
    """Compare a JSON-RPC response id with ours (tolerating stringified ids)."""
    if request_id is None:
        return False
    message_id = message.get("id")
    if message_id is None:
        return False
    return message_id == request_id or str(message_id) == str(request_id)


def _rpc_error_message(error: Any) -> str:
    """Extract a message from a JSON-RPC ``error`` member of any shape."""
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(error, ensure_ascii=False)
    return str(error)


def derive_health_url_from_mcp(mcp_url: str) -> str:
    """Return default health URL for an MCP endpoint."""
    parsed = urlparse(mcp_url.rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}/api/health"

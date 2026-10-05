"""User-friendly MCP error messages."""

from __future__ import annotations

MCP_SESSION_EXPIRED_RETRY_MESSAGE = (
    "MCP session expired; re-initialized — retry the tool call."
)


def is_mcp_session_expired_status(status: int, *, had_session: bool) -> bool:
    """Return True when an HTTP status means the MCP session is gone.

    Per the MCP streamable-HTTP transport a server answers requests carrying
    a terminated ``Mcp-Session-Id`` with 404 (some proxies use 400). Without
    a session id a 404 means the URL is wrong, not that the session expired.
    """
    return had_session and status in {400, 404}


def friendly_mcp_http_error(
    *,
    method: str,
    status: int,
    body: str,
    session_expired: bool = False,
) -> str:
    """Map MCP HTTP failures to Assist-friendly messages."""
    detail = body.strip()[:200]
    if session_expired:
        return (
            f"MCP session expired (HTTP {status}) during {method}; "
            "the session was re-initialized. Retry the request."
        )
    if status in {401, 403}:
        return "MCP authentication failed. Check the bearer token in HA Agent settings."
    if status == 404:
        return (
            f"MCP endpoint not found for {method} (HTTP 404, no active session). "
            "Verify the MCP Proxy URL in HA Agent settings."
        )
    if status >= 500:
        return (
            f"MCP Proxy is unavailable (HTTP {status}). "
            "Check that the proxy service is running."
        )
    if detail:
        return f"MCP {method} failed (HTTP {status}): {detail}"
    return f"MCP {method} failed with HTTP {status}."


def friendly_mcp_json_error(message: str) -> str:
    """Map MCP JSON-RPC errors to Assist-friendly messages."""
    lowered = message.lower()
    if "not authenticated" in lowered or "unauthorized" in lowered:
        return "MCP authentication failed. Check the bearer token in HA Agent settings."
    if "timeout" in lowered or "timed out" in lowered:
        return "MCP Proxy timed out. Try again or increase the MCP timeout."
    return f"MCP error: {message}"

"""User-friendly MCP error messages."""

from __future__ import annotations


def friendly_mcp_http_error(
    *,
    method: str,
    status: int,
    body: str,
) -> str:
    """Map MCP HTTP failures to Assist-friendly messages."""
    detail = body.strip()[:200]
    if status in {401, 403}:
        return (
            "MCP authentication failed. Check the bearer token in HA Agent settings."
        )
    if status == 404:
        return (
            f"MCP endpoint not found for {method}. "
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
        return (
            "MCP authentication failed. Check the bearer token in HA Agent settings."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "MCP Proxy timed out. Try again or increase the MCP timeout."
    return f"MCP error: {message}"

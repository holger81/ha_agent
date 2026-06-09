#!/usr/bin/env python3
"""Smoke-test MCP Proxy via direct JSON-RPC (Phase 5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "ha_agent"


def _load_mcp_client():
    import importlib.util
    import types

    package = types.ModuleType("ha_agent")
    package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
    sys.modules["ha_agent"] = package

    ha_exc = types.ModuleType("homeassistant.exceptions")

    class HomeAssistantError(Exception):
        pass

    ha_exc.HomeAssistantError = HomeAssistantError
    sys.modules["homeassistant.exceptions"] = ha_exc

    for name in ("const", "config_helpers", "mcp_errors", "mcp_session", "mcp_client"):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    return (
        sys.modules["ha_agent.config_helpers"].McpConfig,
        sys.modules["ha_agent.config_helpers"].default_mcp_health_url,
        sys.modules["ha_agent.mcp_client"].McpProxyClient,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test MCP Proxy")
    parser.add_argument(
        "--mcp-url",
        default=os.environ.get("HA_AGENT_MCP_URL", "http://192.168.10.31:2222/mcp"),
    )
    parser.add_argument(
        "--mcp-token",
        default=os.environ.get("HA_AGENT_MCP_TOKEN", ""),
    )
    parser.add_argument(
        "--tool-name",
        default=os.environ.get("HA_AGENT_MCP_TOOL", "mcp_news__news_curate"),
    )
    args = parser.parse_args()

    McpConfig, default_mcp_health_url, McpProxyClient = _load_mcp_client()
    mcp_url = args.mcp_url.rstrip("/")
    config = McpConfig(
        url=mcp_url,
        bearer_token=args.mcp_token or None,
        timeout=60,
        health_url=default_mcp_health_url(mcp_url),
    )

    async with aiohttp.ClientSession() as session:
        client = McpProxyClient(session, config)
        await client.check_health()
        await client.initialize()
        tools = await client.get_llm_tools()
        print(f"MCP OK: {len(tools)} session tools")

        result = await client.call_tool(args.tool_name, {"limit": 1})
        preview = result if isinstance(result, str) else json.dumps(result)[:300]
        print(f"Tool {args.tool_name} OK: {preview}")

    print("Phase 5 MCP smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as err:
        print(f"Phase 5 MCP smoke test failed: {err}", file=sys.stderr)
        raise SystemExit(1) from err

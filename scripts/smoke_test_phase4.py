#!/usr/bin/env python3
"""Smoke-test Phase 4 backends: LLM models endpoint and MCP initialize/tools."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "ha_agent"


def _load_client_modules():
    """Load ha_agent client modules without Home Assistant installed."""
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

    for name in (
        "const",
        "config_helpers",
        "embedded_tools",
        "mcp_session",
        "mcp_client",
        "llm_client",
    ):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    config_helpers = sys.modules["ha_agent.config_helpers"]
    llm_client = sys.modules["ha_agent.llm_client"]
    mcp_client = sys.modules["ha_agent.mcp_client"]

    return (
        config_helpers.LlmBackend,
        config_helpers.McpConfig,
        config_helpers.default_mcp_health_url,
        llm_client.LlmClient,
        mcp_client.McpProxyClient,
    )


async def _check_llm(
    session: aiohttp.ClientSession,
    backend,
    llm_client_cls,
) -> None:
    client = llm_client_cls(session)
    await client.check_connection(backend)
    url = f"{backend.base_url}/models"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
        response.raise_for_status()
        data = await response.json()
    models = [item.get("id") for item in data.get("data", []) if item.get("id")]
    print(f"LLM OK: {len(models)} models at {backend.base_url}")
    if backend.model not in models:
        print(
            f"WARN: configured model not loaded: {backend.model}",
            file=sys.stderr,
        )
        if models:
            print(f"      available: {models[0]}", file=sys.stderr)


async def _check_mcp(session: aiohttp.ClientSession, config, client_cls) -> None:
    client = client_cls(session, config)
    await client.check_health()
    init_result = await client.initialize()
    tools = await client.get_llm_tools()
    instructions = str(init_result.get("instructions") or "").strip()
    print(f"MCP OK: health {config.health_url}")
    print(f"MCP initialize: {len(instructions)} chars of instructions")
    print(f"MCP tools/list: {len(tools)} session tools")
    tool_names = [
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    ]
    print(f"Session tools: {', '.join(name for name in tool_names if name)}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test HA Agent Phase 4 backends")
    parser.add_argument(
        "--llm-url",
        default=os.environ.get("HA_AGENT_LLM_URL", "http://192.168.10.31:9292/v1"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get(
            "HA_AGENT_LLM_MODEL",
            "unsloth/gemma-4-26B-A4B-it-GGUF:IQ4_XS",
        ),
    )
    parser.add_argument(
        "--mcp-url",
        default=os.environ.get("HA_AGENT_MCP_URL", "http://192.168.10.31:2222/mcp"),
    )
    parser.add_argument(
        "--mcp-token",
        default=os.environ.get("HA_AGENT_MCP_TOKEN", ""),
    )
    args = parser.parse_args()

    (
        LlmBackend,
        McpConfig,
        default_mcp_health_url,
        LlmClient,
        McpProxyClient,
    ) = _load_client_modules()

    backend = LlmBackend(
        base_url=args.llm_url.rstrip("/"),
        model=args.llm_model,
        api_key=os.environ.get("HA_AGENT_LLM_API_KEY") or None,
        max_tokens=256,
        temperature=0.2,
        timeout=60,
        enable_thinking=False,
    )
    mcp_url = args.mcp_url.rstrip("/")
    mcp_config = McpConfig(
        url=mcp_url,
        bearer_token=args.mcp_token or None,
        timeout=60,
        health_url=default_mcp_health_url(mcp_url),
    )

    async with aiohttp.ClientSession() as session:
        await _check_llm(session, backend, LlmClient)
        await _check_mcp(session, mcp_config, McpProxyClient)

    print("Phase 4 smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as err:
        print(f"Phase 4 smoke test failed: {err}", file=sys.stderr)
        raise SystemExit(1) from err

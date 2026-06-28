"""Unit tests for chat API event streaming."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


async def _fake_run_agent(*_args, **_kwargs):
    class Delta:
        def __init__(
            self,
            content=None,
            thinking=None,
            tool=None,
            thinking_clear=False,
            content_clear=False,
            skill=None,
            meta=None,
        ):
            self.content = content
            self.thinking = thinking
            self.tool = tool
            self.thinking_clear = thinking_clear
            self.content_clear = content_clear
            self.skill = skill
            self.meta = meta

    yield Delta(content="Hello")
    yield Delta(
        meta={
            "route": "news",
            "classification": "keyword → news (news: news)",
        }
    )
    yield Delta(
        tool={
            "phase": "start",
            "name": "mcp_news__news_curate",
            "call_name": "callTool",
        }
    )
    yield Delta(
        tool={
            "phase": "done",
            "name": "mcp_news__news_curate",
            "call_name": "callTool",
        }
    )


def _load_chat_module():
    for mod in list(sys.modules):
        if mod == "ha_agent" or mod.startswith("ha_agent."):
            del sys.modules[mod]

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object

        def callback(func):
            return func

        ha_core.callback = callback
        sys.modules["homeassistant.core"] = ha_core

    if "homeassistant.helpers.aiohttp_client" not in sys.modules:
        ha_helpers = types.ModuleType("homeassistant.helpers")
        ha_aiohttp = types.ModuleType("homeassistant.helpers.aiohttp_client")
        ha_aiohttp.async_get_clientsession = lambda _hass: MagicMock()
        sys.modules["homeassistant.helpers"] = ha_helpers
        sys.modules["homeassistant.helpers.aiohttp_client"] = ha_aiohttp

    package = types.ModuleType("ha_agent")
    package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
    sys.modules["ha_agent"] = package

    agent_stub = types.ModuleType("ha_agent.agent")
    agent_stub.run_agent = _fake_run_agent
    sys.modules["ha_agent.agent"] = agent_stub

    conversation_stub = types.ModuleType("ha_agent.conversation")
    conversation_stub.collect_exposed_entities = AsyncMock(return_value=[])
    sys.modules["ha_agent.conversation"] = conversation_stub

    for name in ("const", "config_helpers", "memory", "status", "threads"):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    api_pkg = types.ModuleType("ha_agent.api")
    api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.api"] = api_pkg

    for name in ("helpers", "chat"):
        path = COMPONENT / "api" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.api.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ha_agent.api.{name}"] = module
        spec.loader.exec_module(module)

    return sys.modules["ha_agent.api.chat"]


@pytest.mark.asyncio
async def test_stream_chat_fires_delta_and_done_events() -> None:
    chat = _load_chat_module()

    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.data = {}
    entry.domain = "ha_agent"

    hass = MagicMock()
    hass.data = {}
    hass.bus = MagicMock()
    hass.config_entries = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry

    def create_task(coro, name=None):
        return asyncio.create_task(coro, name=name)

    hass.async_create_task = create_task

    with (
        patch.object(chat, "get_entry", return_value=entry),
        patch.object(
            chat,
            "get_agent_config",
            return_value=MagicMock(history_turns=6, max_iterations=8),
        ),
        patch.object(
            chat,
            "get_llm_backend",
            return_value=MagicMock(timeout=120),
        ),
        patch.object(
            chat,
            "get_mcp_config",
            return_value=MagicMock(timeout=120),
        ),
        patch.object(chat, "get_router_config", return_value=MagicMock()),
        patch.object(chat, "get_skills_config", return_value=MagicMock()),
        patch.object(chat, "collect_exposed_entities", new=AsyncMock(return_value=[])),
        patch.object(chat, "LlmClient", return_value=MagicMock()),
        patch.object(chat, "McpProxyClient", return_value=MagicMock()),
        patch.object(chat, "run_agent", new=_fake_run_agent),
        patch.object(chat, "get_agent_status", return_value={"last_route": "chat"}),
        patch.object(chat, "upsert_thread"),
    ):
        await chat.stream_chat(
            hass,
            entry_id="entry-1",
            conversation_id="conv-1",
            text="hi",
        )

    fired = [call.args for call in hass.bus.async_fire.call_args_list]
    assert fired[0][1]["content"] == "Hello"
    assert fired[1][1]["meta"]["route"] == "news"
    assert fired[2][0] == "ha_agent_chat_delta"
    assert fired[2][1]["tool"]["phase"] == "start"
    assert fired[3][1]["tool"]["phase"] == "done"
    assert fired[4][0] == "ha_agent_chat_done"
    assert fired[4][1]["last_route"] == "chat"
    assert fired[4][1]["turn_meta"]["route"] == "news"
    assert "classification" in fired[4][1]["turn_meta"]

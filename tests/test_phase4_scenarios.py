"""Phase 4 exit-criteria scenario tests for the agent loop."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)

MODULE_DEPS: dict[str, list[str]] = {
    "config_helpers": ["const"],
    "llm_client": ["const", "config_helpers"],
    "mcp_client": ["config_helpers"],
    "context": [],
    "tools": ["llm_client", "mcp_client"],
    "memory": ["const"],
    "agent": [
        "const",
        "config_helpers",
        "llm_client",
        "mcp_client",
        "context",
        "tools",
        "memory",
        "router",
        "status",
    ],
    "router": ["config_helpers", "context"],
    "status": ["const"],
}


def _ensure_ha_stubs() -> None:
    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistantError(Exception):
            pass

        def callback(func):
            return func

        ha_core.HomeAssistant = object
        ha_core.callback = callback
        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc
        sys.modules["homeassistant.core"] = ha_core


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    _ensure_ha_stubs()

    for dep in MODULE_DEPS.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load_module(dep)

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


agent_mod = _load_module("agent")
config_helpers = _load_module("config_helpers")
llm_client = _load_module("llm_client")


def _backend() -> config_helpers.LlmBackend:
    return config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        enable_thinking=False,
    )


def _agent_config(*, streaming: bool = False) -> config_helpers.AgentConfig:
    return config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=6,
        history_turns=4,
        enable_streaming=streaming,
    )


def _router_config() -> config_helpers.RouterConfig:
    return config_helpers.RouterConfig(action_enabled=False, action_backend=None)


def _hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    return hass


def _tool_call(
    name: str,
    arguments: dict,
    *,
    call_id: str = "call_1",
) -> llm_client.ToolCall:
    return llm_client.ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )


def _chat_result(
    *,
    content: str | None = None,
    tool_calls: list[llm_client.ToolCall] | None = None,
) -> llm_client.ChatResult:
    tool_calls = tool_calls or []
    assistant_message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ]
    return llm_client.ChatResult(
        content=content,
        tool_calls=tool_calls,
        assistant_message=assistant_message,
    )


@pytest.mark.asyncio
async def test_phase4_light_off_with_exposed_entity() -> None:
    """Exposed light can be turned off with one MCP service call."""
    service_call = _tool_call(
        "callTool",
        {
            "toolName": "home_assistant__ha_call_service",
            "arguments": {
                "domain": "light",
                "service": "turn_off",
                "entity_id": "light.dining",
            },
        },
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            _chat_result(tool_calls=[service_call]),
            _chat_result(content="The dining room lights are off."),
        ]
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"success": true}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-light",
            user_text="turn off the dining room lights",
            exposed_entities=[
                {
                    "entity_id": "light.dining",
                    "name": "Dining",
                    "state": "on",
                    "area_name": "Dining room",
                }
            ],
        )
    ]

    assert chunks == ["The dining room lights are off."]
    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args.args
    assert call_args[0] == "callTool"
    assert call_args[1]["toolName"] == "home_assistant__ha_call_service"


@pytest.mark.asyncio
async def test_phase4_cover_open_without_exposed_entity() -> None:
    """Cover actions can search smart-home tools then call open_cover."""
    search_call = _tool_call(
        "searchToolsForDomain",
        {"domain": "smart-home", "query": "open cover"},
        call_id="call_search",
    )
    open_call = _tool_call(
        "callTool",
        {
            "toolName": "home_assistant__ha_call_service",
            "arguments": {
                "domain": "cover",
                "service": "open_cover",
                "entity_id": "cover.patio",
            },
        },
        call_id="call_open",
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            _chat_result(tool_calls=[search_call]),
            _chat_result(tool_calls=[open_call]),
            _chat_result(content="I opened the patio cover."),
        ]
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(
        side_effect=[
            '{"tools":[{"toolName":"home_assistant__ha_call_service"}]}',
            '{"success": true}',
        ]
    )
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-cover",
            user_text="open the patio cover",
            exposed_entities=[],
        )
    ]

    assert chunks == ["I opened the patio cover."]
    assert mock_mcp.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_phase4_news_query_uses_mcp_tool() -> None:
    """News questions execute MCP news tools before answering."""
    news_call = _tool_call(
        "callTool",
        {"toolName": "mcp_news__news_curate", "arguments": {"limit": 5}},
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            _chat_result(tool_calls=[news_call]),
            _chat_result(content="Here are today's headlines."),
        ]
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"headlines":["Example headline"]}')
    mock_mcp.get_session_prompt = AsyncMock(
        return_value="MCP SERVER INSTRUCTIONS:\nUse domain news."
    )
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-news",
            user_text="What's the news?",
            exposed_entities=[],
        )
    ]

    assert chunks == ["Here are today's headlines."]
    mock_mcp.call_tool.assert_awaited_once()
    assert mock_mcp.call_tool.await_args.args[1]["toolName"] == "mcp_news__news_curate"


@pytest.mark.asyncio
async def test_phase4_email_unread_count_uses_mcp_tool() -> None:
    """Email questions execute MCP mail tools before answering."""
    mail_call = _tool_call(
        "callTool",
        {
            "toolName": "mail_mcp__imap_search_messages",
            "arguments": {"mailbox": "INBOX", "unread_only": True},
        },
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            _chat_result(tool_calls=[mail_call]),
            _chat_result(content="You have 3 unread emails."),
        ]
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"count": 3}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-email",
            user_text="how many unread emails do I have",
            exposed_entities=[],
        )
    ]

    assert chunks == ["You have 3 unread emails."]
    mock_mcp.call_tool.assert_awaited_once()
    assert (
        mock_mcp.call_tool.await_args.args[1]["toolName"]
        == "mail_mcp__imap_search_messages"
    )


@pytest.mark.asyncio
async def test_phase4_conversation_memory_across_turns() -> None:
    """Second turn includes prior user and assistant messages."""
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=_chat_result(content="Still sunny."))
    mock_mcp = MagicMock()
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    hass = _hass()
    backend = _backend()
    agent_config = _agent_config()

    async def _run(user_text: str) -> None:
        async for _chunk in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=_router_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-memory",
            user_text=user_text,
            exposed_entities=[],
        ):
            pass

    await _run("what is the weather")
    await _run("and tomorrow")

    second_messages = mock_llm.chat.await_args_list[1].args[0]
    roles = [message["role"] for message in second_messages]
    contents = [message.get("content", "") for message in second_messages]

    assert roles.count("user") == 2
    assert roles.count("assistant") == 1
    assert "what is the weather" in contents
    assert "Still sunny." in contents
    assert second_messages[-1]["content"] == "and tomorrow"

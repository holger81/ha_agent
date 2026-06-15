"""Unit tests for the agent tool loop."""

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
        "embedded_tools",
        "mcp_session",
        "mcp_errors",
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

        class ServiceCall:
            def __init__(self, data: dict | None = None) -> None:
                self.data = data or {}

        ha_core.ServiceCall = ServiceCall
        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc
        sys.modules["homeassistant.core"] = ha_core

    if "homeassistant.components.conversation" not in sys.modules:
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = types.ModuleType(
            "homeassistant.components.conversation"
        )


def _load_skills_modules() -> None:
    """Load skills subpackage modules for agent imports."""
    skills_path = COMPONENT / "skills"
    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(skills_path)]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in (
        "models",
        "store",
        "format",
        "discovery",
        "runtime",
        "creator",
        "evaluator",
        "commands",
    ):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = skills_path / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


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

    if name == "agent":
        _load_skills_modules()

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

DEFAULT_SKILLS_CONFIG = config_helpers.SkillsConfig(
    learning_enabled=False,
    auto_save=False,
    use_enabled=False,
    max_inject=3,
)


def _mock_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.async_add_executor_job = AsyncMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )
    hass.async_create_task = MagicMock()
    return hass


@pytest.mark.asyncio
async def test_run_agent_executes_tool_then_replies() -> None:
    """Agent loop executes MCP tool before final answer."""
    tool_call = llm_client.ToolCall(
        id="call_1",
        name="callTool",
        arguments=json.dumps(
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.dining",
                },
            }
        ),
    )
    first = llm_client.ChatResult(
        content=None,
        tool_calls=[tool_call],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1"}],
        },
    )
    second = llm_client.ChatResult(content="Done.", tool_calls=[])

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(side_effect=[first, second])

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"success": true}')
    mock_mcp.get_session_prompt = AsyncMock(
        return_value="MCP SERVER INSTRUCTIONS:\nUse searchToolsForDomain."
    )
    mock_mcp.get_llm_tools = AsyncMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "callTool",
                    "description": "Execute upstream tools",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=False,
        show_reasoning_in_chat=False,
    )
    router_config = config_helpers.RouterConfig(
        action_enabled=False,
        action_backend=None,
    )

    hass = _mock_hass()

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=router_config,
            skills_config=DEFAULT_SKILLS_CONFIG,
            entry_id="test-entry",
            conversation_id="test-conv",
            user_text="turn off dining room lights",
            exposed_entities=[
                {
                    "entity_id": "light.dining",
                    "name": "Dining",
                    "state": "on",
                }
            ],
        )
    ]

    assert _agent_content(chunks) == ["Done."]
    assert mock_llm.chat.await_count == 2
    mock_mcp.call_tool.assert_awaited_once()


def _agent_content(deltas: list) -> list[str]:
    return [delta.content for delta in deltas if delta.content]


def _make_stream(text: str):
    async def _stream(*_args, **_kwargs):
        for char in text:
            yield llm_client.StreamChunk(content=char)

    return _stream


def _make_incremental_stream(parts: list[str]):
    async def _stream(*_args, **_kwargs):
        for part in parts:
            yield llm_client.StreamChunk(content=part)

    return _stream


@pytest.mark.asyncio
async def test_run_agent_yields_stream_deltas_to_assist() -> None:
    """Streaming mode forwards LLM deltas instead of one buffered reply."""
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock()
    mock_llm.chat_stream = _make_stream("Hello there.")

    mock_mcp = MagicMock()
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=True,
        show_reasoning_in_chat=False,
    )
    router_config = config_helpers.RouterConfig(
        action_enabled=False,
        action_backend=None,
    )

    hass = _mock_hass()

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=router_config,
            skills_config=DEFAULT_SKILLS_CONFIG,
            entry_id="test-entry",
            conversation_id="test-conv",
            user_text="hi",
            exposed_entities=[],
        )
    ]

    assert "".join(_agent_content(chunks)) == "Hello there."
    assert len(chunks) > 1
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_streams_chunks_as_they_arrive() -> None:
    """Assist receives partial text before the LLM stream finishes."""
    mock_llm = MagicMock()
    mock_llm.chat_stream = _make_incremental_stream(["Hel", "lo ", "there."])

    mock_mcp = MagicMock()
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=True,
        show_reasoning_in_chat=False,
    )

    hass = _mock_hass()

    chunks: list = []
    async for chunk in agent_mod.run_agent(
        hass,
        llm=mock_llm,
        mcp_client=mock_mcp,
        backend=backend,
        agent_config=agent_config,
        router_config=config_helpers.RouterConfig(
            action_enabled=False,
            action_backend=None,
        ),
        skills_config=DEFAULT_SKILLS_CONFIG,
        entry_id="test-entry",
        conversation_id="test-conv",
        user_text="hi",
        exposed_entities=[],
    ):
        chunks.append(chunk)

    assert _agent_content(chunks) == ["Hel", "lo", " there."]
    assert "".join(_agent_content(chunks)) == "Hello there."


@pytest.mark.asyncio
async def test_run_agent_executes_embedded_stream_tool_call() -> None:
    """Embedded tool markup in a stream is executed, not shown to the user."""
    tool_markup = (
        '<|tool_call|>call:home_assistant__ha_search_entities'
        '{query:<|"|>camera snapshot<|"|>}<tool_call|>'
    )
    stream_texts = iter([tool_markup, "Found the camera."])

    mock_llm = MagicMock()
    mock_llm.chat_stream = MagicMock(
        side_effect=lambda *_args, **_kwargs: _make_stream(next(stream_texts))()
    )

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"entities": []}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=True,
        show_reasoning_in_chat=False,
    )
    router_config = config_helpers.RouterConfig(
        action_enabled=False,
        action_backend=None,
    )

    hass = _mock_hass()

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=router_config,
            skills_config=DEFAULT_SKILLS_CONFIG,
            entry_id="test-entry",
            conversation_id="test-conv",
            user_text="snapshot the front door",
            exposed_entities=[],
        )
    ]

    assert tool_markup not in "".join(_agent_content(chunks))
    assert "".join(_agent_content(chunks)) == "Found the camera."
    mock_mcp.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_streams_reasoning_separately() -> None:
    """Reasoning chunks are yielded as thinking, not spoken content."""
    async def _stream(*_args, **_kwargs):
        yield llm_client.StreamChunk(reasoning_content="Thinking…")
        yield llm_client.StreamChunk(content="Hello.")

    mock_llm = MagicMock()
    mock_llm.chat_stream = _stream
    mock_mcp = MagicMock()
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="medium",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=True,
        show_reasoning_in_chat=True,
    )

    hass = _mock_hass()
    deltas = [
        delta
        async for delta in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=config_helpers.RouterConfig(
                action_enabled=False,
                action_backend=None,
            ),
            skills_config=DEFAULT_SKILLS_CONFIG,
            entry_id="test-entry",
            conversation_id="test-conv",
            user_text="hi",
            exposed_entities=[],
        )
    ]

    assert [delta.thinking for delta in deltas if delta.thinking] == ["Thinking…"]
    assert "".join(_agent_content(deltas)) == "Hello."


@pytest.mark.asyncio
async def test_run_agent_shows_tool_progress_in_chat() -> None:
    """Tool calls emit start and completion lines in the thinking panel."""
    tool_call = llm_client.ToolCall(
        id="call_1",
        name="callTool",
        arguments=(
            '{"toolName":"home_assistant__ha_call_service",'
            '"arguments":{"domain":"light","service":"turn_off",'
            '"entity_id":"light.dining"}}'
        ),
    )
    first = llm_client.ChatResult(
        content=None,
        tool_calls=[tool_call],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1"}],
        },
    )
    second = llm_client.ChatResult(content="Done.", tool_calls=[])

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(side_effect=[first, second])
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"success": true}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )
    agent_config = config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=False,
        show_reasoning_in_chat=True,
    )

    hass = _mock_hass()
    deltas = [
        delta
        async for delta in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=config_helpers.RouterConfig(
                action_enabled=False,
                action_backend=None,
            ),
            skills_config=DEFAULT_SKILLS_CONFIG,
            entry_id="test-entry",
            conversation_id="test-conv",
            user_text="turn off dining room lights",
            exposed_entities=[
                {
                    "entity_id": "light.dining",
                    "name": "Dining",
                    "state": "on",
                }
            ],
        )
    ]

    thinking = [delta.thinking for delta in deltas if delta.thinking]
    assert thinking[0] == "Calling turn_off on light.dining…\n"
    assert thinking[1] == "home_assistant__ha_call_service done\n"
    assert _agent_content(deltas) == ["Done."]

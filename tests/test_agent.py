"""Unit tests for the agent tool loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ha_agent import AgentConfig, ConversationMemory, LlmBackend, run_agent
from ha_agent.llm_client import ChatResult, LlmClient, ToolCall


@pytest.mark.asyncio
async def test_run_agent_executes_tool_then_replies() -> None:
    """Agent loop executes MCP tool before final answer."""
    tool_call = ToolCall(
        id="call_1",
        name="mcp_call_tool",
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
    first = ChatResult(
        content=None,
        tool_calls=[tool_call],
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1"}],
        },
    )
    second = ChatResult(content="Done.", tool_calls=[])

    mock_llm = MagicMock(spec=LlmClient)
    mock_llm.chat = AsyncMock(side_effect=[first, second])

    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"success": true}')

    backend = LlmBackend(
        base_url="http://example/v1",
        model="test",
        max_tokens=128,
        temperature=0.2,
        timeout=30,
    )
    agent_config = AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=4,
        history_turns=2,
        enable_streaming=False,
    )
    memory = ConversationMemory()

    chunks = [
        chunk
        async for chunk in run_agent(
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            conversation_id="test-conv",
            user_text="turn off dining room lights",
            exposed_entities=[
                {
                    "entity_id": "light.dining",
                    "name": "Dining",
                    "state": "on",
                }
            ],
            memory=memory,
        )
    ]

    assert chunks == ["Done."]
    assert mock_llm.chat.await_count == 2
    mock_mcp.call_tool.assert_awaited_once()
    assert memory.get_history("test-conv", max_turns=2)

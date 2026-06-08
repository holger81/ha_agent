"""Agent loop with MCP tool calling."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from .config_helpers import AgentConfig, LlmBackend
from .context import build_messages, build_system_message, build_tool_context
from .embedded_tools import is_tool_call_only_text, strip_embedded_tool_markup
from .llm_client import MCP_CALL_TOOL_SCHEMA, LlmClient
from .memory import append_turn, get_history
from .tools import execute_tool, tool_result_message

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

FALLBACK_MESSAGE = "Sorry, I couldn't complete that request."


async def run_agent(
    hass: HomeAssistant,
    *,
    llm: LlmClient,
    mcp_client: McpProxyClient,
    backend: LlmBackend,
    agent_config: AgentConfig,
    conversation_id: str | None,
    user_text: str,
    exposed_entities: list[dict[str, Any]],
    extra_system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run the tool loop and yield assistant text deltas."""
    tool_context = build_tool_context(user_text, exposed_entities)
    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        tool_context=tool_context,
        extra_system_prompt=extra_system_prompt,
    )
    history = get_history(
        hass,
        conversation_id,
        max_turns=agent_config.history_turns,
    )
    messages = build_messages(
        system_message=system_message,
        history=history,
        user_text=user_text,
    )
    tools = [MCP_CALL_TOOL_SCHEMA]
    collected: list[str] = []

    for _ in range(agent_config.max_iterations):
        result = await llm.chat(messages, backend, tools=tools)
        if result.tool_calls:
            messages.append(result.assistant_message)
            for call in result.tool_calls:
                output = await execute_tool(mcp_client, call)
                messages.append(tool_result_message(call, output))
            continue

        if agent_config.enable_streaming:
            async for delta in llm.chat_stream(messages, backend):
                collected.append(delta)
            assistant_text = strip_embedded_tool_markup("".join(collected))
            if assistant_text:
                yield assistant_text
        else:
            assistant_text = (result.content or "").strip()
            if assistant_text and not is_tool_call_only_text(assistant_text):
                yield assistant_text
            elif assistant_text:
                assistant_text = ""

        append_turn(
            hass,
            conversation_id,
            user_text,
            assistant_text,
            max_turns=agent_config.history_turns,
        )
        return

    fallback = FALLBACK_MESSAGE
    yield fallback
    append_turn(
        hass,
        conversation_id,
        user_text,
        fallback,
        max_turns=agent_config.history_turns,
    )

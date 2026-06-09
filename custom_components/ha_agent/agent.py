"""Agent loop with MCP tool calling."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from .config_helpers import AgentConfig, LlmBackend
from .const import LOGGER
from .context import build_messages, build_system_message, build_tool_context
from .embedded_tools import (
    is_tool_call_only_text,
    parse_embedded_tool_calls,
    safe_stream_display_text,
    strip_embedded_tool_markup,
)
from .llm_client import LlmClient, StreamChatSession, ToolCall
from .mcp_session import FALLBACK_MCP_TOOLS, mcp_tools_to_openai_schemas
from .memory import append_turn, get_history
from .tools import execute_tool, tool_result_message

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

FALLBACK_MESSAGE = "Sorry, I couldn't complete that request."


async def _yield_streamed_assistant_text(
    llm: LlmClient,
    messages: list[dict[str, Any]],
    backend: LlmBackend,
    tools: list[dict[str, Any]],
) -> AsyncGenerator[tuple[str, StreamChatSession], None]:
    """Stream assistant text to Assist while accumulating the raw response."""
    session = StreamChatSession()
    raw_buffer = ""
    yielded_len = 0

    async for delta in llm.chat_stream(
        messages,
        backend,
        tools=tools,
        session=session,
    ):
        raw_buffer += delta
        safe = safe_stream_display_text(raw_buffer)
        if len(safe) > yielded_len:
            chunk = safe[yielded_len:]
            yielded_len = len(safe)
            yield chunk, session

    session.content = raw_buffer
    assistant_text = strip_embedded_tool_markup(raw_buffer)
    if len(assistant_text) > yielded_len:
        yield assistant_text[yielded_len:], session
    elif not yielded_len:
        yield "", session


async def _execute_embedded_tool_calls(
    mcp_client: McpProxyClient,
    content: str,
    messages: list[dict[str, Any]],
) -> bool:
    """Parse and run embedded tool calls from model text. Return True if any ran."""
    embedded = parse_embedded_tool_calls(content)
    if not embedded:
        return False

    messages.append({"role": "assistant", "content": content})
    for index, call in enumerate(embedded):
        tool_call = ToolCall(
            id=call.id or f"call_embedded_{index}",
            name=call.name,
            arguments=call.arguments,
        )
        output = await execute_tool(mcp_client, tool_call)
        messages.append(tool_result_message(tool_call, output))
    return True


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
    mcp_session_prompt = ""
    llm_tools = mcp_tools_to_openai_schemas(FALLBACK_MCP_TOOLS)
    try:
        mcp_session_prompt = await mcp_client.get_session_prompt()
        llm_tools = await mcp_client.get_llm_tools()
    except Exception as err:
        LOGGER.warning("Failed to load MCP session: %s", err)

    tool_context = build_tool_context(user_text, exposed_entities)
    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        mcp_session_prompt=mcp_session_prompt,
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
    tools = llm_tools

    for _ in range(agent_config.max_iterations):
        if agent_config.enable_streaming:
            session = StreamChatSession()
            async for chunk, active_session in _yield_streamed_assistant_text(
                llm,
                messages,
                backend,
                tools,
            ):
                session = active_session
                if chunk:
                    yield chunk

            raw_buffer = session.content
            if session.tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in session.tool_calls
                    ],
                }
                messages.append(assistant_message)
                for call in session.tool_calls:
                    output = await execute_tool(mcp_client, call)
                    messages.append(tool_result_message(call, output))
                continue

            if await _execute_embedded_tool_calls(
                mcp_client,
                raw_buffer.strip(),
                messages,
            ):
                continue

            assistant_text = strip_embedded_tool_markup(raw_buffer)
            if not assistant_text and is_tool_call_only_text(raw_buffer):
                assistant_text = ""
        else:
            result = await llm.chat(messages, backend, tools=tools)
            if result.tool_calls:
                messages.append(result.assistant_message)
                for call in result.tool_calls:
                    output = await execute_tool(mcp_client, call)
                    messages.append(tool_result_message(call, output))
                continue

            if await _execute_embedded_tool_calls(
                mcp_client,
                (result.content or "").strip(),
                messages,
            ):
                continue

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

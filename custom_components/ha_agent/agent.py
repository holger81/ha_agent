"""Agent loop with MCP tool calling."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from .config_helpers import AgentConfig, LlmBackend, RouterConfig
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
from .router import TaskRoute, backend_for_route, classify_route
from .status import record_route, update_agent_status
from .tools import execute_tool, tool_result_message

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

FALLBACK_MESSAGE = "Sorry, I couldn't complete that request."


async def _yield_buffered_text(text: str) -> AsyncGenerator[str, None]:
    """Yield a completed reply in chunks for Assist without a second LLM call."""
    if not text:
        return
    for index, word in enumerate(text.split(" ")):
        if index == 0:
            yield word
        else:
            yield f" {word}"


async def _yield_streamed_assistant_text(
    llm: LlmClient,
    messages: list[dict[str, Any]],
    backend: LlmBackend,
    *,
    tools: list[dict[str, Any]] | None = None,
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
    router_config: RouterConfig,
    entry_id: str,
    conversation_id: str | None,
    user_text: str,
    exposed_entities: list[dict[str, Any]],
    extra_system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run the tool loop and yield assistant text deltas."""
    route = classify_route(user_text, exposed_entities, router_config)
    record_route(hass, entry_id, route)

    mcp_session_prompt = ""
    llm_tools = mcp_tools_to_openai_schemas(FALLBACK_MCP_TOOLS)
    try:
        mcp_session_prompt = await mcp_client.get_session_prompt()
        llm_tools = await mcp_client.get_llm_tools()
        update_agent_status(
            hass,
            entry_id,
            mcp_tool_count=len(llm_tools),
            mcp_reachable=True,
            last_error=None,
        )
    except Exception as err:
        LOGGER.warning("Failed to load MCP session: %s", err)
        update_agent_status(
            hass,
            entry_id,
            mcp_reachable=False,
            last_error=str(err),
        )

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
    use_chat_backend = route != TaskRoute.HA_ACTION

    for _ in range(agent_config.max_iterations):
        active_backend = (
            backend
            if use_chat_backend
            else backend_for_route(
                route,
                chat_backend=backend,
                router_config=router_config,
            )
        )

        result = await llm.chat(messages, active_backend, tools=tools)
        if result.tool_calls:
            messages.append(result.assistant_message)
            for call in result.tool_calls:
                output = await execute_tool(mcp_client, call)
                messages.append(tool_result_message(call, output))
            use_chat_backend = True
            continue

        if await _execute_embedded_tool_calls(
            mcp_client,
            (result.content or "").strip(),
            messages,
        ):
            use_chat_backend = True
            continue

        assistant_text = strip_embedded_tool_markup((result.content or "").strip())
        if is_tool_call_only_text(result.content):
            assistant_text = ""

        if agent_config.enable_streaming:
            if assistant_text:
                async for chunk in _yield_buffered_text(assistant_text):
                    yield chunk
            else:
                session = StreamChatSession()
                async for chunk, active_session in _yield_streamed_assistant_text(
                    llm,
                    messages,
                    active_backend,
                    tools=None,
                ):
                    session = active_session
                    if chunk:
                        yield chunk
                assistant_text = strip_embedded_tool_markup(session.content)
                if await _execute_embedded_tool_calls(
                    mcp_client,
                    session.content.strip(),
                    messages,
                ):
                    use_chat_backend = True
                    continue
        elif assistant_text:
            yield assistant_text

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

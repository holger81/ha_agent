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
from .tools import (
    execute_tool,
    ha_service_entity_id,
    memory_assistant_text,
    tool_result_message,
)

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


async def _run_tool_call(
    mcp_client: McpProxyClient,
    call: ToolCall,
    *,
    exposed_entities: list[dict[str, Any]],
    controlled_entity_ids: list[str],
) -> str:
    """Execute one tool call and track controlled homeassistant entities."""
    output = await execute_tool(
        mcp_client,
        call,
        exposed_entities=exposed_entities,
    )
    if not output.startswith("Tool error:") and (
        entity_id := ha_service_entity_id(
            call,
            exposed_entities=exposed_entities,
        )
    ):
        controlled_entity_ids.append(entity_id)
    return output


async def _execute_embedded_tool_calls(
    mcp_client: McpProxyClient,
    content: str,
    messages: list[dict[str, Any]],
    *,
    exposed_entities: list[dict[str, Any]],
    controlled_entity_ids: list[str],
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
        output = await _run_tool_call(
            mcp_client,
            tool_call,
            exposed_entities=exposed_entities,
            controlled_entity_ids=controlled_entity_ids,
        )
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

    history = get_history(
        hass,
        conversation_id,
        max_turns=agent_config.history_turns,
    )
    tool_context = build_tool_context(
        user_text,
        exposed_entities,
        history=history,
    )
    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        mcp_session_prompt=mcp_session_prompt,
        tool_context=tool_context,
        extra_system_prompt=extra_system_prompt,
    )
    messages = build_messages(
        system_message=system_message,
        history=history,
        user_text=user_text,
    )
    tools = llm_tools
    use_chat_backend = route != TaskRoute.HA_ACTION
    controlled_entity_ids: list[str] = []

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

        if agent_config.enable_streaming:
            session = StreamChatSession()
            async for chunk, active_session in _yield_streamed_assistant_text(
                llm,
                messages,
                active_backend,
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
                    output = await _run_tool_call(
                        mcp_client,
                        call,
                        exposed_entities=exposed_entities,
                        controlled_entity_ids=controlled_entity_ids,
                    )
                    messages.append(tool_result_message(call, output))
                use_chat_backend = True
                continue

            if await _execute_embedded_tool_calls(
                mcp_client,
                raw_buffer.strip(),
                messages,
                exposed_entities=exposed_entities,
                controlled_entity_ids=controlled_entity_ids,
            ):
                use_chat_backend = True
                continue

            assistant_text = strip_embedded_tool_markup(raw_buffer)
            if not assistant_text and is_tool_call_only_text(raw_buffer):
                assistant_text = ""
        else:
            result = await llm.chat(messages, active_backend, tools=tools)
            if result.tool_calls:
                messages.append(result.assistant_message)
                for call in result.tool_calls:
                    output = await _run_tool_call(
                        mcp_client,
                        call,
                        exposed_entities=exposed_entities,
                        controlled_entity_ids=controlled_entity_ids,
                    )
                    messages.append(tool_result_message(call, output))
                use_chat_backend = True
                continue

            if await _execute_embedded_tool_calls(
                mcp_client,
                (result.content or "").strip(),
                messages,
                exposed_entities=exposed_entities,
                controlled_entity_ids=controlled_entity_ids,
            ):
                use_chat_backend = True
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
            memory_assistant_text(assistant_text, controlled_entity_ids),
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

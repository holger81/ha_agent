"""Agent loop with MCP tool calling."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from .activity import record_turn
from .config_helpers import AgentConfig, LlmBackend, RouterConfig, SkillsConfig
from .const import LOGGER
from .context import (
    build_messages,
    build_system_message,
    build_tool_context,
    is_affirmative,
)
from .embedded_tools import (
    is_tool_call_only_text,
    parse_embedded_tool_calls,
    safe_stream_display_text,
    strip_embedded_tool_markup,
)
from .llm_client import LlmClient, StreamChatSession, ToolCall, stream_text_delta
from .loop_policy import LoopState, TurnOutcome, check_stuck, finalize_output
from .mcp_session import FALLBACK_MCP_TOOLS, mcp_tools_to_openai_schemas
from .memory import append_turn, get_history
from .router import TaskRoute, backend_for_route, classify_route, route_playbook
from .skills.commands import (
    _MANUAL_SAVE,
    is_skill_admin_query,
    queue_pending_save,
    try_confirm_pending_save,
    try_handle_skill_command,
)
from .skills.creator import create_skill_from_trace
from .skills.discovery import build_skill_hints, discover_skills
from .skills.evaluator import evaluate_skill_use
from .skills.learning_gate import assess_skill_worth_learning
from .skills.models import TurnTrace
from .skills.runtime import should_offer_skill_creation
from .skills.store import get_skill_store
from .status import record_route, update_agent_status
from .tools import (
    execute_tool,
    ha_service_entity_id,
    memory_assistant_text,
    parse_tool_arguments,
    tool_result_message,
)

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

FALLBACK_MESSAGE = "Sorry, I couldn't complete that request."


@dataclass(slots=True, frozen=True)
class AgentDelta:
    """One streamed update for the Assist chat log."""

    content: str = ""
    thinking: str = ""
    tool: dict[str, Any] | None = None
    thinking_clear: bool = False


def _tool_call_payload(call: ToolCall) -> tuple[str, dict[str, Any]]:
    """Return MCP tool name and argument object from a tool call."""
    args = parse_tool_arguments(call.arguments)
    inner = args.get("arguments")
    if isinstance(inner, dict):
        return str(args.get("toolName") or call.name), inner
    return str(args.get("toolName") or call.name), args


def _tool_display_name(call: ToolCall) -> str:
    tool_name, _arguments = _tool_call_payload(call)
    return tool_name


def _tool_event(call: ToolCall, phase: str, *, detail: str = "") -> dict[str, Any]:
    """Build a structured tool progress event for the console."""
    name, arguments = _tool_call_payload(call)
    event: dict[str, Any] = {
        "phase": phase,
        "name": name,
        "call_name": call.name,
    }
    if arguments:
        event["arguments"] = arguments
    if detail:
        event["detail"] = detail
    return event


def _tool_event_detail(call: ToolCall, output: str) -> str:
    """Return a short result summary for a completed tool call."""
    if output.startswith("Tool error:"):
        return output.removeprefix("Tool error:").strip()[:200]
    compact = output.replace("\n", " ").strip()
    return compact[:160]


def thinking_from_tool_event(tool: dict[str, Any]) -> str:
    """Map a structured tool event to Assist thinking text."""
    name = str(tool.get("name") or tool.get("call_name") or "tool")
    phase = tool.get("phase")
    arguments = tool.get("arguments")
    if phase == "start":
        if isinstance(arguments, dict) and "ha_call_service" in name:
            service = arguments.get("service", "service")
            entity = arguments.get("entity_id", "entity")
            return f"Calling {service} on {entity}…\n"
        if isinstance(arguments, dict) and (query := arguments.get("query")):
            return f"Calling {name} ({query})…\n"
        return f"Calling {name}…\n"
    if phase == "error":
        detail = str(tool.get("detail") or "error").strip()
        if len(detail) > 80:
            detail = f"{detail[:77]}..."
        return f"{name} failed: {detail}\n" if detail else f"{name} failed\n"
    if phase == "done":
        return f"{name} done\n"
    return ""


def _record_tool_call(trace: TurnTrace, call: ToolCall, output: str) -> None:
    """Append a tool call to the turn trace."""
    tool_name, arguments = _tool_call_payload(call)
    trace.tool_calls.append(
        {
            "toolName": tool_name,
            "name": tool_name,
            "arguments": arguments,
        }
    )
    if output.startswith("Tool error:"):
        trace.tool_errors += 1


async def _yield_streamed_assistant_text(
    llm: LlmClient,
    messages: list[dict[str, Any]],
    backend: LlmBackend,
    tools: list[dict[str, Any]],
    *,
    show_reasoning: bool,
) -> AsyncGenerator[tuple[AgentDelta, StreamChatSession], None]:
    """Stream assistant text to Assist while accumulating the raw response."""
    session = StreamChatSession()
    raw_buffer = ""
    yielded_len = 0
    reasoning_buffer = ""
    reasoning_yielded_len = 0

    async for chunk in llm.chat_stream(
        messages,
        backend,
        tools=tools,
        session=session,
    ):
        if show_reasoning and chunk.reasoning_content:
            reasoning_buffer, _ = stream_text_delta(
                reasoning_buffer,
                chunk.reasoning_content,
            )
            if len(reasoning_buffer) > reasoning_yielded_len:
                text = reasoning_buffer[reasoning_yielded_len:]
                reasoning_yielded_len = len(reasoning_buffer)
                if text:
                    yield AgentDelta(thinking=text), session
        if not chunk.content:
            continue
        raw_buffer += chunk.content
        safe = safe_stream_display_text(raw_buffer)
        if len(safe) > yielded_len:
            text = safe[yielded_len:]
            yielded_len = len(safe)
            yield AgentDelta(content=text), session

    session.content = raw_buffer
    assistant_text = strip_embedded_tool_markup(raw_buffer)
    if len(assistant_text) > yielded_len:
        yield AgentDelta(content=assistant_text[yielded_len:]), session
    elif not yielded_len:
        yield AgentDelta(), session


async def _run_tool_call(
    mcp_client: McpProxyClient,
    call: ToolCall,
    *,
    exposed_entities: list[dict[str, Any]],
    controlled_entity_ids: list[str],
    trace: TurnTrace | None = None,
) -> str:
    """Execute one tool call and track controlled homeassistant entities."""
    output = await execute_tool(
        mcp_client,
        call,
        exposed_entities=exposed_entities,
    )
    if trace is not None:
        _record_tool_call(trace, call, output)
    if not output.startswith("Tool error:") and (
        entity_id := ha_service_entity_id(
            call,
            exposed_entities=exposed_entities,
        )
    ):
        controlled_entity_ids.append(entity_id)
    return output


def _finalize_stuck_turn(trace: TurnTrace, loop_state: LoopState) -> str:
    """Apply terminal fields when the loop detects a repeated tool call."""
    trace.outcome = TurnOutcome.STUCK
    trace.assistant_text = loop_state.stuck_message
    trace.verification_notes = list(loop_state.verification_notes)
    return loop_state.stuck_message


async def _process_tool_calls(
    agent_config: AgentConfig,
    calls: list[ToolCall],
    mcp_client: McpProxyClient,
    messages: list[dict[str, Any]],
    *,
    hass: HomeAssistant,
    exposed_entities: list[dict[str, Any]],
    controlled_entity_ids: list[str],
    loop_state: LoopState,
    trace: TurnTrace | None = None,
) -> AsyncGenerator[AgentDelta, None]:
    """Run tool calls and yield chat progress deltas."""
    for call in calls:
        tool_name, arguments = _tool_call_payload(call)
        if stuck_msg := check_stuck(loop_state, tool_name, arguments):
            yield AgentDelta(tool=_tool_event(call, "error", detail=stuck_msg[:200]))
            messages.append(
                tool_result_message(call, f"Tool error: {stuck_msg}")
            )
            if trace is not None:
                _record_tool_call(trace, call, f"Tool error: {stuck_msg}")
            continue

        yield AgentDelta(tool=_tool_event(call, "start"))
        raw_output = await _run_tool_call(
            mcp_client,
            call,
            exposed_entities=exposed_entities,
            controlled_entity_ids=controlled_entity_ids,
            trace=trace,
        )
        output = finalize_output(
            tool_name,
            arguments,
            raw_output,
            hass=hass,
            loop_state=loop_state,
        )
        phase = "error" if output.startswith("Tool error:") else "done"
        yield AgentDelta(
            tool=_tool_event(
                call,
                phase,
                detail=_tool_event_detail(call, output),
            )
        )
        messages.append(tool_result_message(call, output))


async def _process_embedded_tool_calls(
    agent_config: AgentConfig,
    mcp_client: McpProxyClient,
    content: str,
    messages: list[dict[str, Any]],
    *,
    hass: HomeAssistant,
    exposed_entities: list[dict[str, Any]],
    controlled_entity_ids: list[str],
    loop_state: LoopState,
    trace: TurnTrace | None = None,
) -> AsyncGenerator[AgentDelta, None]:
    """Parse embedded tool markup, run tools, and yield progress deltas."""
    embedded = parse_embedded_tool_calls(content)
    if not embedded:
        return

    messages.append({"role": "assistant", "content": content})
    calls = [
        ToolCall(
            id=call.id or f"call_embedded_{index}",
            name=call.name,
            arguments=call.arguments,
        )
        for index, call in enumerate(embedded)
    ]
    async for delta in _process_tool_calls(
        agent_config,
        calls,
        mcp_client,
        messages,
        hass=hass,
        exposed_entities=exposed_entities,
        controlled_entity_ids=controlled_entity_ids,
        loop_state=loop_state,
        trace=trace,
    ):
        yield delta


async def _update_skill_status(hass: HomeAssistant, entry_id: str) -> None:
    """Refresh skill diagnostic counters."""
    store = get_skill_store(hass, entry_id)

    def _counts() -> tuple[int, int]:
        return store.count_skills(), store.count_skills(enabled_only=True)

    total, enabled = await hass.async_add_executor_job(_counts)
    update_agent_status(
        hass,
        entry_id,
        skills_total=total,
        skills_enabled=enabled,
    )


async def _post_turn_skills(
    hass: HomeAssistant,
    *,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    skills_config: SkillsConfig,
    trace: TurnTrace,
    history: list[dict[str, str]],
    matched_skills: list,
) -> str:
    """Run skill learning/evaluation hooks. Return optional reply suffix."""
    suffix = ""

    if matched_skills and skills_config.use_enabled:
        primary = matched_skills[0]

        async def _evaluate() -> None:
            try:
                await evaluate_skill_use(
                    hass,
                    entry_id,
                    llm,
                    backend,
                    skill=primary,
                    trace=trace,
                )
                await _update_skill_status(hass, entry_id)
            except Exception as err:
                LOGGER.warning("Skill evaluation failed: %s", err)

        hass.async_create_task(_evaluate())

    manual_save = bool(_MANUAL_SAVE.search(trace.user_text))
    if not manual_save:
        if not should_offer_skill_creation(
            trace,
            learning_enabled=skills_config.learning_enabled,
        ):
            return suffix
        if not await assess_skill_worth_learning(
            llm,
            backend,
            trace=trace,
            history=history,
        ):
            return suffix

    if manual_save or skills_config.auto_save:

        async def _save() -> None:
            try:
                await create_skill_from_trace(
                    hass,
                    entry_id,
                    llm,
                    backend,
                    trace=trace,
                    history=history,
                )
                await _update_skill_status(hass, entry_id)
            except Exception as err:
                LOGGER.warning("Skill creation failed: %s", err)

        if skills_config.auto_save or manual_save:
            hass.async_create_task(_save())
            suffix = " Saving this as a skill."
        return suffix

    queue_pending_save(
        hass,
        entry_id,
        trace.conversation_id,
        trace=trace,
        history=history,
    )
    return " Save this as a skill?"


async def run_agent(
    hass: HomeAssistant,
    *,
    llm: LlmClient,
    mcp_client: McpProxyClient,
    backend: LlmBackend,
    agent_config: AgentConfig,
    router_config: RouterConfig,
    skills_config: SkillsConfig,
    entry_id: str,
    conversation_id: str | None,
    user_text: str,
    exposed_entities: list[dict[str, Any]],
    extra_system_prompt: str | None = None,
) -> AsyncGenerator[AgentDelta, None]:
    """Run the tool loop and yield assistant chat deltas."""
    history = get_history(
        hass,
        conversation_id,
        max_turns=agent_config.history_turns,
    )

    if is_affirmative(user_text) and (
        confirm := await try_confirm_pending_save(
            hass,
            entry_id,
            conversation_id,
            user_text,
            llm=llm,
            backend=backend,
        )
    ):
        append_turn(
            hass,
            conversation_id,
            user_text,
            confirm,
            max_turns=agent_config.history_turns,
            entry_id=entry_id,
        )
        await _update_skill_status(hass, entry_id)
        yield AgentDelta(content=confirm)
        return

    if is_skill_admin_query(user_text) and (
        reply := await try_handle_skill_command(hass, entry_id, user_text)
    ):
        append_turn(
            hass,
            conversation_id,
            user_text,
            reply,
            max_turns=agent_config.history_turns,
            entry_id=entry_id,
        )
        await _update_skill_status(hass, entry_id)
        yield AgentDelta(content=reply)
        return

    route = classify_route(user_text, exposed_entities, router_config)
    record_route(hass, entry_id, route)

    matched_skills = []
    skill_hints = ""
    if skills_config.use_enabled:
        matched_skills = await discover_skills(
            hass,
            entry_id,
            user_text,
            max_inject=skills_config.max_inject,
        )
        skill_hints = build_skill_hints(matched_skills)
        update_agent_status(
            hass,
            entry_id,
            active_skill=matched_skills[0].title if matched_skills else "none",
        )
    else:
        update_agent_status(hass, entry_id, active_skill="none")

    trace = TurnTrace(
        user_text=user_text,
        history_len=len(history),
        matched_skill_ids=[skill.id for skill in matched_skills],
        conversation_id=conversation_id,
    )

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

    tool_context = build_tool_context(
        user_text,
        exposed_entities,
        history=history,
        skill_hints=skill_hints,
        route=route.value,
    )
    playbook = route_playbook(route)
    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        mcp_session_prompt=mcp_session_prompt,
        tool_context=tool_context,
        extra_system_prompt=extra_system_prompt,
        route_playbook=playbook,
    )
    messages = build_messages(
        system_message=system_message,
        history=history,
        user_text=user_text,
    )
    tools = llm_tools
    use_chat_backend = route != TaskRoute.HA_ACTION
    controlled_entity_ids: list[str] = []
    loop_state = LoopState()

    for iteration in range(agent_config.max_iterations):
        trace.iterations = iteration + 1
        active_backend = (
            backend
            if use_chat_backend
            else backend_for_route(
                route,
                chat_backend=backend,
                router_config=router_config,
            )
        )

        if iteration > 0 and agent_config.show_reasoning_in_chat:
            yield AgentDelta(thinking_clear=True)

        if agent_config.enable_streaming:
            session = StreamChatSession()
            async for delta, active_session in _yield_streamed_assistant_text(
                llm,
                messages,
                active_backend,
                tools,
                show_reasoning=agent_config.show_reasoning_in_chat,
            ):
                session = active_session
                if delta.content or delta.thinking or delta.thinking_clear:
                    yield delta

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
                async for delta in _process_tool_calls(
                    agent_config,
                    session.tool_calls,
                    mcp_client,
                    messages,
                    hass=hass,
                    exposed_entities=exposed_entities,
                    controlled_entity_ids=controlled_entity_ids,
                    loop_state=loop_state,
                    trace=trace,
                ):
                    yield delta
                if loop_state.stuck:
                    yield AgentDelta(content=_finalize_stuck_turn(trace, loop_state))
                    append_turn(
                        hass,
                        conversation_id,
                        user_text,
                        loop_state.stuck_message,
                        max_turns=agent_config.history_turns,
                        entry_id=entry_id,
                    )
                    record_turn(hass, entry_id, trace)
                    return
                use_chat_backend = True
                continue

            embedded_ran = False
            if parse_embedded_tool_calls(raw_buffer.strip()):
                async for delta in _process_embedded_tool_calls(
                    agent_config,
                    mcp_client,
                    raw_buffer.strip(),
                    messages,
                    hass=hass,
                    exposed_entities=exposed_entities,
                    controlled_entity_ids=controlled_entity_ids,
                    loop_state=loop_state,
                    trace=trace,
                ):
                    yield delta
                if loop_state.stuck:
                    yield AgentDelta(content=_finalize_stuck_turn(trace, loop_state))
                    append_turn(
                        hass,
                        conversation_id,
                        user_text,
                        loop_state.stuck_message,
                        max_turns=agent_config.history_turns,
                        entry_id=entry_id,
                    )
                    record_turn(hass, entry_id, trace)
                    return
                embedded_ran = True
            if embedded_ran:
                use_chat_backend = True
                continue

            assistant_text = strip_embedded_tool_markup(raw_buffer)
            if not assistant_text and is_tool_call_only_text(raw_buffer):
                assistant_text = ""
        else:
            result = await llm.chat(messages, active_backend, tools=tools)
            if result.tool_calls:
                messages.append(result.assistant_message)
                async for delta in _process_tool_calls(
                    agent_config,
                    result.tool_calls,
                    mcp_client,
                    messages,
                    hass=hass,
                    exposed_entities=exposed_entities,
                    controlled_entity_ids=controlled_entity_ids,
                    loop_state=loop_state,
                    trace=trace,
                ):
                    yield delta
                if loop_state.stuck:
                    yield AgentDelta(content=_finalize_stuck_turn(trace, loop_state))
                    append_turn(
                        hass,
                        conversation_id,
                        user_text,
                        loop_state.stuck_message,
                        max_turns=agent_config.history_turns,
                        entry_id=entry_id,
                    )
                    record_turn(hass, entry_id, trace)
                    return
                use_chat_backend = True
                continue

            embedded_ran = False
            if parse_embedded_tool_calls((result.content or "").strip()):
                async for delta in _process_embedded_tool_calls(
                    agent_config,
                    mcp_client,
                    (result.content or "").strip(),
                    messages,
                    hass=hass,
                    exposed_entities=exposed_entities,
                    controlled_entity_ids=controlled_entity_ids,
                    loop_state=loop_state,
                    trace=trace,
                ):
                    yield delta
                if loop_state.stuck:
                    yield AgentDelta(content=_finalize_stuck_turn(trace, loop_state))
                    append_turn(
                        hass,
                        conversation_id,
                        user_text,
                        loop_state.stuck_message,
                        max_turns=agent_config.history_turns,
                        entry_id=entry_id,
                    )
                    record_turn(hass, entry_id, trace)
                    return
                embedded_ran = True
            if embedded_ran:
                use_chat_backend = True
                continue

            assistant_text = (result.content or "").strip()
            if agent_config.show_reasoning_in_chat and result.reasoning_content:
                yield AgentDelta(thinking=result.reasoning_content)
            if assistant_text and not is_tool_call_only_text(assistant_text):
                yield AgentDelta(content=assistant_text)
            elif assistant_text:
                assistant_text = ""

        trace.assistant_text = assistant_text
        trace.controlled_entity_ids = list(controlled_entity_ids)
        trace.verification_notes = list(loop_state.verification_notes)
        if any(note.startswith("VERIFICATION FAILED") for note in trace.verification_notes):
            trace.outcome = TurnOutcome.PARTIAL
        elif trace.tool_errors and not assistant_text:
            trace.outcome = TurnOutcome.FAILED
        else:
            trace.outcome = TurnOutcome.SUCCESS
        suffix = await _post_turn_skills(
            hass,
            entry_id=entry_id,
            llm=llm,
            backend=backend,
            skills_config=skills_config,
            trace=trace,
            history=history,
            matched_skills=matched_skills,
        )
        if suffix:
            assistant_text = f"{assistant_text}{suffix}".strip()
            yield AgentDelta(content=suffix)

        append_turn(
            hass,
            conversation_id,
            user_text,
            memory_assistant_text(assistant_text, controlled_entity_ids),
            max_turns=agent_config.history_turns,
            entry_id=entry_id,
        )
        record_turn(hass, entry_id, trace)
        return

    trace.fallback = True
    trace.outcome = TurnOutcome.PARTIAL if trace.tool_calls else TurnOutcome.FAILED
    trace.verification_notes = list(loop_state.verification_notes)
    fallback = (
        f"{FALLBACK_MESSAGE} I used {trace.iterations} steps"
        f"{' and hit a repeated-tool guard' if loop_state.stuck else ''}."
    )
    yield AgentDelta(content=fallback)
    append_turn(
        hass,
        conversation_id,
        user_text,
        fallback,
        max_turns=agent_config.history_turns,
        entry_id=entry_id,
    )
    record_turn(hass, entry_id, trace)

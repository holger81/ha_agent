"""Agent loop with MCP tool calling."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from homeassistant.core import HomeAssistant

from .activity import record_turn
from .compaction import compact_messages_if_needed
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
from .llm_telemetry import record_llm_call
from .loop_policy import (
    INTERNAL_GUIDANCE_ROLE,
    LoopState,
    TurnOutcome,
    build_empty_response_nudge,
    build_pending_failure_summary,
    check_stuck,
    finalize_output,
    initialize_loop_plan,
    inject_loop_context,
    mark_iteration_outcome,
    maybe_suspend_skill_plan_from_reasoning,
    reasoning_execution_mismatch,
    record_iteration_failure,
    record_mcp_guidance,
    record_plan_tool_result,
    reset_iteration_flags,
    should_retry_empty_response,
    skill_plan_blocks_discovery,
    suspend_skill_plan,
    user_requests_skill_override,
)
from .mcp_session import FALLBACK_MCP_TOOLS, mcp_tools_to_openai_schemas
from .memory import append_turn, conversation_history_for_turn
from .orchestrator import (
    Complexity,
    plan_subtasks,
    replan_after_failure,
    triage_complexity,
)
from .playbooks import async_select_playbook
from .prepass import run_turn_prepass
from .recovery_hints import async_recovery_hints
from .role_registry import (
    ModelRole,
    RoleRegistry,
    build_role_registry,
    collapse_identical_roles,
)
from .route_keywords import async_route_keyword_map
from .router import TaskRoute, backend_for_route, resolve_route_with_classifier
from .skills.commands import (
    _MANUAL_SAVE,
    is_skill_admin_query,
    queue_pending_save,
    try_confirm_pending_save,
    try_handle_skill_command,
)
from .skills.creator import save_skill_from_draft
from .skills.discovery import build_skill_hints
from .skills.evaluator import evaluate_skill_use
from .skills.models import TurnTrace
from .skills.observer import (
    is_discovery_tool,
    observe_skill_candidate,
    observe_skill_fork,
    observe_skill_override,
)
from .skills.params import (
    apply_slot_defaults,
    bind_tool_steps,
    bindings_diverge_from_defaults,
    infer_slot_bindings,
    missing_required_bindings,
)
from .skills.repair import auto_repair_skill
from .skills.route_skills import load_route_skill, merge_route_and_learned_skills
from .skills.runtime import (
    override_turn_eligible_for_learning,
    should_offer_skill_creation,
)
from .skills.selection import filter_tool_steps_for_route, resolve_skills_for_turn
from .skills.store import get_skill_store
from .status import record_route, update_agent_status
from .subagent import WorkerResult, run_worker
from .tool_pruning import prune_loop_tools
from .tools import (
    classify_tool_error,
    execute_tool,
    ha_service_entity_id,
    is_discovery_tool_name,
    memory_assistant_text,
    parse_tool_arguments,
    tool_result_message,
)
from .verifier import build_verifier_retry_guidance, verify_turn

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

FALLBACK_MESSAGE = "Sorry, I couldn't complete that request."
_MAX_VERIFIER_RETRIES = 1


@dataclass(slots=True, frozen=True)
class AgentDelta:
    """One streamed update for the Assist chat log."""

    content: str = ""
    thinking: str = ""
    tool: dict[str, Any] | None = None
    thinking_clear: bool = False
    content_clear: bool = False
    skill: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    subagent: dict[str, Any] | None = None


def _backend_host(base_url: str) -> str:
    """Return host portion of an LLM base URL for display."""
    try:
        return urlparse(base_url).netloc or base_url
    except ValueError:
        return base_url


def _model_chip(backend: LlmBackend) -> dict[str, str]:
    return {
        "model": backend.model,
        "host": _backend_host(backend.base_url),
    }


def _agent_model_role(route: TaskRoute, *, use_chat_backend: bool) -> str:
    if route == TaskRoute.EMAIL:
        return "email"
    if route == TaskRoute.NEWS:
        return "news"
    if route == TaskRoute.HA_ACTION and not use_chat_backend:
        return "action"
    return "chat"


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
    error_kind, error_message, missing_fields = classify_tool_error(output)
    succeeded = error_kind is None
    trace.tool_calls.append(
        {
            "toolName": tool_name,
            "name": tool_name,
            "arguments": arguments,
            "succeeded": succeeded,
            "discovery": is_discovery_tool(tool_name),
            "error": error_message or None,
            "error_kind": error_kind,
            "missing_fields": missing_fields,
        }
    )
    if not succeeded:
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


def _prepare_next_loop_iteration(loop_state: LoopState) -> None:
    """Capture failures from the current iteration for the next LLM step."""
    build_pending_failure_summary(loop_state)


def _preferred_loop_tool_names(skill_steps: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for step in skill_steps or []:
        name = step.get("toolName")
        if name:
            names.append(str(name))
    return names


def _schedule_post_turn_skills(
    hass: HomeAssistant,
    **kwargs: Any,
) -> None:
    """Run learning hooks off the critical path (Phase 8e)."""

    async def _run() -> None:
        try:
            await _post_turn_skills(hass, **kwargs)
        except Exception as err:
            LOGGER.warning("Post-turn skill hooks failed: %s", err)

    hass.async_create_task(_run())


def _handle_tool_result(
    call: ToolCall,
    raw_output: str,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    hass: HomeAssistant,
    loop_state: LoopState,
    hint_rules: list[Any] | None,
    messages: list[dict[str, Any]],
) -> tuple[str, AgentDelta]:
    """Finalize one tool output and append the tool message."""
    output = finalize_output(
        tool_name,
        arguments,
        raw_output,
        hass=hass,
        loop_state=loop_state,
        hint_rules=hint_rules,
    )
    phase = "error" if output.startswith("Tool error:") else "done"
    verification_failed = False
    if phase == "done":
        loop_state.iteration_had_successful_tool = True
        if is_discovery_tool_name(tool_name):
            loop_state.include_full_tool_catalog = True
        if "VERIFICATION FAILED" in output:
            verification_failed = True
            for line in output.splitlines():
                if "VERIFICATION FAILED" in line:
                    record_iteration_failure(
                        loop_state,
                        tool_name,
                        arguments,
                        line.strip(),
                    )
                    break
    else:
        record_iteration_failure(loop_state, tool_name, arguments, output)
    record_plan_tool_result(
        loop_state,
        tool_name,
        arguments,
        succeeded=phase == "done",
        verification_failed=verification_failed,
    )
    record_mcp_guidance(loop_state, tool_name, output)
    messages.append(tool_result_message(call, output))
    return phase, AgentDelta(
        tool=_tool_event(
            call,
            phase,
            detail=_tool_event_detail(call, output),
        )
    )


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
    hint_rules: list[Any] | None = None,
    reasoning: str = "",
) -> AsyncGenerator[AgentDelta, None]:
    """Run tool calls and yield chat progress deltas."""
    maybe_suspend_skill_plan_from_reasoning(loop_state, reasoning)
    blocked_ids: set[str] = set()
    if calls and reasoning.strip():
        execution_names = [_tool_call_payload(call)[0] for call in calls]
        if mismatch := reasoning_execution_mismatch(reasoning, execution_names):
            for call in calls:
                tool_name, arguments = _tool_call_payload(call)
                if is_discovery_tool_name(tool_name):
                    continue
                blocked = f"Tool error: {mismatch}"
                blocked_ids.add(call.id)
                yield AgentDelta(
                    tool=_tool_event(
                        call,
                        "error",
                        detail="Blocked: conflicts with reasoning.",
                    )
                )
                record_iteration_failure(
                    loop_state,
                    tool_name,
                    arguments,
                    blocked,
                )
                record_plan_tool_result(
                    loop_state,
                    tool_name,
                    arguments,
                    succeeded=False,
                )
                messages.append(tool_result_message(call, blocked))
                if trace is not None:
                    _record_tool_call(trace, call, blocked)

    runnable: list[ToolCall] = []
    for call in calls:
        if call.id in blocked_ids:
            continue
        tool_name, arguments = _tool_call_payload(call)
        if is_discovery_tool_name(tool_name) and skill_plan_blocks_discovery(
            loop_state
        ):
            blocked = (
                "Tool error: Active skill lists concrete tool steps; "
                "do not run discovery while following that workflow. "
                "If the skill does not fit the user's goal, declare "
                "SKILL_OVERRIDE: <reason> in your reasoning, then retry."
            )
            record_iteration_failure(loop_state, tool_name, arguments, blocked)
            record_plan_tool_result(
                loop_state,
                tool_name,
                arguments,
                succeeded=False,
            )
            yield AgentDelta(
                tool=_tool_event(
                    call,
                    "error",
                    detail="Discovery blocked — follow skill steps.",
                )
            )
            messages.append(tool_result_message(call, blocked))
            if trace is not None:
                _record_tool_call(trace, call, blocked)
            continue
        if stuck_msg := check_stuck(loop_state, tool_name, arguments):
            loop_state.iteration_had_duplicate_block = True
            record_plan_tool_result(
                loop_state,
                tool_name,
                arguments,
                succeeded=False,
            )
            record_iteration_failure(
                loop_state,
                tool_name,
                arguments,
                stuck_msg,
            )
            yield AgentDelta(tool=_tool_event(call, "error", detail=stuck_msg[:200]))
            messages.append(
                tool_result_message(call, f"Tool error: {stuck_msg}")
            )
            if trace is not None:
                _record_tool_call(trace, call, f"Tool error: {stuck_msg}")
            continue
        runnable.append(call)

    if len(runnable) > 1:
        for call in runnable:
            yield AgentDelta(tool=_tool_event(call, "start"))
        raw_outputs = await asyncio.gather(
            *[
                _run_tool_call(
                    mcp_client,
                    call,
                    exposed_entities=exposed_entities,
                    controlled_entity_ids=controlled_entity_ids,
                    trace=trace,
                )
                for call in runnable
            ]
        )
        for call, raw_output in zip(runnable, raw_outputs, strict=True):
            tool_name, arguments = _tool_call_payload(call)
            _, delta = _handle_tool_result(
                call,
                raw_output,
                tool_name=tool_name,
                arguments=arguments,
                hass=hass,
                loop_state=loop_state,
                hint_rules=hint_rules,
                messages=messages,
            )
            yield delta
        return

    for call in runnable:
        tool_name, arguments = _tool_call_payload(call)
        yield AgentDelta(tool=_tool_event(call, "start"))
        raw_output = await _run_tool_call(
            mcp_client,
            call,
            exposed_entities=exposed_entities,
            controlled_entity_ids=controlled_entity_ids,
            trace=trace,
        )
        _, delta = _handle_tool_result(
            call,
            raw_output,
            tool_name=tool_name,
            arguments=arguments,
            hass=hass,
            loop_state=loop_state,
            hint_rules=hint_rules,
            messages=messages,
        )
        yield delta


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
    hint_rules: list[Any] | None = None,
    reasoning: str = "",
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
        hint_rules=hint_rules,
        reasoning=reasoning,
    ):
        yield delta


async def _run_orchestrated_turn(
    hass: HomeAssistant,
    *,
    llm: LlmClient,
    mcp_client: McpProxyClient,
    registry: RoleRegistry,
    agent_config: AgentConfig,
    skills_config: SkillsConfig,
    entry_id: str,
    conversation_id: str | None,
    user_text: str,
    orch_plan: Any,
    exposed_entities: list[dict[str, Any]],
    mcp_session_prompt: str,
    llm_tools: list[dict[str, Any]],
    backend: LlmBackend,
    trace: TurnTrace,
    hint_rules: list[Any] | None,
) -> AsyncGenerator[AgentDelta, None]:
    """Execute a complex multi-subtask turn via worker subagents."""
    from .context import build_messages, build_system_message

    worker_results: list[WorkerResult] = []
    pending_subtasks = list(orch_plan.subtasks)
    replans = 0
    structured = agent_config.structured_output_enabled
    index = 0
    while index < len(pending_subtasks):
        subtask = pending_subtasks[index]
        sub_skills: list = []
        if skills_config.use_enabled:
            sub_sel = await resolve_skills_for_turn(
                hass,
                entry_id,
                llm,
                registry.backend_for(ModelRole.ROUTER),
                subtask.subgoal,
                history=[],
                route=subtask.route,
                max_inject=1,
                structured_output_enabled=structured,
                trace=trace,
            )
            sub_skills = sub_sel.skills
            route_skill = await load_route_skill(
                hass,
                entry_id,
                llm,
                registry.backend_for(ModelRole.ROUTER),
                user_text=subtask.subgoal,
                route_value=subtask.route,
                history=[],
            )
            sub_skills = merge_route_and_learned_skills(route_skill, sub_skills)

        worker_result: WorkerResult | None = None
        async for meta, result in run_worker(
            hass,
            llm=llm,
            mcp_client=mcp_client,
            registry=registry,
            agent_config=agent_config,
            subgoal=subtask.subgoal,
            route_value=subtask.route,
            exposed_entities=exposed_entities,
            matched_skills=sub_skills,
            mcp_session_prompt=mcp_session_prompt,
            llm_tools=llm_tools,
            prior_results=worker_results,
        ):
            if meta:
                yield AgentDelta(subagent=meta)
            if result is not None:
                worker_result = result

        if worker_result is not None:
            worker_results.append(worker_result)
            trace.subtask_results.append(
                {
                    "subgoal": worker_result.subgoal,
                    "route": worker_result.route,
                    "summary": worker_result.assistant_text[:500],
                    "tool_errors": worker_result.tool_errors,
                }
            )
            if (
                worker_result.tool_errors > 0
                and replans < agent_config.max_replans
            ):
                replans += 1
                completed = [
                    {
                        "subgoal": item.subgoal,
                        "summary": item.assistant_text[:300],
                    }
                    for item in worker_results
                ]
                revised = await replan_after_failure(
                    llm,
                    registry,
                    user_text=user_text,
                    plan=orch_plan,
                    failed_subtask=subtask,
                    completed_summaries=completed,
                    structured_output_enabled=structured,
                    trace=trace,
                )
                pending_subtasks = revised.subtasks + pending_subtasks[index + 1 :]
                index = 0
                continue
        index += 1

    synth_backend = registry.backend_for(ModelRole.WORKER_CHAT)
    synth_body = "\n".join(
        f"- {r.subgoal}: {r.assistant_text}" for r in worker_results
    )
    synth_messages = build_messages(
        system_message=build_system_message(
            agent_config.system_prompt,
            agent_config.tool_instructions,
            extra_system_prompt=(
                "Synthesize subtask results into one concise reply for the user."
            ),
        ),
        history=[],
        user_text=f"Original request: {user_text}\n\nSubtask results:\n{synth_body}",
    )
    synth = await llm.chat(synth_messages, synth_backend, tools=[])
    record_llm_call(trace, role="synth", backend=synth_backend, result=synth)
    assistant_text = (synth.content or "").strip() or FALLBACK_MESSAGE
    yield AgentDelta(content=assistant_text)

    v_result = await verify_turn(
        llm,
        registry.backend_for(ModelRole.VERIFIER),
        user_text=user_text,
        assistant_text=assistant_text,
        tool_calls=trace.tool_calls,
        tool_errors=sum(r.tool_errors for r in worker_results),
        structured_output_enabled=structured,
        trace=trace,
    )
    trace.verifier_verdict = "pass" if v_result.passed else "fail"
    trace.verifier_detail = v_result.reason
    trace.skill_followed = v_result.skill_followed
    yield AgentDelta(
        meta={
            "verifier_verdict": trace.verifier_verdict,
            "verifier_detail": trace.verifier_detail,
        }
    )

    trace.assistant_text = assistant_text
    trace.outcome = TurnOutcome.SUCCESS if v_result.passed else TurnOutcome.PARTIAL
    _schedule_post_turn_skills(
        hass,
        entry_id=entry_id,
        llm=llm,
        backend=backend,
        observer_backend=registry.backend_for(ModelRole.OBSERVER),
        skills_config=skills_config,
        trace=trace,
        history=[],
        matched_skills=[],
    )

    append_turn(
        hass,
        conversation_id,
        user_text,
        assistant_text,
        max_turns=agent_config.history_turns,
        entry_id=entry_id,
        turn_meta={
            "complexity": orch_plan.complexity.value,
            "verifier_verdict": trace.verifier_verdict,
            "subtask_count": len(worker_results),
        },
    )
    record_turn(hass, entry_id, trace)


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
    observer_backend: LlmBackend,
    skills_config: SkillsConfig,
    trace: TurnTrace,
    history: list[dict[str, str]],
    matched_skills: list,
) -> tuple[str, dict[str, Any] | None]:
    """Run skill learning/evaluation hooks. Return suffix and optional meta patch."""
    suffix = ""
    meta_patch: dict[str, Any] | None = None
    primary_learned = next((s for s in matched_skills if not s.is_builtin), None)

    if primary_learned and skills_config.use_enabled:
        repair_result = await hass.async_add_executor_job(
            auto_repair_skill,
            hass,
            entry_id,
            primary_learned,
            trace,
        )
        if repair_result is not None:
            suffix = (
                f" Updated skill: {repair_result.skill.title} "
                f"(v{repair_result.from_version}→v{repair_result.skill.version}) "
                f"— {repair_result.reason}."
            )
            meta_patch = {
                "skill_update": {
                    "title": repair_result.skill.title,
                    "from_version": repair_result.from_version,
                    "to_version": repair_result.skill.version,
                    "reason": repair_result.reason,
                    "revision_id": repair_result.revision_id,
                }
            }
            update_agent_status(
                hass,
                entry_id,
                last_skill_improved=repair_result.skill.title,
            )

        async def _evaluate() -> None:
            try:
                await evaluate_skill_use(
                    hass,
                    entry_id,
                    llm,
                    backend,
                    skill=primary_learned,
                    trace=trace,
                )
                await _update_skill_status(hass, entry_id)
            except Exception as err:
                LOGGER.warning("Skill evaluation failed: %s", err)

        hass.async_create_task(_evaluate())

        if (
            trace.skill_plan_override
            and skills_config.learning_enabled
            and override_turn_eligible_for_learning(trace)
        ):
            override_obs = await observe_skill_override(
                llm,
                observer_backend,
                parent_skill=primary_learned,
                trace=trace,
                history=history,
            )
            if (
                override_obs
                and override_obs.learn
                and override_obs.draft is not None
            ):
                update_target = (
                    primary_learned if override_obs.update_parent else None
                )
                draft = override_obs.draft
                action = "update" if update_target else "create"

                async def _save_override(
                    *,
                    _draft=draft,
                    _update=update_target,
                    _action=action,
                ) -> None:
                    try:
                        await save_skill_from_draft(
                            hass,
                            entry_id,
                            _draft,
                            update_existing=_update,
                        )
                        await _update_skill_status(hass, entry_id)
                    except Exception as err:
                        LOGGER.warning("Skill override save failed: %s", err)

                if skills_config.auto_save:
                    from_version = (
                        update_target.version if update_target is not None else None
                    )
                    hass.async_create_task(_save_override())
                    if action == "update" and from_version is not None:
                        suffix = (
                            f" Updating skill: {draft.title} "
                            f"(v{from_version}→v{from_version + 1}) "
                            f"— {override_obs.reason}."
                        )
                        meta_patch = {
                            "skill_update": {
                                "title": draft.title,
                                "from_version": from_version,
                                "to_version": from_version + 1,
                                "reason": override_obs.reason,
                                "override": True,
                            }
                        }
                    else:
                        suffix = f" Saving skill: {draft.title}."
                    return suffix, meta_patch

                queue_pending_save(
                    hass,
                    entry_id,
                    trace.conversation_id,
                    trace=trace,
                    history=history,
                    skill_draft=draft,
                    observer_reason=override_obs.reason,
                    update_skill_id=(
                        primary_learned.id if update_target is not None else None
                    ),
                )
                if update_target is not None:
                    return (
                        f" I can update skill “{draft.title}” with this workflow. "
                        "Reply yes to confirm.",
                        meta_patch,
                    )
                return (
                    f" I can save a new skill: {draft.title}. Reply yes to confirm.",
                    meta_patch,
                )

        if (
            trace.slot_bindings
            and bindings_diverge_from_defaults(primary_learned, trace.slot_bindings)
            and skills_config.learning_enabled
        ):
            forked = await observe_skill_fork(
                llm,
                observer_backend,
                parent_skill=primary_learned,
                trace=trace,
                history=history,
            )
            if forked and forked.learn and forked.draft is not None:
                if skills_config.auto_save:

                    async def _save_fork() -> None:
                        try:
                            await save_skill_from_draft(
                                hass,
                                entry_id,
                                forked.draft,
                                update_existing=None,
                            )
                            await _update_skill_status(hass, entry_id)
                        except Exception as err:
                            LOGGER.warning("Skill fork save failed: %s", err)

                    hass.async_create_task(_save_fork())
                    return (
                        f" Saving skill variant: {forked.draft.title}.",
                        meta_patch,
                    )
                queue_pending_save(
                    hass,
                    entry_id,
                    trace.conversation_id,
                    trace=trace,
                    history=history,
                    skill_draft=forked.draft,
                    observer_reason=forked.reason,
                )
                return (
                    f" I can save a variant skill: {forked.draft.title}. "
                    "Reply yes to confirm.",
                    meta_patch,
                )

    manual_save = bool(_MANUAL_SAVE.search(trace.user_text))
    if manual_save:
        if not should_offer_skill_creation(
            trace,
            learning_enabled=skills_config.learning_enabled,
            manual_save=True,
        ):
            return (
                " I couldn't save that — this turn has no successful tool workflow.",
                meta_patch,
            )
    elif not should_offer_skill_creation(
        trace,
        learning_enabled=skills_config.learning_enabled,
    ):
        return suffix, meta_patch

    observed = await observe_skill_candidate(
        llm,
        observer_backend,
        trace=trace,
        history=history,
        manual_save=manual_save,
    )
    if not observed.learn or observed.draft is None:
        if manual_save:
            return (
                " I don't think this turn has a reusable workflow worth saving "
                "as a skill.",
                meta_patch,
            )
        return suffix, meta_patch

    if manual_save or skills_config.auto_save:

        async def _save() -> None:
            try:
                store = get_skill_store(hass, entry_id)

                def _find_dup():
                    return store.find_duplicate(observed.draft.triggers)

                duplicate = await hass.async_add_executor_job(_find_dup)
                await save_skill_from_draft(
                    hass,
                    entry_id,
                    observed.draft,
                    update_existing=duplicate,
                )
                await _update_skill_status(hass, entry_id)
            except Exception as err:
                LOGGER.warning("Skill creation failed: %s", err)

        hass.async_create_task(_save())
        return f" Saving skill: {observed.draft.title}.", meta_patch

    queue_pending_save(
        hass,
        entry_id,
        trace.conversation_id,
        trace=trace,
        history=history,
        skill_draft=observed.draft,
        observer_reason=observed.reason,
    )
    return f" Save skill “{observed.draft.title}”?", meta_patch


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
    history = conversation_history_for_turn(
        hass,
        conversation_id,
        user_text,
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

    route_keywords = await async_route_keyword_map(hass, entry_id)
    classifier_backend = router_config.classifier_backend or backend
    role_registry = collapse_identical_roles(
        build_role_registry(backend, router_config)
    )
    structured = agent_config.structured_output_enabled
    trace = TurnTrace(
        user_text=user_text,
        history_len=len(history),
        conversation_id=conversation_id,
    )

    prepass_result = None
    skill_selection = None
    matched_skills: list = []
    slot_bindings: dict[str, str] = {}
    if agent_config.prepass_enabled:
        prepass_result = await run_turn_prepass(
            hass,
            entry_id,
            llm,
            classifier_backend,
            user_text=user_text,
            history=history,
            router_config=router_config,
            route_keywords=route_keywords,
            skills_enabled=skills_config.use_enabled,
            max_inject=skills_config.max_inject,
            structured_output_enabled=structured,
            trace=trace,
        )

    if prepass_result is not None:
        route_resolution = prepass_result.route_resolution
        route = route_resolution.route
        orch_plan = prepass_result.orch_plan
        skill_selection = prepass_result.skill_selection
        matched_skills = list(skill_selection.skills) if skill_selection else []
        slot_bindings = dict(prepass_result.slot_bindings)
    else:
        route_resolution = await resolve_route_with_classifier(
            llm,
            classifier_backend,
            user_text=user_text,
            exposed_entities=exposed_entities,
            router_config=router_config,
            route_keywords=route_keywords,
            history=history,
            structured_output_enabled=structured,
            trace=trace,
        )
        route = route_resolution.route
        orch_plan = await triage_complexity(
            llm,
            role_registry,
            user_text=user_text,
            history=history,
            structured_output_enabled=structured,
            trace=trace,
        )
    record_route(hass, entry_id, route)
    if orch_plan.complexity == Complexity.COMPLEX:
        orch_plan = await plan_subtasks(
            llm,
            role_registry,
            user_text=user_text,
            plan=orch_plan,
            structured_output_enabled=structured,
            trace=trace,
        )
    yield AgentDelta(
        meta={
            "route": route.value,
            "route_classifier": route_resolution.classifier_summary,
            "route_classifier_detail": route_resolution.classifier_detail,
            "route_classifier_raw": route_resolution.classifier_raw,
            "keyword_hint": route_resolution.keyword_hint,
            "classification": route_resolution.classifier_summary,
            "route_method": route_resolution.method,
            "complexity": orch_plan.complexity.value,
            "orchestration_reason": orch_plan.reason,
            "subtask_count": len(orch_plan.subtasks),
            "planner": role_registry.chip_for(ModelRole.PLANNER),
        }
    )
    hint_rules = await async_recovery_hints(hass, entry_id)

    skill_hints = ""
    if skills_config.use_enabled:
        if prepass_result is None or (
            not matched_skills and prepass_result.method == "prepass"
        ):
            skill_selection = await resolve_skills_for_turn(
                hass,
                entry_id,
                llm,
                classifier_backend,
                user_text,
                history=history,
                route=route.value,
                max_inject=skills_config.max_inject,
                structured_output_enabled=structured,
                trace=trace,
            )
            matched_skills = skill_selection.skills
        learned_only = [s for s in matched_skills if not s.is_builtin]
        route_skill = await load_route_skill(
            hass,
            entry_id,
            llm,
            classifier_backend,
            user_text=user_text,
            route_value=route.value,
            history=history,
        )
        matched_skills = merge_route_and_learned_skills(route_skill, learned_only)
        primary_learned = next((s for s in matched_skills if not s.is_builtin), None)
        if primary_learned and not slot_bindings:
            slot_bindings = await infer_slot_bindings(
                llm,
                role_registry.backend_for(ModelRole.ROUTER),
                user_text=user_text,
                skill=primary_learned,
                route=route.value,
                structured_output_enabled=structured,
                trace=trace,
            )
            slot_bindings = apply_slot_defaults(
                slot_bindings,
                primary_learned,
                route=route.value,
            )
        skill_hints = build_skill_hints(
            matched_skills,
            route=route.value,
            slot_bindings=slot_bindings,
        )
        update_agent_status(
            hass,
            entry_id,
            active_skill=(
                primary_learned.title
                if primary_learned
                else (matched_skills[0].title if matched_skills else "none")
            ),
        )
        if matched_skills:
            primary = primary_learned or matched_skills[0]
            yield AgentDelta(
                skill={
                    "id": primary.id,
                    "slug": primary.slug,
                    "title": primary.title,
                }
            )
    else:
        update_agent_status(hass, entry_id, active_skill="none")

    trace.matched_skill_ids = [skill.id for skill in matched_skills]
    trace.matched_learned_skill_ids = [
        skill.id for skill in matched_skills if not skill.is_builtin
    ]
    trace.route = route.value
    trace.complexity = orch_plan.complexity.value
    trace.slot_bindings = slot_bindings
    trace.orchestration_plan = [
        {
            "id": st.id,
            "subgoal": st.subgoal,
            "route": st.route,
        }
        for st in orch_plan.subtasks
    ]
    trace.exposed_entities = list(exposed_entities)

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

    if orch_plan.complexity == Complexity.COMPLEX and orch_plan.subtasks:
        async for handled in _run_orchestrated_turn(
            hass,
            llm=llm,
            mcp_client=mcp_client,
            registry=role_registry,
            agent_config=agent_config,
            skills_config=skills_config,
            entry_id=entry_id,
            conversation_id=conversation_id,
            user_text=user_text,
            orch_plan=orch_plan,
            exposed_entities=exposed_entities,
            mcp_session_prompt=mcp_session_prompt,
            llm_tools=llm_tools,
            backend=backend,
            trace=trace,
            hint_rules=hint_rules,
        ):
            yield handled
        return

    tool_context = build_tool_context(
        user_text,
        exposed_entities,
        history=history,
        skill_hints=skill_hints,
        route=route.value,
    )
    playbook_selection = await async_select_playbook(
        hass,
        entry_id,
        llm,
        router_config.classifier_backend or backend,
        user_text=user_text,
        route_value=route.value,
        history=history,
    )
    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        mcp_session_prompt=mcp_session_prompt,
        tool_context=tool_context,
        extra_system_prompt=extra_system_prompt,
        route_playbook=playbook_selection.body,
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
    skill_steps: list[dict[str, Any]] | None = None
    if matched_skills:
        primary = next(
            (s for s in matched_skills if not s.is_builtin),
            matched_skills[0],
        )
        raw_steps = filter_tool_steps_for_route(primary.tool_steps, route.value)
        if raw_steps:
            skill_steps = bind_tool_steps(raw_steps, slot_bindings)
    initialize_loop_plan(
        loop_state,
        goal=user_text,
        route=route.value,
        tool_steps=skill_steps,
        skill_title=matched_skills[0].title if matched_skills else "",
        slot_bindings=slot_bindings or None,
    )
    if matched_skills and user_requests_skill_override(user_text):
        suspend_skill_plan(
            loop_state,
            "User asked to override the active skill workflow.",
        )
    if matched_skills and slot_bindings:
        primary_learned = next(
            (s for s in matched_skills if not s.is_builtin), None
        )
        if primary_learned:
            missing_slots = missing_required_bindings(primary_learned, slot_bindings)
            if missing_slots:
                loop_state.mcp_guidance.insert(
                    0,
                    (
                        "Bind required skill slots before calling tools: "
                        + ", ".join(missing_slots)
                    ),
                )

    classifier_backend = router_config.classifier_backend or backend
    turn_meta: dict[str, Any] = {
        "route": route.value,
        "route_classifier": route_resolution.classifier_summary,
        "route_classifier_detail": route_resolution.classifier_detail,
        "keyword_hint": route_resolution.keyword_hint,
        "classification": route_resolution.classifier_summary,
        "route_method": route_resolution.method,
        "playbook": playbook_selection.key,
        "playbook_method": playbook_selection.method,
        "playbook_detail": playbook_selection.detail,
        "skill": matched_skills[0].title if matched_skills else None,
        "skill_slug": matched_skills[0].slug if matched_skills else None,
        "skill_classifier": skill_selection.summary if skill_selection else None,
        "skill_classifier_detail": (
            skill_selection.detail if skill_selection else None
        ),
        "complexity": orch_plan.complexity.value,
        "slot_bindings": slot_bindings or None,
        "verifier": role_registry.chip_for(ModelRole.VERIFIER),
        "history_messages": len(history),
        "mcp_tools": len(llm_tools),
        "classifier": _model_chip(classifier_backend),
        "max_iterations": agent_config.max_iterations,
        "llm_calls": len(trace.llm_calls),
        "prepass": prepass_result.method if prepass_result else None,
    }
    yield AgentDelta(meta=turn_meta)

    verifier_retries = 0
    for iteration in range(agent_config.max_iterations):
        trace.iterations = iteration + 1
        reset_iteration_flags(loop_state)
        if iteration > 0:
            yield AgentDelta(content_clear=True)
            inject_loop_context(messages, loop_state)
            if agent_config.show_reasoning_in_chat:
                yield AgentDelta(thinking_clear=True)
        if agent_config.turn_token_budget > 0:
            compact_messages_if_needed(
                messages,
                token_budget=agent_config.turn_token_budget,
            )
        tools = prune_loop_tools(
            llm_tools,
            preferred_names=_preferred_loop_tool_names(skill_steps),
            max_tools=agent_config.max_loop_tools,
            include_full_catalog=loop_state.include_full_tool_catalog,
        )
        active_backend = backend_for_route(
            route,
            chat_backend=backend,
            router_config=router_config,
            prefer_action=route == TaskRoute.HA_ACTION and not use_chat_backend,
        )
        model_role = _agent_model_role(route, use_chat_backend=use_chat_backend)
        iteration_meta = {
            "iteration": iteration + 1,
            "model_role": model_role,
            **_model_chip(active_backend),
        }
        turn_meta.update(iteration_meta)
        yield AgentDelta(meta=iteration_meta)

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
                if (
                    delta.content
                    or delta.thinking
                    or delta.thinking_clear
                    or delta.content_clear
                ):
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
                    hint_rules=hint_rules,
                    reasoning=session.reasoning_content,
                ):
                    yield delta
                mark_iteration_outcome(loop_state)
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
                _prepare_next_loop_iteration(loop_state)
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
                    hint_rules=hint_rules,
                    reasoning=session.reasoning_content,
                ):
                    yield delta
                mark_iteration_outcome(loop_state)
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
                _prepare_next_loop_iteration(loop_state)
                use_chat_backend = True
                continue

            assistant_text = strip_embedded_tool_markup(raw_buffer)
            if not assistant_text and is_tool_call_only_text(raw_buffer):
                assistant_text = ""
        else:
            result = await llm.chat(messages, active_backend, tools=tools)
            record_llm_call(
                trace,
                role=model_role,
                backend=active_backend,
                result=result,
            )
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
                    hint_rules=hint_rules,
                    reasoning=result.reasoning_content or "",
                ):
                    yield delta
                mark_iteration_outcome(loop_state)
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
                _prepare_next_loop_iteration(loop_state)
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
                    hint_rules=hint_rules,
                    reasoning=result.reasoning_content or "",
                ):
                    yield delta
                mark_iteration_outcome(loop_state)
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
                _prepare_next_loop_iteration(loop_state)
                use_chat_backend = True
                continue

            assistant_text = (result.content or "").strip()
            if agent_config.show_reasoning_in_chat and result.reasoning_content:
                yield AgentDelta(thinking=result.reasoning_content)
            if assistant_text and not is_tool_call_only_text(assistant_text):
                yield AgentDelta(content=assistant_text)
            elif assistant_text:
                assistant_text = ""

        if not assistant_text and should_retry_empty_response(
            loop_state, iteration, agent_config.max_iterations
        ):
            messages.append(
                {
                    "role": INTERNAL_GUIDANCE_ROLE,
                    "content": build_empty_response_nudge(loop_state),
                }
            )
            _prepare_next_loop_iteration(loop_state)
            use_chat_backend = True
            continue

        if not assistant_text:
            assistant_text = FALLBACK_MESSAGE
            yield AgentDelta(content=assistant_text)

        primary_learned = next(
            (s for s in matched_skills if not s.is_builtin), None
        )
        if primary_learned is not None or trace.tool_errors > 0:
            v_result = await verify_turn(
                llm,
                role_registry.backend_for(ModelRole.VERIFIER),
                user_text=user_text,
                assistant_text=assistant_text,
                tool_calls=trace.tool_calls,
                tool_errors=trace.tool_errors,
                skill=primary_learned,
                slot_bindings=slot_bindings,
                structured_output_enabled=structured,
                trace=trace,
            )
        else:
            from .verifier import VerifierResult

            v_result = VerifierResult(passed=True, reason="no skill workflow")
        trace.verifier_verdict = "pass" if v_result.passed else "fail"
        trace.verifier_detail = v_result.reason
        trace.skill_followed = v_result.skill_followed
        yield AgentDelta(
            meta={
                "verifier_verdict": trace.verifier_verdict,
                "verifier_detail": trace.verifier_detail,
            }
        )
        if not v_result.passed and verifier_retries < _MAX_VERIFIER_RETRIES:
            verifier_retries += 1
            messages.append(
                {
                    "role": INTERNAL_GUIDANCE_ROLE,
                    "content": build_verifier_retry_guidance(v_result),
                }
            )
            _prepare_next_loop_iteration(loop_state)
            use_chat_backend = True
            continue

        trace.assistant_text = assistant_text
        trace.controlled_entity_ids = list(controlled_entity_ids)
        trace.verification_notes = list(loop_state.verification_notes)
        failed_verification = any(
            note.startswith("VERIFICATION FAILED")
            for note in trace.verification_notes
        )
        if failed_verification or trace.verifier_verdict == "fail":
            trace.outcome = TurnOutcome.PARTIAL
        elif trace.tool_errors and not assistant_text:
            trace.outcome = TurnOutcome.FAILED
        else:
            trace.outcome = TurnOutcome.SUCCESS
        trace.recovery_hints = list(loop_state.mcp_guidance)
        trace.skill_plan_override = loop_state.skill_plan_override
        trace.skill_plan_override_reason = loop_state.skill_plan_override_reason
        turn_meta.update(
            {
                "verifier_verdict": trace.verifier_verdict,
                "verifier_detail": trace.verifier_detail,
                "llm_calls": trace.llm_calls,
            }
        )
        _schedule_post_turn_skills(
            hass,
            entry_id=entry_id,
            llm=llm,
            backend=backend,
            observer_backend=role_registry.backend_for(ModelRole.OBSERVER),
            skills_config=skills_config,
            trace=trace,
            history=history,
            matched_skills=matched_skills,
        )

        append_turn(
            hass,
            conversation_id,
            user_text,
            memory_assistant_text(assistant_text, controlled_entity_ids),
            max_turns=agent_config.history_turns,
            entry_id=entry_id,
            turn_meta=turn_meta,
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
        turn_meta=turn_meta,
    )
    record_turn(hass, entry_id, trace)

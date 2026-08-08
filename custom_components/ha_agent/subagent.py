"""Bounded worker subagent for orchestrated subtasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from .compaction import compact_messages_if_needed
from .config_helpers import AgentConfig, RouterConfig
from .context import build_messages, build_system_message, build_tool_context
from .llm_client import LlmClient
from .loop_policy import (
    LoopState,
    cache_mcp_tools_from_schemas,
    initialize_loop_plan,
    inject_loop_context,
    reset_iteration_flags,
)
from .mcp_session import FALLBACK_MCP_TOOLS, mcp_tools_to_openai_schemas
from .role_registry import ModelRole, RoleRegistry
from .skills.backend import backend_for_skill
from .skills.discovery import build_skill_hints
from .skills.models import Skill, TurnTrace
from .skills.params import bind_tool_steps, infer_slot_bindings
from .skills.selection import filter_tool_steps_for_route
from .tool_pruning import prune_loop_tools

if TYPE_CHECKING:
    from .mcp_client import McpProxyClient

_MAX_WORKER_ITERATIONS = 4


@dataclass(slots=True)
class WorkerResult:
    """Structured result from one subagent worker."""

    subgoal: str
    route: str
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: int = 0
    iterations: int = 0
    skill_title: str | None = None
    slot_bindings: dict[str, str] = field(default_factory=dict)


async def run_worker(
    hass: HomeAssistant,
    *,
    llm: LlmClient,
    mcp_client: McpProxyClient,
    registry: RoleRegistry,
    agent_config: AgentConfig,
    subgoal: str,
    route_value: str,
    exposed_entities: list[dict[str, Any]],
    matched_skills: list[Skill],
    mcp_session_prompt: str = "",
    llm_tools: list[dict[str, Any]] | None = None,
    prior_results: list[WorkerResult] | None = None,
    router_config: RouterConfig | None = None,
) -> AsyncGenerator[tuple[dict[str, Any] | None, WorkerResult | None], None]:
    """Run a bounded worker loop for one subtask; yields (subagent_meta, result)."""
    from .agent import _model_chip, _process_tool_calls

    tools = llm_tools or mcp_tools_to_openai_schemas(FALLBACK_MCP_TOOLS)

    primary_skill = matched_skills[0] if matched_skills else None
    backend = backend_for_skill(
        primary_skill,
        route=route_value,
        chat_backend=registry.chat_backend,
        action_backend=(
            registry.backend_for(ModelRole.WORKER_ACTION)
            if registry.action_enabled
            else None
        ),
        router_config=router_config,
    )
    slot_bindings: dict[str, str] = {}
    if primary_skill:
        slot_bindings = await infer_slot_bindings(
            llm,
            registry.backend_for(ModelRole.ROUTER),
            user_text=subgoal,
            skill=primary_skill,
            route=route_value,
            exposed_entities=exposed_entities,
        )

    yield (
        {
            "phase": "start",
            "subgoal": subgoal,
            "route": route_value,
            "skill": primary_skill.title if primary_skill else None,
            **_model_chip(backend),
        },
        None,
    )

    skill_hints = build_skill_hints(matched_skills, route=route_value)
    tool_context = build_tool_context(
        subgoal,
        exposed_entities,
        skill_hints=skill_hints,
        route=route_value,
    )
    if prior_results:
        prior_lines = [
            f"- {r.subgoal}: {r.assistant_text[:200]}" for r in prior_results
        ]
        tool_context += "\n\nPRIOR SUBTASK RESULTS:\n" + "\n".join(prior_lines)

    system_message = build_system_message(
        agent_config.system_prompt,
        agent_config.tool_instructions,
        mcp_session_prompt=mcp_session_prompt,
        tool_context=tool_context,
    )
    messages = build_messages(
        system_message=system_message,
        history=[],
        user_text=subgoal,
    )

    loop_state = LoopState()
    skill_steps = None
    if primary_skill:
        raw_steps = filter_tool_steps_for_route(primary_skill.tool_steps, route_value)
        if raw_steps:
            skill_steps = bind_tool_steps(raw_steps, slot_bindings)
    discovery = (
        (primary_skill.route_scope or "").lower()
        if primary_skill and (primary_skill.route_scope or "").strip()
        else None
    )
    initialize_loop_plan(
        loop_state,
        goal=subgoal,
        route=route_value,
        tool_steps=skill_steps,
        skill_title=primary_skill.title if primary_skill else "",
        slot_bindings=slot_bindings or None,
        discovery_domain=discovery,
    )
    cache_mcp_tools_from_schemas(loop_state, tools)

    preferred_tool_names: list[str] = []
    if skill_steps:
        preferred_tool_names = [
            str(step.get("tool") or step.get("name") or "").strip()
            for step in skill_steps
            if str(step.get("tool") or step.get("name") or "").strip()
        ]

    trace = TurnTrace(user_text=subgoal, history_len=0, route=route_value)
    max_iter = min(agent_config.max_iterations, _MAX_WORKER_ITERATIONS)
    assistant_text = ""

    for iteration in range(max_iter):
        trace.iterations = iteration + 1
        reset_iteration_flags(loop_state)
        inject_loop_context(messages, loop_state)
        if agent_config.turn_token_budget > 0:
            compact_messages_if_needed(
                messages,
                token_budget=agent_config.turn_token_budget,
            )
        loop_tools = prune_loop_tools(
            tools,
            preferred_names=preferred_tool_names,
            max_tools=agent_config.max_loop_tools,
            include_full_catalog=loop_state.include_full_tool_catalog,
        )

        result = await llm.chat(messages, backend, tools=loop_tools)
        if result.tool_calls:
            messages.append(result.assistant_message)
            async for delta in _process_tool_calls(
                agent_config,
                result.tool_calls,
                mcp_client,
                messages,
                hass=hass,
                exposed_entities=exposed_entities,
                controlled_entity_ids=[],
                loop_state=loop_state,
                trace=trace,
                hint_rules=None,
                reasoning=result.reasoning_content or "",
            ):
                if delta.tool:
                    yield (
                        {"phase": "tool", "subgoal": subgoal, **delta.tool},
                        None,
                    )
            continue

        assistant_text = (result.content or "").strip()
        if assistant_text:
            break

    worker_result = WorkerResult(
        subgoal=subgoal,
        route=route_value,
        assistant_text=assistant_text or "No response.",
        tool_calls=trace.tool_calls,
        tool_errors=trace.tool_errors,
        iterations=trace.iterations,
        skill_title=primary_skill.title if primary_skill else None,
        slot_bindings=slot_bindings,
    )
    yield (
        {
            "phase": "done",
            "subgoal": subgoal,
            "route": route_value,
            "summary": assistant_text[:200],
        },
        worker_result,
    )

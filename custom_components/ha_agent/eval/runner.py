"""Eval suite orchestrator."""

from __future__ import annotations

import time
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant, callback

from ..activity import list_turns
from ..config_helpers import (
    AgentConfig,
    LlmBackend,
    RouterConfig,
    SkillsConfig,
    get_action_backend,
    get_agent_config,
    get_classifier_backend,
    get_email_backend,
    get_llm_backend,
    get_news_backend,
    get_router_config,
)
from ..const import DATA_KEY, LOGGER
from ..llm_client import LlmClient
from ..llm_server import eval_candidate_models, preload_models, probe_server
from ..skills.models import TurnTrace
from .cases import list_eval_cases
from .classifier_runner import run_classifier_case
from .mcp_mock import EvalMcpClient
from .models import EVAL_TASKS, EvalCaseScore, EvalRun, EvalRunState
from .recommender import (
    finalize_settings_recommendation,
    recommend_settings,
    settings_recommendation_to_dict,
)
from .scorer import aggregate_task_scores, score_case
from .store import get_eval_store

EVAL_STATE_KEY = "eval_run_states"


def _trace_from_activity(data: dict[str, Any]) -> TurnTrace:
    return TurnTrace(
        user_text=str(data.get("user_text") or ""),
        history_len=int(data.get("history_len") or 0),
        tool_calls=list(data.get("tool_calls") or []),
        tool_errors=int(data.get("tool_errors") or 0),
        iterations=int(data.get("iterations") or 0),
        fallback=bool(data.get("fallback")),
        assistant_text=str(data.get("assistant_text") or ""),
        outcome=str(data.get("outcome") or ""),
        conversation_id=data.get("conversation_id"),
    )


@callback
def _state_store(hass: HomeAssistant) -> dict[str, EvalRunState]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(EVAL_STATE_KEY, {})


def get_eval_state(hass: HomeAssistant, entry_id: str) -> EvalRunState | None:
    return _state_store(hass).get(entry_id)


def eval_run_to_dict(state: EvalRunState) -> dict[str, Any]:
    run = state.run
    return {
        "id": run.id,
        "entry_id": run.entry_id,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
        "progress": dict(run.progress),
        "server_capabilities": dict(run.server_capabilities),
        "settings_recommendation": dict(run.settings_recommendation),
        "task_scores": [
            {
                "task": item.task,
                "model": item.model,
                "score": item.score,
                "case_count": item.case_count,
                "passed_count": item.passed_count,
                "avg_latency_ms": item.avg_latency_ms,
            }
            for item in run.task_scores
        ],
        "case_scores": [
            {
                "case_id": item.case_id,
                "task": item.task,
                "model": item.model,
                "score": item.score,
                "passed": item.passed,
                "latency_ms": item.latency_ms,
                "iterations": item.iterations,
                "outcome": item.outcome,
                "details": list(item.details),
            }
            for item in run.case_scores
        ],
        "cancel_requested": state.cancel_requested,
    }


async def run_eval_suite(
    hass: HomeAssistant,
    entry_id: str,
    *,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    include_settings: bool = True,
    preload_models_flag: bool = False,
) -> EvalRun:
    """Run benchmark cases across models and recommend settings."""
    from ..agent import run_agent

    store = get_eval_store(hass, entry_id)
    state_store = _state_store(hass)
    existing = state_store.get(entry_id)
    if (
        existing
        and existing.run.status == "running"
        and existing.run.id != "pending"
    ):
        raise RuntimeError("An eval run is already in progress for this entry.")

    run = store.create_run(entry_id)
    state = EvalRunState(run=run)
    state_store[entry_id] = state

    chat_backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    agent_config = get_agent_config(hass.config_entries.async_get_entry(entry_id))
    router_config = get_router_config(hass.config_entries.async_get_entry(entry_id))
    entry = hass.config_entries.async_get_entry(entry_id)
    configured_models = [chat_backend.model]
    action_backend = get_action_backend(entry)
    classifier_backend = get_classifier_backend(entry)
    email_backend = get_email_backend(entry)
    news_backend = get_news_backend(entry)
    if action_backend:
        configured_models.append(action_backend.model)
    if classifier_backend:
        configured_models.append(classifier_backend.model)
    if email_backend:
        configured_models.append(email_backend.model)
    if news_backend:
        configured_models.append(news_backend.model)
    skills_config = SkillsConfig(
        learning_enabled=False,
        auto_save=False,
        use_enabled=False,
        max_inject=0,
    )
    eval_agent_config = AgentConfig(
        system_prompt=agent_config.system_prompt,
        tool_instructions=agent_config.tool_instructions,
        max_iterations=agent_config.max_iterations,
        history_turns=0,
        enable_streaming=False,
        show_reasoning_in_chat=False,
    )

    try:
        async with aiohttp.ClientSession() as session:
            llm = LlmClient(session)
            run.progress = {"phase": "probe"}
            capabilities = await probe_server(session, chat_backend)
            run.server_capabilities = capabilities.to_dict()

            include_unloaded = bool(models)
            candidate_models = eval_candidate_models(
                capabilities,
                configured_models=configured_models,
                explicit_models=models,
                include_unloaded=include_unloaded,
            )
            if not include_unloaded and len(capabilities.models) > len(
                candidate_models
            ):
                LOGGER.info(
                    "Eval limiting to %d loaded/configured models "
                    "(pass models=[...] to benchmark all %d catalog entries)",
                    len(candidate_models),
                    len(capabilities.models),
                )

            if preload_models_flag and candidate_models:
                run.progress = {"phase": "preload"}
                preload_results = await preload_models(
                    session,
                    chat_backend,
                    candidate_models,
                    loaded_models=capabilities.loaded_models,
                )
                failed_preload = [
                    item
                    for item in preload_results
                    if not item.get("ok") and not item.get("skipped")
                ]
                if failed_preload:
                    LOGGER.warning(
                        "Eval preload failed for %d model(s): %s",
                        len(failed_preload),
                        [item.get("model") for item in failed_preload],
                    )
                capabilities = await probe_server(session, chat_backend)
                run.server_capabilities = capabilities.to_dict()

            selected_tasks = list(tasks or EVAL_TASKS)
            cases = list_eval_cases(tasks=selected_tasks)
            if not cases:
                raise RuntimeError("No eval cases matched the requested tasks.")

            for model in candidate_models:
                if state.cancel_requested:
                    run.status = "cancelled"
                    break
                model_backend = LlmBackend(
                    base_url=chat_backend.base_url,
                    model=model,
                    api_key=chat_backend.api_key,
                    max_tokens=min(chat_backend.max_tokens, 1024),
                    temperature=chat_backend.temperature,
                    timeout=chat_backend.timeout,
                    thinking_level="off",
                )
                for case in cases:
                    if state.cancel_requested:
                        run.status = "cancelled"
                        break
                    run.progress = {
                        "phase": "benchmark",
                        "model": model,
                        "task": case.task,
                        "case_id": case.id,
                    }
                    if case.task == "classifier":
                        classifier_backend = LlmBackend(
                            base_url=chat_backend.base_url,
                            model=model,
                            api_key=chat_backend.api_key,
                            max_tokens=256,
                            temperature=0.0,
                            timeout=chat_backend.timeout,
                            thinking_level="off",
                        )
                        try:
                            run.case_scores.append(
                                await run_classifier_case(
                                    llm,
                                    classifier_backend,
                                    case,
                                )
                            )
                        except Exception as err:
                            LOGGER.warning(
                                "Classifier eval case %s failed for %s: %s",
                                case.id,
                                model,
                                err,
                            )
                            run.case_scores.append(
                                EvalCaseScore(
                                    case_id=case.id,
                                    task=case.task,
                                    model=model,
                                    score=0.0,
                                    passed=False,
                                    details=[str(err)],
                                )
                            )
                        continue

                    mcp = EvalMcpClient(
                        session_prompt=(
                            "MCP SERVER INSTRUCTIONS:\n"
                            "Use callTool with exact toolName values."
                        ),
                        responses=list(case.mock_mcp_responses),
                    )
                    conversation_id = f"eval-{run.id}-{case.id}-{model}"
                    started = time.perf_counter()
                    try:
                        async for _delta in run_agent(
                            hass,
                            llm=llm,
                            mcp_client=mcp,
                            backend=model_backend,
                            agent_config=eval_agent_config,
                            router_config=_router_config_for_case(
                                case.task, router_config
                            ),
                            skills_config=skills_config,
                            entry_id=entry_id,
                            conversation_id=conversation_id,
                            user_text=case.user_text,
                            exposed_entities=list(case.exposed_entities),
                        ):
                            pass
                    except Exception as err:
                        LOGGER.warning(
                            "Eval case %s failed for model %s: %s",
                            case.id,
                            model,
                            err,
                        )
                        run.case_scores.append(
                            EvalCaseScore(
                                case_id=case.id,
                                task=case.task,
                                model=model,
                                score=0.0,
                                passed=False,
                                latency_ms=(time.perf_counter() - started) * 1000,
                                details=[str(err)],
                            )
                        )
                        continue
                    latency_ms = (time.perf_counter() - started) * 1000
                    turns, _total = list_turns(hass, entry_id, limit=1)
                    trace = _trace_from_activity(turns[0]) if turns else TurnTrace(
                        user_text=case.user_text,
                        history_len=0,
                    )
                    run.case_scores.append(
                        score_case(
                            case,
                            model=model,
                            trace=trace,
                            latency_ms=latency_ms,
                        )
                    )

            if state.cancel_requested:
                run.status = "cancelled"
            else:
                run.task_scores = aggregate_task_scores(run.case_scores)
                if include_settings and run.task_scores:
                    run.progress = {"phase": "recommend"}
                    recommendation = await recommend_settings(
                        llm,
                        chat_backend,
                        capabilities=capabilities,
                        task_scores=run.task_scores,
                    )
                    recommendation = finalize_settings_recommendation(
                        recommendation,
                        capabilities=capabilities,
                    )
                    run.settings_recommendation = settings_recommendation_to_dict(
                        recommendation
                    )
                run.status = "completed"
    except Exception as err:
        LOGGER.exception("Eval run failed for entry %s: %s", entry_id, err)
        run.status = "failed"
        run.error = str(err)
    finally:
        run.finished_at = time.time()
        run.progress = {"phase": run.status}
        store.finish_run(run)
        state_store[entry_id] = state

    return run


def _router_config_for_case(task: str, router_config: RouterConfig) -> RouterConfig:
    if task != "action":
        return RouterConfig(action_enabled=False, action_backend=None)
    return router_config


async def start_eval_background(
    hass: HomeAssistant,
    entry_id: str,
    *,
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    include_settings: bool = True,
    preload_models_flag: bool = False,
) -> EvalRun:
    """Schedule an eval run and return the placeholder run record."""
    state_store = _state_store(hass)
    if entry_id in state_store and state_store[entry_id].run.status == "running":
        raise RuntimeError("An eval run is already in progress for this entry.")

    placeholder = EvalRun(
        id="pending",
        entry_id=entry_id,
        status="running",
        started_at=time.time(),
        progress={"phase": "starting"},
    )
    state_store[entry_id] = EvalRunState(run=placeholder)

    async def _run() -> None:
        await run_eval_suite(
            hass,
            entry_id,
            models=models,
            tasks=tasks,
            include_settings=include_settings,
            preload_models_flag=preload_models_flag,
        )

    hass.async_create_task(_run())
    return placeholder


def request_eval_cancel(hass: HomeAssistant, entry_id: str) -> bool:
    state = _state_store(hass).get(entry_id)
    if state is None or state.run.status != "running":
        return False
    state.cancel_requested = True
    return True


async def probe_entry_server(hass: HomeAssistant, entry_id: str) -> dict[str, Any]:
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    async with aiohttp.ClientSession() as session:
        caps = await probe_server(session, backend)
    return caps.to_dict()

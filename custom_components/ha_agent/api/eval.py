"""Eval API for the HA Agent console."""

from __future__ import annotations

import time
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..activity import list_turns
from ..config_helpers import get_llm_backend
from ..eval.case_promote import (
    build_case_from_turn,
    find_activity_turn,
    turn_dict_to_trace,
)
from ..eval.case_serde import eval_case_to_dict
from ..eval.cases import list_eval_cases_for_entry
from ..eval.discover_config import get_discover_config
from ..eval.discover_models import propose_models_from_web as discover_propose_models
from ..eval.discover_runner import (
    approve_discover_download,
    approve_discover_trial,
    discover_run_to_dict,
    discover_status_dict,
    get_discover_state,
    start_discover_background,
    start_discover_retry_background,
)
from ..eval.model_registry import get_model_registry
from ..eval.preset import recommendations_to_preset
from ..eval.runner import (
    eval_run_to_dict,
    get_eval_state,
    probe_entry_server,
    request_eval_cancel,
    start_eval_background,
)
from ..eval.server_apply import server_apply_mode, verify_settings_applied
from ..eval.store import get_eval_store
from ..llm_server import (
    apply_props_settings,
    delete_model_from_router,
    load_model,
    preload_models,
    probe_server,
    router_supports_hf_download,
    unload_model,
)
from .config import set_config


async def get_eval_status(hass: HomeAssistant, entry_id: str) -> dict[str, Any]:
    """Return current eval and discover pipeline status."""
    discover_state = get_discover_state(hass, entry_id)
    eval_state = get_eval_state(hass, entry_id)
    discover_active = bool(
        discover_state
        and discover_state.run.status in {"running", "awaiting_approval"}
    )
    eval_active = bool(eval_state and eval_state.run.status == "running")

    store = get_eval_store(hass, entry_id)
    latest = await hass.async_add_executor_job(store.latest_run)
    return {
        "running": discover_active or eval_active,
        "pipeline": (
            "discover"
            if discover_active
            else ("eval" if eval_active else None)
        ),
        "discover": discover_status_dict(discover_state),
        "run": eval_run_to_dict(eval_state)
        if eval_active and eval_state
        else latest,
    }


async def list_eval_runs(
    hass: HomeAssistant,
    entry_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    store = get_eval_store(hass, entry_id)
    return await hass.async_add_executor_job(store.list_runs, limit=limit)


async def get_eval_run(
    hass: HomeAssistant,
    entry_id: str,
    run_id: str,
) -> dict[str, Any]:
    store = get_eval_store(hass, entry_id)
    run = await hass.async_add_executor_job(store.get_run, run_id)
    if run is None:
        raise HomeAssistantError(f"Eval run not found: {run_id}")
    return run


async def start_eval(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a background eval suite."""
    payload = payload or {}
    models = payload.get("models")
    tasks = payload.get("tasks")
    include_settings = bool(payload.get("include_settings", True))
    preload_models_flag = bool(payload.get("preload_models", False))
    parsed_models = [str(item) for item in models] if isinstance(models, list) else None
    parsed_tasks = [str(item) for item in tasks] if isinstance(tasks, list) else None
    await start_eval_background(
        hass,
        entry_id,
        models=parsed_models,
        tasks=parsed_tasks,
        include_settings=include_settings,
        preload_models_flag=preload_models_flag,
    )
    state = get_eval_state(hass, entry_id)
    return {
        "started": True,
        "run": eval_run_to_dict(state) if state else {"status": "running"},
    }


async def cancel_eval(hass: HomeAssistant, entry_id: str) -> dict[str, Any]:
    cancelled = request_eval_cancel(hass, entry_id)
    return {"cancelled": cancelled}


async def probe_server_capabilities(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    return await probe_entry_server(hass, entry_id)


async def apply_eval_recommendations(
    hass: HomeAssistant,
    entry_id: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Apply recommended model assignments to the config entry."""
    store = get_eval_store(hass, entry_id)
    run = await hass.async_add_executor_job(
        store.get_run, run_id
    ) if run_id else await hass.async_add_executor_job(store.latest_run)
    if not run:
        raise HomeAssistantError("No eval run with recommendations found.")
    recommendation = run.get("settings_recommendation") or {}
    assignments = recommendation.get("model_assignments") or {}
    if not isinstance(assignments, dict) or not assignments:
        raise HomeAssistantError("Eval run has no model assignments to apply.")

    updates: dict[str, Any] = {}
    chat = assignments.get("chat") or {}
    action = assignments.get("action") or {}
    email = assignments.get("email") or {}
    news = assignments.get("news") or {}
    classifier = assignments.get("classifier") or {}
    planner = assignments.get("planner") or {}
    verifier = assignments.get("verifier") or {}
    if isinstance(chat, dict) and chat.get("model"):
        updates["llm_model"] = chat["model"]
    if isinstance(action, dict) and action.get("model"):
        updates["action_model_enabled"] = True
        updates["action_llm_model"] = action["model"]
    if isinstance(email, dict) and email.get("model"):
        updates["email_model_enabled"] = True
        updates["email_llm_model"] = email["model"]
    if isinstance(news, dict) and news.get("model"):
        updates["news_model_enabled"] = True
        updates["news_llm_model"] = news["model"]
    if isinstance(classifier, dict) and classifier.get("model"):
        updates["classifier_model_enabled"] = True
        updates["classifier_llm_model"] = classifier["model"]
    role_model = None
    if isinstance(planner, dict) and planner.get("model"):
        role_model = planner["model"]
    if isinstance(verifier, dict) and verifier.get("model"):
        role_model = verifier["model"]
    if role_model and not updates.get("classifier_llm_model"):
        updates["classifier_model_enabled"] = True
        updates["classifier_llm_model"] = role_model

    if not updates:
        raise HomeAssistantError("No supported model assignments to apply.")

    config = await set_config(hass, entry_id, updates)
    return {"applied": updates, "config": config}


async def apply_server_settings(
    hass: HomeAssistant,
    entry_id: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Apply recommended llama.cpp server settings with probe verification."""
    store = get_eval_store(hass, entry_id)
    run = await hass.async_add_executor_job(
        store.get_run, run_id
    ) if run_id else await hass.async_add_executor_job(store.latest_run)
    if not run:
        raise HomeAssistantError("No eval run with settings recommendations found.")
    recommendation = run.get("settings_recommendation") or {}
    items = recommendation.get("recommendations") or []
    if not isinstance(items, list) or not items:
        raise HomeAssistantError("Eval run has no server settings to apply.")

    settings = {
        str(item.get("setting")): str(item.get("value"))
        for item in items
        if isinstance(item, dict) and item.get("setting") and item.get("value")
    }
    if not settings:
        raise HomeAssistantError("Eval run has no valid server settings to apply.")

    preset_ini = recommendation.get("preset_ini") or recommendations_to_preset(items)
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    async with aiohttp.ClientSession() as session:
        before = await probe_server(session, backend)
        mode = server_apply_mode(before)
        if mode == "preset":
            return {
                "mode": "preset",
                "applied": [],
                "failed": [],
                "verification": None,
                "preset_ini": preset_ini,
                "before": before.summary(),
                "after": before.summary(),
                "message": (
                    "Router mode cannot apply settings via POST /props. "
                    "Copy the preset below into your llama.cpp preset volume, "
                    "then restart the llama Docker container."
                ),
                "docker_hint": "docker compose restart <llama-service>",
            }

        results = await apply_props_settings(session, backend, settings)
        after = await probe_server(session, backend)
    applied = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    verification = verify_settings_applied(before, after, settings)
    if not applied and failed:
        raise HomeAssistantError(
            "Server rejected all settings changes. "
            "Try copying the preset and restarting the llama container."
        )
    return {
        "mode": "props",
        "applied": applied,
        "failed": failed,
        "verification": verification,
        "preset_ini": preset_ini,
        "before": before.summary(),
        "after": after.summary(),
        "message": (
            f"Applied {len(applied)} setting(s); "
            f"verified {verification['verified_count']}/{len(settings)} via re-probe."
        ),
    }


async def load_eval_model(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Load one model on the llama.cpp router via HTTP."""
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    async with aiohttp.ClientSession() as session:
        result = await load_model(session, backend, model_id)
        caps = await probe_server(session, backend)
    return {"result": result, "capabilities": caps.to_dict()}


async def unload_eval_model(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Unload one model from the llama.cpp router via HTTP."""
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    async with aiohttp.ClientSession() as session:
        result = await unload_model(session, backend, model_id)
        caps = await probe_server(session, backend)
    return {"result": result, "capabilities": caps.to_dict()}


async def delete_eval_model(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Unload and delete a cached model from the llama.cpp router."""
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    registry = get_model_registry(hass, entry_id)
    async with aiohttp.ClientSession() as session:
        caps = await probe_server(session, backend)
        unload_result = await unload_model(session, backend, model_id)
        delete_result: dict[str, Any] = {
            "ok": False,
            "skipped": True,
            "reason": "Router cache delete API not available.",
        }
        if router_supports_hf_download(caps):
            delete_result = await delete_model_from_router(session, backend, model_id)
        caps_after = await probe_server(session, backend)
    if delete_result.get("ok"):
        registry.mark_deleted(
            model_id,
            notes="Deleted from llama.cpp cache via eval API.",
        )
    return {
        "unload": unload_result,
        "delete": delete_result,
        "capabilities": caps_after.to_dict(),
    }


async def preload_eval_models(
    hass: HomeAssistant,
    entry_id: str,
    model_ids: list[str],
) -> dict[str, Any]:
    """Load multiple eval candidate models before benchmarking."""
    if not model_ids:
        raise HomeAssistantError("No models specified to preload.")
    backend = get_llm_backend(hass.config_entries.async_get_entry(entry_id))
    async with aiohttp.ClientSession() as session:
        before = await probe_server(session, backend)
        results = await preload_models(
            session,
            backend,
            model_ids,
            loaded_models=before.loaded_models,
        )
        after = await probe_server(session, backend)
    loaded = [item for item in results if item.get("ok")]
    failed = [
        item
        for item in results
        if not item.get("ok") and not item.get("skipped")
    ]
    return {
        "results": results,
        "loaded_count": len(loaded),
        "failed_count": len(failed),
        "capabilities": after.to_dict(),
    }


async def export_server_preset(
    hass: HomeAssistant,
    entry_id: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return a llama.cpp preset INI snippet from the latest eval run."""
    store = get_eval_store(hass, entry_id)
    run = await hass.async_add_executor_job(
        store.get_run, run_id
    ) if run_id else await hass.async_add_executor_job(store.latest_run)
    if not run:
        raise HomeAssistantError("No eval run found.")
    recommendation = run.get("settings_recommendation") or {}
    preset = recommendation.get("preset_ini")
    if not preset:
        items = recommendation.get("recommendations") or []
        preset = recommendations_to_preset(items if isinstance(items, list) else [])
    return {"preset_ini": preset}


async def discover_models(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Search the web and return model proposals without starting the pipeline."""
    entry = hass.config_entries.async_get_entry(entry_id)
    config = get_discover_config(entry)
    backend = get_llm_backend(entry)
    registry = get_model_registry(hass, entry_id)
    async with aiohttp.ClientSession() as session:
        from ..llm_client import LlmClient
        from ..llm_server import probe_server

        capabilities = await probe_server(session, backend)
        llm = LlmClient(session)
        proposals = await discover_propose_models(
            session,
            llm,
            backend,
            capabilities=capabilities,
            max_models=config.max_models,
            skip_model_ids={
                model_id
                for model_id in capabilities.models
                if registry.should_skip_download(model_id)
            },
        )
    return {
        "implemented": True,
        "proposals": [
            {
                **item.to_dict(),
                "skip_download": registry.should_skip_download(item.model_id),
            }
            for item in proposals
        ],
    }


async def start_discover(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the full discover/download/trial pipeline."""
    payload = payload or {}
    entry = hass.config_entries.async_get_entry(entry_id)
    config = get_discover_config(entry)
    require_download = payload.get("require_download_approval")
    require_trial = payload.get("require_trial_approval")
    max_models = payload.get("max_models")
    models_dir = payload.get("models_dir")
    if require_download is None:
        require_download = config.require_download_approval
    if require_trial is None:
        require_trial = config.require_trial_approval
    parsed_max = int(max_models) if max_models is not None else None
    parsed_dir = str(models_dir).strip() if models_dir else None
    parsed_webhook = str(payload.get("download_webhook_url") or "").strip() or None
    await start_discover_background(
        hass,
        entry_id,
        require_download_approval=bool(require_download),
        require_trial_approval=bool(require_trial),
        max_models=parsed_max,
        models_dir=parsed_dir or config.models_dir,
        download_webhook_url=parsed_webhook or config.download_webhook_url,
    )
    state = get_discover_state(hass, entry_id)
    return {
        "started": True,
        "discover": discover_run_to_dict(state) if state else {"status": "running"},
    }


async def approve_discover_downloads(
    hass: HomeAssistant,
    entry_id: str,
    model_ids: list[str],
) -> dict[str, Any]:
    """Approve downloading selected proposal models."""
    if not approve_discover_download(hass, entry_id, model_ids):
        raise HomeAssistantError(
            "No discover pipeline is waiting for download approval."
        )
    state = get_discover_state(hass, entry_id)
    return {
        "approved": model_ids,
        "discover": discover_run_to_dict(state) if state else None,
    }


async def approve_discover_trial_run(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
    *,
    approved: bool,
) -> dict[str, Any]:
    """Approve or skip benchmarking one downloaded model."""
    if not approve_discover_trial(
        hass,
        entry_id,
        model_id=model_id,
        approved=approved,
    ):
        raise HomeAssistantError(
            f"No discover pipeline is waiting for trial approval on {model_id}."
        )
    state = get_discover_state(hass, entry_id)
    return {
        "model_id": model_id,
        "approved": approved,
        "discover": discover_run_to_dict(state) if state else None,
    }


async def retry_discover_model(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Re-download, load, and benchmark one discover candidate."""
    model_id = str(model_id or "").strip()
    if not model_id:
        raise HomeAssistantError("No model_id specified for retry.")
    try:
        await start_discover_retry_background(hass, entry_id, model_id)
    except RuntimeError as err:
        raise HomeAssistantError(str(err)) from err
    state = get_discover_state(hass, entry_id)
    return {
        "started": True,
        "model_id": model_id,
        "discover": discover_status_dict(state),
    }


async def mark_model_for_cleanup(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
    *,
    notes: str | None = None,
) -> dict[str, Any]:
    """Unload and delete a model from the llama.cpp router cache."""
    result = await delete_eval_model(hass, entry_id, model_id)
    deleted = bool(result.get("delete", {}).get("ok"))
    if not deleted and notes:
        get_model_registry(hass, entry_id).mark_deleted(model_id, notes=notes)
    return {
        "model_id": model_id,
        "status": "deleted" if deleted else "failed",
        "notes": notes,
        **result,
    }


async def list_eval_cases_api(
    hass: HomeAssistant,
    entry_id: str,
    *,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Return built-in and promoted eval cases for the console."""
    cases = list_eval_cases_for_entry(hass, entry_id, tasks=tasks)
    return {
        "cases": [eval_case_to_dict(case) for case in cases],
        "promoted_count": sum(1 for case in cases if case.source == "promoted"),
    }


async def promote_activity_turn(
    hass: HomeAssistant,
    entry_id: str,
    *,
    timestamp: float,
    task: str | None = None,
) -> dict[str, Any]:
    """Promote one activity turn into a custom eval benchmark case."""
    turns, _total = list_turns(hass, entry_id, limit=200)
    turn = find_activity_turn(turns, timestamp)
    if turn is None:
        raise HomeAssistantError("Activity turn not found. Refresh and try again.")
    trace = turn_dict_to_trace(turn)
    case = build_case_from_turn(
        trace,
        source_timestamp=float(turn.get("timestamp") or timestamp),
        task_override=task,
    )
    store = get_eval_store(hass, entry_id)
    existing = await hass.async_add_executor_job(store.get_custom_case, case.id)
    if existing is not None:
        case = build_case_from_turn(
            trace,
            source_timestamp=float(turn.get("timestamp") or timestamp),
            case_id=f"{case.id}-{int(time.time())}",
            task_override=task,
        )
    saved = await hass.async_add_executor_job(store.save_custom_case, case)
    return {"case": eval_case_to_dict(saved)}


async def delete_eval_case(
    hass: HomeAssistant,
    entry_id: str,
    case_id: str,
) -> dict[str, Any]:
    """Delete a promoted eval case."""
    store = get_eval_store(hass, entry_id)
    deleted = await hass.async_add_executor_job(store.delete_custom_case, case_id)
    if not deleted:
        raise HomeAssistantError(f"Promoted eval case not found: {case_id}")
    return {"deleted": case_id}

"""Eval API for the HA Agent console."""

from __future__ import annotations

from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..config_helpers import get_llm_backend
from ..eval.discover_config import get_discover_config
from ..eval.discover_models import propose_models_from_web as discover_propose_models
from ..eval.discover_runner import (
    approve_discover_download,
    approve_discover_trial,
    discover_run_to_dict,
    get_discover_state,
    start_discover_background,
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
    load_model,
    preload_models,
    probe_server,
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
        "discover": discover_run_to_dict(discover_state)
        if discover_state
        else None,
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


async def mark_model_for_cleanup(
    hass: HomeAssistant,
    entry_id: str,
    model_id: str,
    *,
    notes: str | None = None,
) -> dict[str, Any]:
    """Record that a superseded model was deleted to free disk space."""
    registry = get_model_registry(hass, entry_id)
    registry.mark_deleted(model_id, notes=notes)
    return {"model_id": model_id, "status": "deleted"}

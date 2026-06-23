"""Eval API for the HA Agent console."""

from __future__ import annotations

from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..config_helpers import get_llm_backend
from ..eval.model_registry import get_model_registry, propose_models_from_web
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
    """Return current or latest eval run status."""
    state = get_eval_state(hass, entry_id)
    if state is not None:
        return {
            "running": state.run.status == "running",
            "run": eval_run_to_dict(state),
        }
    store = get_eval_store(hass, entry_id)
    latest = await hass.async_add_executor_job(store.latest_run)
    return {"running": False, "run": latest}


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
    """Phase 3 stub: web model discovery."""
    capabilities = await probe_entry_server(hass, entry_id)
    registry = get_model_registry(hass, entry_id)
    proposals = await propose_models_from_web(
        hass,
        entry_id,
        capabilities_summary=capabilities.get("summary", {}),
    )
    return {
        "implemented": False,
        "proposals": [
            {
                "model_id": item.model_id,
                "source_url": item.source_url,
                "reason": item.reason,
                "expected_benefit": item.expected_benefit,
                "skip_download": registry.should_skip_download(item.model_id),
            }
            for item in proposals
        ],
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

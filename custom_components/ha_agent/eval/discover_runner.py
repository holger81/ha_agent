"""Phase-3 discover, download, trial, and cleanup orchestrator."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant, callback

from ..config_helpers import get_llm_backend
from ..const import DATA_KEY, LOGGER
from ..llm_client import LlmClient
from ..llm_server import (
    delete_model_from_router,
    download_model_on_router,
    load_model_with_progress,
    probe_server,
    router_supports_hf_download,
    unload_model,
    wait_for_model_on_server,
)
from .discover_config import DiscoverConfig, get_discover_config
from .discover_models import propose_models_from_web
from .model_download import (
    delete_local_model_file,
    download_hf_gguf,
    download_via_webhook,
    manual_download_hint,
)
from .model_registry import ModelProposal, get_model_registry
from .models import DiscoverRun, DiscoverRunState
from .runner import benchmark_single_model, get_eval_state
from .store import get_eval_store

DISCOVER_STATE_KEY = "discover_run_states"


class DiscoverCancelled(Exception):
    """Raised when the user cancels a discover pipeline."""


@callback
def _state_store(hass: HomeAssistant) -> dict[str, DiscoverRunState]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(DISCOVER_STATE_KEY, {})


def get_discover_state(hass: HomeAssistant, entry_id: str) -> DiscoverRunState | None:
    return _state_store(hass).get(entry_id)


def discover_run_to_dict(state: DiscoverRunState) -> dict[str, Any]:
    run = state.run
    return {
        "id": run.id,
        "entry_id": run.entry_id,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "progress": dict(run.progress),
        "proposals": list(run.proposals),
        "trial_results": list(run.trial_results),
        "error": run.error,
        "cancel_requested": state.cancel_requested,
        "pending_approval": _pending_approval(state),
        "pending_trial_model_id": state.pending_trial_model_id,
    }


def _pending_approval(state: DiscoverRunState) -> str | None:
    if state.run.status != "awaiting_approval":
        return None
    if not state.download_approval_ready and state.run.progress.get(
        "phase"
    ) == "awaiting_download_approval":
        return "download"
    if not state.trial_approval_ready and state.pending_trial_model_id:
        return "trial"
    return None


def _pipeline_busy(hass: HomeAssistant, entry_id: str) -> bool:
    eval_state = get_eval_state(hass, entry_id)
    if eval_state and eval_state.run.status == "running":
        return True
    discover_state = get_discover_state(hass, entry_id)
    return bool(
        discover_state
        and discover_state.run.status in {"running", "awaiting_approval"}
    )


def _set_progress(
    state: DiscoverRunState,
    *,
    phase: str,
    message: str,
    **extra: Any,
) -> None:
    state.run.progress = {"phase": phase, "message": message, **extra}
    model_id = extra.get("model_id")
    if phase in _CANCELLABLE_PHASES and model_id:
        state.active_model_id = str(model_id)
        state.cancellable_phase = phase
    elif phase not in _CANCELLABLE_PHASES:
        state.active_model_id = None
        state.cancellable_phase = None


async def _abort_cancellable_model(
    session: aiohttp.ClientSession,
    backend,
    state: DiscoverRunState,
) -> None:
    """Unload an in-flight router download/load when the pipeline is cancelled."""
    model_id = state.active_model_id
    if not model_id:
        return
    state.active_model_id = None
    state.cancellable_phase = None
    await unload_model(session, backend, model_id)


def _check_cancel(state: DiscoverRunState) -> None:
    if state.cancel_requested:
        raise DiscoverCancelled


async def _wait_for_download_approval(state: DiscoverRunState) -> None:
    _set_progress(
        state,
        phase="awaiting_download_approval",
        message="Waiting for download approval — select models in the Eval tab.",
    )
    state.run.status = "awaiting_approval"
    while not state.download_approval_ready:
        _check_cancel(state)
        await asyncio.sleep(0.5)


async def _wait_for_trial_approval(state: DiscoverRunState, model_id: str) -> bool:
    state.pending_trial_model_id = model_id
    state.trial_approval_ready = False
    state.trial_approved = None
    _set_progress(
        state,
        phase="awaiting_trial_approval",
        message=f"Waiting for trial approval for {model_id}.",
        model_id=model_id,
    )
    state.run.status = "awaiting_approval"
    while not state.trial_approval_ready:
        _check_cancel(state)
        await asyncio.sleep(0.5)
    approved = bool(state.trial_approved)
    state.pending_trial_model_id = None
    state.trial_approval_ready = False
    state.trial_approved = None
    state.run.status = "running"
    return approved


def _incumbent_baseline(store, incumbent_model: str) -> float | None:
    run = store.latest_run()
    if not run or run.get("status") != "completed":
        return None
    scores = [
        float(item["score"])
        for item in run.get("task_scores", [])
        if item.get("model") == incumbent_model
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _mean_task_score(task_scores: list) -> float:
    if not task_scores:
        return 0.0
    return sum(item.score for item in task_scores) / len(task_scores)


_DELETABLE_DOWNLOAD_MODES = frozenset(
    {"server", "local", "poll", "webhook", "webhook_poll"},
)

_CANCELLABLE_PHASES = frozenset(
    {"downloading", "waiting_for_model", "loading"},
)


async def _cleanup_rejected_model(
    session: aiohttp.ClientSession,
    backend,
    registry,
    proposal: ModelProposal,
    *,
    capabilities,
    local_path: str | None,
) -> None:
    """Unload and remove a rejected trial model from RAM and disk/cache."""
    model_id = proposal.model_id
    await unload_model(session, backend, model_id)

    if proposal.download_mode == "existing":
        return

    if local_path:
        if delete_local_model_file(local_path):
            registry.mark_deleted(
                model_id,
                notes="Deleted local file after eval did not beat incumbent.",
            )
        return

    if proposal.download_mode not in _DELETABLE_DOWNLOAD_MODES:
        return
    if not router_supports_hf_download(capabilities):
        return

    result = await delete_model_from_router(session, backend, model_id)
    if result.get("ok"):
        registry.mark_deleted(
            model_id,
            notes="Deleted from llama.cpp model cache after rejected trial.",
        )
        return
    if result.get("preset_model"):
        LOGGER.info(
            "Skipped router cache delete for preset model %s",
            model_id,
        )
        return
    LOGGER.warning(
        "Could not delete %s from llama.cpp cache: %s",
        model_id,
        result.get("error"),
    )


async def _ensure_model_available(
    session: aiohttp.ClientSession,
    state: DiscoverRunState,
    backend,
    registry,
    proposal: ModelProposal,
    *,
    index: int,
    total: int,
    models_dir: str | None,
    webhook_url: str | None,
) -> tuple[bool, str | None]:
    """Download or wait until the model appears on the llama.cpp server."""
    if not proposal.hf_repo or not proposal.hf_filename:
        return True, proposal.local_path

    model_id = proposal.model_id
    caps = await probe_server(session, backend)
    if model_id in caps.models:
        proposal.download_mode = "existing"
        return True, proposal.local_path

    if router_supports_hf_download(caps):

        def _server_download_progress(
            data: dict[str, Any],
            *,
            _model_id=model_id,
            _index=index,
            _total=total,
        ) -> None:
            bytes_done = data.get("bytes_done")
            bytes_total = data.get("bytes_total")
            if bytes_done is not None and bytes_total:
                pct = int((bytes_done / bytes_total) * 100)
                message = f"Downloading {_model_id} on llama.cpp ({pct}%)…"
            else:
                wait_seconds = data.get("wait_seconds")
                message = (
                    f"Downloading {_model_id} on llama.cpp "
                    f"({wait_seconds or 0}s)…"
                )
            _set_progress(
                state,
                phase="downloading",
                message=message,
                model_id=_model_id,
                current=_index,
                total=_total,
                download_mode="server",
                **data,
            )

        _set_progress(
            state,
            phase="downloading",
            message=f"Requesting llama.cpp download for {model_id}…",
            model_id=model_id,
            current=index,
            total=total,
            download_mode="server",
        )
        result = await download_model_on_router(
            session,
            backend,
            model_id,
            capabilities=caps,
            cancel_check=lambda: state.cancel_requested,
            on_progress=_server_download_progress,
            abort_on_cancel=True,
        )
        if result.get("ok"):
            registry.record_download(
                model_id,
                source_url=proposal.source_url,
                notes=(
                    "Downloaded via llama.cpp router API "
                    f"({result.get('via') or result.get('request_path') or 'server'})."
                ),
            )
            proposal.download_mode = "server"
            return True, None
        if not result.get("unsupported"):
            LOGGER.warning(
                "Router download failed for %s: %s — trying fallback paths.",
                model_id,
                result.get("error"),
            )

    if models_dir:
        dest = Path(models_dir) / proposal.hf_filename
        proposal.local_path = str(dest)
        _set_progress(
            state,
            phase="downloading",
            message=f"Downloading {model_id} to shared path ({index}/{total})…",
            model_id=model_id,
            current=index,
            total=total,
            download_mode="local",
        )

        def _download_progress(
            data: dict[str, Any],
            *,
            _model_id=model_id,
            _index=index,
            _total=total,
        ) -> None:
            mb = (data.get("bytes_done") or 0) // (1024 * 1024)
            _set_progress(
                state,
                phase="downloading",
                message=f"Downloading {_model_id} ({mb} MB)…",
                model_id=_model_id,
                current=_index,
                total=_total,
                bytes_done=data.get("bytes_done"),
                bytes_total=data.get("bytes_total"),
                download_mode="local",
            )

        result = await download_hf_gguf(
            session,
            repo_id=proposal.hf_repo,
            filename=proposal.hf_filename,
            dest_path=dest,
            cancel_check=lambda: state.cancel_requested,
            on_progress=_download_progress,
        )
        if not result.get("ok"):
            return False, None
        registry.record_download(
            model_id,
            source_url=proposal.source_url,
            notes=f"Downloaded to {dest}",
        )
        proposal.download_mode = "local"
        return True, str(dest)

    if webhook_url:
        _set_progress(
            state,
            phase="downloading",
            message=f"Requesting host download for {model_id}…",
            model_id=model_id,
            current=index,
            total=total,
            download_mode="webhook",
        )
        result = await download_via_webhook(
            session,
            webhook_url,
            model_id=model_id,
            hf_repo=proposal.hf_repo,
            hf_filename=proposal.hf_filename,
            source_url=proposal.source_url,
            cancel_check=lambda: state.cancel_requested,
        )
        if not result.get("ok"):
            return False, None

    hints = manual_download_hint(proposal.hf_repo, proposal.hf_filename)
    _set_progress(
        state,
        phase="waiting_for_model",
        message=(
            f"Download {model_id} on the llama host, then wait — "
            f"{hints['hf_url']}"
        ),
        model_id=model_id,
        current=index,
        total=total,
        download_mode="poll" if not webhook_url else "webhook_poll",
        manual_download=hints,
    )

    def _wait_progress(data: dict[str, Any], *, _model_id=model_id) -> None:
        _set_progress(
            state,
            phase="waiting_for_model",
            message=(
                f"Waiting for {_model_id} on llama.cpp "
                f"({data.get('wait_seconds', 0)}s)… "
                "Download on the llama host if you have not already."
            ),
            model_id=_model_id,
            current=index,
            total=total,
            wait_seconds=data.get("wait_seconds"),
            manual_download=hints,
        )

    wait = await wait_for_model_on_server(
        session,
        backend,
        model_id,
        cancel_check=lambda: state.cancel_requested,
        on_progress=_wait_progress,
    )
    if wait.get("ok"):
        registry.record_download(
            model_id,
            source_url=proposal.source_url,
            notes="Model appeared on llama.cpp server.",
        )
        proposal.download_mode = "poll" if not webhook_url else "webhook_poll"
        return True, None
    return False, None


async def run_discover_pipeline(
    hass: HomeAssistant,
    entry_id: str,
    *,
    config: DiscoverConfig,
    require_download_approval: bool | None = None,
    require_trial_approval: bool | None = None,
    max_models: int | None = None,
    models_dir_override: str | None = None,
    download_webhook_override: str | None = None,
) -> DiscoverRun:
    """Discover, optionally download, benchmark, and accept/reject models."""
    require_download = (
        config.require_download_approval
        if require_download_approval is None
        else require_download_approval
    )
    require_trial = (
        config.require_trial_approval
        if require_trial_approval is None
        else require_trial_approval
    )
    max_count = max_models if max_models is not None else config.max_models
    models_dir = (models_dir_override or config.models_dir or "").strip() or None
    webhook_url = (
        (download_webhook_override or config.download_webhook_url or "").strip() or None
    )

    state_store = _state_store(hass)
    run = DiscoverRun(
        id=str(uuid.uuid4()),
        entry_id=entry_id,
        status="running",
        started_at=time.time(),
    )
    state = DiscoverRunState(run=run)
    state_store[entry_id] = state

    entry = hass.config_entries.async_get_entry(entry_id)
    chat_backend = get_llm_backend(entry)
    store = get_eval_store(hass, entry_id)
    registry = get_model_registry(hass, entry_id)
    incumbent_model = chat_backend.model
    incumbent_score = _incumbent_baseline(store, incumbent_model)

    try:
        async with aiohttp.ClientSession() as session:
            llm = LlmClient(session)
            _set_progress(
                state,
                phase="probe",
                message="Probing llama.cpp server…",
            )
            capabilities = await probe_server(session, chat_backend)
            _check_cancel(state)

            _set_progress(
                state,
                phase="discovering",
                message="Searching Hugging Face and ranking candidate models…",
            )
            skip_ids = {
                model_id
                for model_id in capabilities.models
                if registry.should_skip_download(model_id)
            }
            proposals = await propose_models_from_web(
                session,
                llm,
                chat_backend,
                capabilities=capabilities,
                max_models=max_count,
                skip_model_ids=skip_ids,
            )
            _check_cancel(state)

            filtered: list[ModelProposal] = []
            for proposal in proposals:
                if registry.should_skip_download(proposal.model_id):
                    proposal.skip_download = True
                filtered.append(proposal)

            run.proposals = [item.to_dict() for item in filtered]
            if not filtered:
                _set_progress(
                    state,
                    phase="completed",
                    message="No new model proposals found.",
                )
                run.status = "completed"
                return run

            _set_progress(
                state,
                phase="discovered",
                message=(
                    f"Found {len(filtered)} candidate model(s). "
                    + (
                        "Approve downloads to continue."
                        if require_download
                        else "Starting downloads…"
                    )
                ),
                total=len(filtered),
            )

            download_ids: list[str]
            if require_download:
                state.download_approval_ready = False
                state.approved_download_ids = []
                await _wait_for_download_approval(state)
                _check_cancel(state)
                download_ids = list(state.approved_download_ids)
                if not download_ids:
                    _set_progress(
                        state,
                        phase="completed",
                        message="Download approval skipped — no models selected.",
                    )
                    run.status = "completed"
                    return run
            else:
                download_ids = [item.model_id for item in filtered]

            approved_proposals = [
                item for item in filtered if item.model_id in set(download_ids)
            ]
            total = len(approved_proposals)

            for index, proposal in enumerate(approved_proposals, start=1):
                _check_cancel(state)
                ready, local_path = await _ensure_model_available(
                    session,
                    state,
                    chat_backend,
                    registry,
                    proposal,
                    index=index,
                    total=total,
                    models_dir=models_dir,
                    webhook_url=webhook_url,
                )
                _check_cancel(state)
                if not ready:
                    run.trial_results.append(
                        {
                            "model_id": proposal.model_id,
                            "accepted": False,
                            "skipped": True,
                            "reason": (
                                state.run.progress.get("message")
                                or "Model not available on llama.cpp."
                            ),
                        }
                    )
                    continue

                if require_trial:
                    approved_trial = await _wait_for_trial_approval(
                        state,
                        proposal.model_id,
                    )
                    _check_cancel(state)
                    if not approved_trial:
                        run.trial_results.append(
                            {
                                "model_id": proposal.model_id,
                                "accepted": False,
                                "skipped": True,
                                "reason": "Trial skipped by user.",
                            }
                        )
                        continue

                _set_progress(
                    state,
                    phase="loading",
                    message=f"Loading {proposal.model_id} on llama.cpp…",
                    model_id=proposal.model_id,
                    current=index,
                    total=total,
                )

                def _load_progress(
                    data: dict[str, Any],
                    *,
                    _model_id=proposal.model_id,
                    _index=index,
                    _total=total,
                ) -> None:
                    progress = data.get("payload") or data.get("progress") or {}
                    if not isinstance(progress, dict):
                        progress = data
                    bytes_done = progress.get("bytes_done")
                    bytes_total = progress.get("bytes_total")
                    if bytes_done is not None and bytes_total:
                        pct = int((bytes_done / bytes_total) * 100)
                        message = f"Loading {_model_id} on llama.cpp ({pct}%)…"
                    elif data.get("status") == "loading":
                        message = f"Loading {_model_id} on llama.cpp…"
                    else:
                        wait_seconds = data.get("wait_seconds")
                        message = (
                            f"Loading {_model_id} on llama.cpp "
                            f"({wait_seconds or 0}s)…"
                        )
                    _set_progress(
                        state,
                        phase="loading",
                        message=message,
                        model_id=_model_id,
                        current=_index,
                        total=_total,
                        **data,
                    )

                load_result = await load_model_with_progress(
                    session,
                    chat_backend,
                    proposal.model_id,
                    capabilities=capabilities,
                    cancel_check=lambda: state.cancel_requested,
                    on_progress=_load_progress,
                    abort_on_cancel=True,
                )
                _check_cancel(state)
                if not load_result.get("ok"):
                    run.trial_results.append(
                        {
                            "model_id": proposal.model_id,
                            "accepted": False,
                            "skipped": True,
                            "reason": load_result.get("error") or "Load failed.",
                        }
                    )
                    continue

                model_id = proposal.model_id
                current_index = index
                total_count = total

                def _bench_progress(
                    data: dict[str, Any],
                    *,
                    _model_id=model_id,
                    _index=current_index,
                    _total=total_count,
                ) -> None:
                    _set_progress(
                        state,
                        phase="benchmarking",
                        message=(
                            f"Benchmarking {_model_id} — "
                            f"{data.get('task')} / {data.get('case_id')}…"
                        ),
                        model_id=_model_id,
                        current=_index,
                        total=_total,
                        **data,
                    )

                _set_progress(
                    state,
                    phase="benchmarking",
                    message=f"Running eval suite for {proposal.model_id}…",
                    model_id=proposal.model_id,
                    current=index,
                    total=total,
                )
                _case_scores, task_scores = await benchmark_single_model(
                    hass,
                    entry_id,
                    proposal.model_id,
                    cancel_check=lambda: state.cancel_requested,
                    progress_callback=_bench_progress,
                )
                _check_cancel(state)

                mean_score = _mean_task_score(task_scores)
                baseline = incumbent_score if incumbent_score is not None else 0.55
                accepted = mean_score >= baseline
                _set_progress(
                    state,
                    phase="comparing",
                    message=(
                        f"{proposal.model_id}: score {mean_score:.2f} vs "
                        f"incumbent {baseline:.2f} — "
                        f"{'accepted' if accepted else 'rejected'}."
                    ),
                    model_id=proposal.model_id,
                    mean_score=mean_score,
                    incumbent_score=baseline,
                )

                registry.record_eval_result(
                    proposal.model_id,
                    eval_score=mean_score,
                    eval_run_id=run.id,
                    accepted=accepted,
                    notes=(
                        f"mean={mean_score:.3f} incumbent={baseline:.3f}"
                        if incumbent_score is not None
                        else f"mean={mean_score:.3f} (no incumbent baseline)"
                    ),
                )
                run.trial_results.append(
                    {
                        "model_id": proposal.model_id,
                        "mean_score": mean_score,
                        "incumbent_score": baseline,
                        "accepted": accepted,
                        "task_scores": [
                            {
                                "task": item.task,
                                "score": item.score,
                                "passed_count": item.passed_count,
                                "case_count": item.case_count,
                            }
                            for item in task_scores
                        ],
                    }
                )

                _set_progress(
                    state,
                    phase="cleanup",
                    message=f"Cleaning up {proposal.model_id}…",
                    model_id=proposal.model_id,
                )
                if accepted:
                    await unload_model(session, chat_backend, proposal.model_id)
                else:
                    await _cleanup_rejected_model(
                        session,
                        chat_backend,
                        registry,
                        proposal,
                        capabilities=capabilities,
                        local_path=local_path,
                    )

            run.status = "completed"
            accepted_count = sum(
                1 for item in run.trial_results if item.get("accepted")
            )
            _set_progress(
                state,
                phase="completed",
                message=(
                    f"Discover pipeline finished — "
                    f"{accepted_count} model(s) accepted out of "
                    f"{len(run.trial_results)} trial(s)."
                ),
            )
    except DiscoverCancelled:
        run.status = "cancelled"
        _set_progress(state, phase="cancelled", message="Discover pipeline cancelled.")
        try:
            async with aiohttp.ClientSession() as abort_session:
                await _abort_cancellable_model(abort_session, chat_backend, state)
        except Exception as err:
            LOGGER.debug("Discover cancel cleanup failed: %s", err)
    except Exception as err:
        LOGGER.exception("Discover pipeline failed for %s: %s", entry_id, err)
        run.status = "failed"
        run.error = str(err)
        _set_progress(state, phase="failed", message=str(err))
    finally:
        run.finished_at = time.time()
        state_store[entry_id] = state

    return run


async def start_discover_background(
    hass: HomeAssistant,
    entry_id: str,
    *,
    require_download_approval: bool | None = None,
    require_trial_approval: bool | None = None,
    max_models: int | None = None,
    models_dir: str | None = None,
    download_webhook_url: str | None = None,
) -> DiscoverRun:
    """Schedule a discover pipeline."""
    if _pipeline_busy(hass, entry_id):
        raise RuntimeError("An eval or discover pipeline is already running.")

    entry = hass.config_entries.async_get_entry(entry_id)
    config = get_discover_config(entry)
    placeholder = DiscoverRun(
        id="pending",
        entry_id=entry_id,
        status="running",
        started_at=time.time(),
        progress={"phase": "starting", "message": "Starting discover pipeline…"},
    )
    _state_store(hass)[entry_id] = DiscoverRunState(run=placeholder)

    async def _run() -> None:
        await run_discover_pipeline(
            hass,
            entry_id,
            config=config,
            require_download_approval=require_download_approval,
            require_trial_approval=require_trial_approval,
            max_models=max_models,
            models_dir_override=models_dir,
            download_webhook_override=download_webhook_url,
        )

    hass.async_create_task(_run())
    return placeholder


def approve_discover_download(
    hass: HomeAssistant,
    entry_id: str,
    model_ids: list[str],
) -> bool:
    """Approve downloading selected proposal model ids."""
    state = get_discover_state(hass, entry_id)
    if state is None or state.run.status != "awaiting_approval":
        return False
    state.approved_download_ids = list(model_ids)
    state.download_approval_ready = True
    state.run.status = "running"
    return True


def approve_discover_trial(
    hass: HomeAssistant,
    entry_id: str,
    *,
    model_id: str,
    approved: bool,
) -> bool:
    """Approve or skip a trial for one model."""
    state = get_discover_state(hass, entry_id)
    if state is None or state.pending_trial_model_id != model_id:
        return False
    state.trial_approved = approved
    state.trial_approval_ready = True
    state.run.status = "running"
    return True


def request_discover_cancel(hass: HomeAssistant, entry_id: str) -> bool:
    state = get_discover_state(hass, entry_id)
    if state is None or state.run.status not in {"running", "awaiting_approval"}:
        return False
    state.cancel_requested = True
    return True


def request_pipeline_cancel(hass: HomeAssistant, entry_id: str) -> bool:
    """Cancel whichever eval or discover pipeline is active."""
    eval_state = get_eval_state(hass, entry_id)
    if eval_state is not None and eval_state.run.status == "running":
        eval_state.cancel_requested = True
        return True
    return request_discover_cancel(hass, entry_id)

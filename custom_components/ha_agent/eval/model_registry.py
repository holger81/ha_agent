"""Phase-3 hooks for web-sourced model discovery and download lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import LOGGER
from .store import EvalStore, get_eval_store


@dataclass(slots=True)
class ModelProposal:
    """A candidate model suggested from web research (phase 3)."""

    model_id: str
    source_url: str | None
    reason: str
    expected_benefit: str = ""


class ModelRegistry:
    """Track downloaded eval candidates and avoid repeat downloads."""

    def __init__(self, store: EvalStore) -> None:
        self._store = store

    def should_skip_download(self, model_id: str) -> bool:
        return self._store.should_skip_download(model_id)

    def record_download(
        self,
        model_id: str,
        *,
        source_url: str | None = None,
        notes: str | None = None,
    ) -> None:
        self._store.record_model_download(
            model_id,
            source_url=source_url,
            status="downloaded",
            notes=notes,
        )

    def record_eval_result(
        self,
        model_id: str,
        *,
        eval_score: float,
        eval_run_id: str,
        accepted: bool,
        notes: str | None = None,
    ) -> None:
        status = "accepted" if accepted else "rejected"
        self._store.record_model_download(
            model_id,
            eval_score=eval_score,
            eval_run_id=eval_run_id,
            status=status,
            notes=notes,
        )
        if not accepted:
            self._store.mark_model_deleted(
                model_id,
                status="deleted_after_eval",
                notes=notes or "Removed after eval did not beat incumbent.",
            )

    def mark_deleted(self, model_id: str, *, notes: str | None = None) -> None:
        self._store.mark_model_deleted(model_id, notes=notes)


async def propose_models_from_web(
    _hass: HomeAssistant,
    _entry_id: str,
    *,
    capabilities_summary: dict[str, Any],
) -> list[ModelProposal]:
    """Phase 3: search the web and propose models for this setup.

    Not implemented yet — returns an empty list. When wired up, this should:
    - search HuggingFace / model cards / community benchmarks
    - respect ``should_skip_download`` history
    - download via HF API into the llama models volume; load/unload via HTTP
    """
    LOGGER.info(
        "Model web discovery not implemented yet (capabilities=%s)",
        capabilities_summary.get("model_count"),
    )
    return []


def get_model_registry(hass: HomeAssistant, entry_id: str) -> ModelRegistry:
    return ModelRegistry(get_eval_store(hass, entry_id))

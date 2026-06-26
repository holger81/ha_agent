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
    hf_repo: str = ""
    hf_filename: str = ""
    skip_download: bool = False
    local_path: str | None = None
    download_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_url": self.source_url,
            "reason": self.reason,
            "expected_benefit": self.expected_benefit,
            "hf_repo": self.hf_repo,
            "hf_filename": self.hf_filename,
            "skip_download": self.skip_download,
            "local_path": self.local_path,
            "download_mode": self.download_mode,
        }


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

    def clear_for_retry(self, model_id: str) -> None:
        """Forget prior reject/skip so discover can download this model again."""
        self._store.clear_model_download_record(model_id)


async def propose_models_from_web(
    hass: HomeAssistant,
    entry_id: str,
    *,
    capabilities_summary: dict[str, Any],
) -> list[ModelProposal]:
    """Legacy stub entry point — use discover_models.propose_models_from_web."""
    LOGGER.info(
        "Use discover pipeline for web proposals (capabilities=%s)",
        capabilities_summary.get("model_count"),
    )
    return []


def get_model_registry(hass: HomeAssistant, entry_id: str) -> ModelRegistry:
    return ModelRegistry(get_eval_store(hass, entry_id))

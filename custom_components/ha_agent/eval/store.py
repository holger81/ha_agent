"""SQLite persistence for eval runs and model benchmark history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import DATA_KEY
from .models import EvalCase, EvalRun
from .scorer import case_score_to_dict, task_score_to_dict

EVAL_STORE_KEY = "eval_stores"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    server_capabilities_json TEXT,
    settings_recommendation_json TEXT,
    task_scores_json TEXT,
    case_scores_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS model_benchmarks (
    model_id TEXT NOT NULL,
    task TEXT NOT NULL,
    case_id TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    latency_ms REAL,
    run_id TEXT NOT NULL,
    run_at REAL NOT NULL,
    details_json TEXT,
    PRIMARY KEY (model_id, task, case_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_model_benchmarks_model
    ON model_benchmarks(model_id, run_at DESC);

CREATE TABLE IF NOT EXISTS model_downloads (
    model_id TEXT PRIMARY KEY,
    source_url TEXT,
    downloaded_at REAL,
    eval_score REAL,
    eval_run_id TEXT,
    deleted_at REAL,
    status TEXT NOT NULL DEFAULT 'unknown',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS custom_eval_cases (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    promoted_at REAL NOT NULL,
    source_timestamp REAL,
    source_conversation_id TEXT,
    case_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_custom_eval_cases_task
    ON custom_eval_cases(task, promoted_at DESC);
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class EvalStore:
    """Per-config-entry eval history and model benchmark cache."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        storage = Path(hass.config.path(".storage"))
        return storage / f"ha_agent_eval_{entry_id}.db"

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def create_run(self, entry_id: str) -> EvalRun:
        run = EvalRun(
            id=str(uuid.uuid4()),
            entry_id=entry_id,
            status="running",
            started_at=time.time(),
        )
        conn = self._connection()
        conn.execute(
            "INSERT INTO eval_runs "
            "(id, entry_id, status, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.entry_id, run.status, run.started_at),
        )
        conn.commit()
        return run

    def finish_run(self, run: EvalRun) -> None:
        conn = self._connection()
        conn.execute(
            "UPDATE eval_runs SET status = ?, finished_at = ?, "
            "server_capabilities_json = ?, settings_recommendation_json = ?, "
            "task_scores_json = ?, case_scores_json = ?, error = ? "
            "WHERE id = ?",
            (
                run.status,
                run.finished_at,
                _json_dumps(run.server_capabilities),
                _json_dumps(run.settings_recommendation),
                _json_dumps([task_score_to_dict(item) for item in run.task_scores]),
                _json_dumps([case_score_to_dict(item) for item in run.case_scores]),
                run.error,
                run.id,
            ),
        )
        now = time.time()
        for item in run.case_scores:
            conn.execute(
                "INSERT OR REPLACE INTO model_benchmarks "
                "(model_id, task, case_id, score, passed, latency_ms, "
                "run_id, run_at, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.model,
                    item.task,
                    item.case_id,
                    item.score,
                    int(item.passed),
                    item.latency_ms,
                    run.id,
                    now,
                    _json_dumps(item.details),
                ),
            )
        conn.commit()

    def list_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute(
                "SELECT id, entry_id, status, started_at, finished_at, error "
                "FROM eval_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = (
            self._connection()
            .execute(
                "SELECT * FROM eval_runs WHERE id = ?",
                (run_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return {
            "id": row["id"],
            "entry_id": row["entry_id"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "server_capabilities": _json_loads(row["server_capabilities_json"], {}),
            "settings_recommendation": _json_loads(
                row["settings_recommendation_json"], {}
            ),
            "task_scores": _json_loads(row["task_scores_json"], []),
            "case_scores": _json_loads(row["case_scores_json"], []),
        }

    def latest_run(self) -> dict[str, Any] | None:
        row = (
            self._connection()
            .execute(
                "SELECT id FROM eval_runs ORDER BY started_at DESC LIMIT 1",
            )
            .fetchone()
        )
        return self.get_run(row["id"]) if row else None

    def model_benchmark_history(self, model_id: str) -> list[dict[str, Any]]:
        rows = (
            self._connection()
            .execute(
                "SELECT task, case_id, score, passed, latency_ms, run_id, run_at "
                "FROM model_benchmarks WHERE model_id = ? ORDER BY run_at DESC",
                (model_id,),
            )
            .fetchall()
        )
        return [dict(row) for row in rows]

    def list_model_scores(
        self,
        *,
        sort_by: str = "mean_score",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        """Return latest eval scores for every previously benchmarked model.

        Uses each model's most recent ``model_benchmarks`` run (by ``run_at``),
        aggregates case scores into per-task means, then ranks models.
        """
        from .models import EVAL_TASKS

        conn = self._connection()
        rows = conn.execute(
            "SELECT model_id, task, case_id, score, passed, latency_ms, "
            "run_id, run_at FROM model_benchmarks ORDER BY run_at DESC, run_id DESC"
        ).fetchall()
        if not rows:
            return []

        latest_run_by_model: dict[str, str] = {}
        latest_at_by_model: dict[str, float] = {}
        for row in rows:
            model_id = str(row["model_id"])
            if model_id in latest_run_by_model:
                continue
            latest_run_by_model[model_id] = str(row["run_id"])
            latest_at_by_model[model_id] = float(row["run_at"] or 0.0)

        # (model_id, task) -> list of case rows from that model's latest run
        grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in rows:
            model_id = str(row["model_id"])
            if str(row["run_id"]) != latest_run_by_model[model_id]:
                continue
            grouped.setdefault((model_id, str(row["task"])), []).append(row)

        download_rows = conn.execute(
            "SELECT model_id, status, eval_score FROM model_downloads"
        ).fetchall()
        downloads = {str(row["model_id"]): dict(row) for row in download_rows}

        models: list[dict[str, Any]] = []
        for model_id, run_id in latest_run_by_model.items():
            task_scores: list[dict[str, Any]] = []
            task_map: dict[str, float] = {}
            total_cases = 0
            total_passed = 0
            for task in EVAL_TASKS:
                items = grouped.get((model_id, task))
                if not items:
                    continue
                latencies = [
                    float(item["latency_ms"])
                    for item in items
                    if item["latency_ms"] is not None
                ]
                passed_count = sum(1 for item in items if item["passed"])
                score = sum(float(item["score"]) for item in items) / len(items)
                task_scores.append(
                    {
                        "task": task,
                        "model": model_id,
                        "score": score,
                        "case_count": len(items),
                        "passed_count": passed_count,
                        "avg_latency_ms": (
                            sum(latencies) / len(latencies) if latencies else None
                        ),
                    }
                )
                task_map[task] = score
                total_cases += len(items)
                total_passed += passed_count

            if not task_scores:
                continue
            mean_score = sum(item["score"] for item in task_scores) / len(task_scores)
            download = downloads.get(model_id) or {}
            models.append(
                {
                    "model_id": model_id,
                    "mean_score": mean_score,
                    "task_scores": task_scores,
                    "scores_by_task": task_map,
                    "case_count": total_cases,
                    "passed_count": total_passed,
                    "last_run_id": run_id,
                    "last_run_at": latest_at_by_model.get(model_id),
                    "download_status": download.get("status"),
                }
            )

        sort_key = (sort_by or "mean_score").strip().lower()
        task_sort = (
            sort_key.removeprefix("task:")
            if sort_key.startswith("task:")
            else (sort_key if sort_key in EVAL_TASKS else None)
        )

        def _sort_key(item: dict[str, Any]) -> tuple[int, float]:
            """(missing_flag, ordered_value) — missing task scores always last."""
            if sort_key == "last_run_at":
                value = float(item.get("last_run_at") or 0.0)
                ordered = -value if descending else value
                return (0, ordered)
            if task_sort:
                scores = item.get("scores_by_task") or {}
                if task_sort not in scores:
                    return (1, 0.0)
                value = float(scores[task_sort])
                ordered = -value if descending else value
                return (0, ordered)
            value = float(item.get("mean_score") or 0.0)
            ordered = -value if descending else value
            return (0, ordered)

        models.sort(key=_sort_key)
        for index, item in enumerate(models, start=1):
            item["rank"] = index
        return models

    def has_benchmarked_model(self, model_id: str) -> bool:
        row = (
            self._connection()
            .execute(
                "SELECT 1 FROM model_benchmarks WHERE model_id = ? LIMIT 1",
                (model_id,),
            )
            .fetchone()
        )
        return row is not None

    def record_model_download(
        self,
        model_id: str,
        *,
        source_url: str | None = None,
        eval_score: float | None = None,
        eval_run_id: str | None = None,
        status: str = "downloaded",
        notes: str | None = None,
    ) -> None:
        """Track a model download for phase-3 deduplication."""
        now = time.time()
        conn = self._connection()
        conn.execute(
            "INSERT INTO model_downloads "
            "(model_id, source_url, downloaded_at, eval_score, eval_run_id, "
            "status, notes) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET "
            "source_url = excluded.source_url, "
            "downloaded_at = excluded.downloaded_at, "
            "eval_score = excluded.eval_score, "
            "eval_run_id = excluded.eval_run_id, "
            "status = excluded.status, "
            "notes = excluded.notes, "
            "deleted_at = NULL",
            (
                model_id,
                source_url,
                now,
                eval_score,
                eval_run_id,
                status,
                notes,
            ),
        )
        conn.commit()

    def mark_model_deleted(
        self,
        model_id: str,
        *,
        status: str = "deleted_after_eval",
        notes: str | None = None,
    ) -> None:
        conn = self._connection()
        conn.execute(
            "UPDATE model_downloads SET deleted_at = ?, status = ?, notes = ? "
            "WHERE model_id = ?",
            (time.time(), status, notes, model_id),
        )
        conn.commit()

    def clear_model_download_record(self, model_id: str) -> None:
        """Remove registry history so a model can be downloaded and trialed again."""
        conn = self._connection()
        conn.execute("DELETE FROM model_downloads WHERE model_id = ?", (model_id,))
        conn.commit()

    def should_skip_download(self, model_id: str) -> bool:
        """Return True when a model was already downloaded and benchmarked."""
        row = (
            self._connection()
            .execute(
                "SELECT status, deleted_at, eval_score FROM model_downloads "
                "WHERE model_id = ?",
                (model_id,),
            )
            .fetchone()
        )
        if row is None:
            return False
        if row["deleted_at"] is not None:
            return True
        return row["status"] in {"rejected", "deleted_after_eval"}

    def list_custom_cases(self) -> list[EvalCase]:
        """Return promoted eval cases newest-first."""
        from .case_serde import eval_case_from_dict

        rows = (
            self._connection()
            .execute(
                "SELECT case_json FROM custom_eval_cases ORDER BY promoted_at DESC",
            )
            .fetchall()
        )
        cases: list[EvalCase] = []
        for row in rows:
            payload = _json_loads(row["case_json"], {})
            if isinstance(payload, dict) and payload.get("id"):
                cases.append(eval_case_from_dict(payload))
        return cases

    def save_custom_case(self, case: EvalCase) -> EvalCase:
        """Persist one promoted eval case."""
        from .case_serde import eval_case_to_dict

        if case.source != "promoted":
            raise ValueError("Only promoted cases can be saved as custom eval cases.")
        conn = self._connection()
        payload = eval_case_to_dict(case)
        conn.execute(
            "INSERT OR REPLACE INTO custom_eval_cases "
            "(id, task, promoted_at, source_timestamp, source_conversation_id, "
            "case_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                case.id,
                case.task,
                float(case.promoted_at or time.time()),
                case.source_timestamp,
                case.source_conversation_id,
                _json_dumps(payload),
            ),
        )
        conn.commit()
        return case

    def delete_custom_case(self, case_id: str) -> bool:
        """Delete a promoted eval case. Built-in ids are not stored here."""
        conn = self._connection()
        cursor = conn.execute(
            "DELETE FROM custom_eval_cases WHERE id = ?",
            (case_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_custom_case(self, case_id: str) -> EvalCase | None:
        from .case_serde import eval_case_from_dict

        row = (
            self._connection()
            .execute(
                "SELECT case_json FROM custom_eval_cases WHERE id = ?",
                (case_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        payload = _json_loads(row["case_json"], {})
        if not isinstance(payload, dict) or not payload.get("id"):
            return None
        return eval_case_from_dict(payload)


def get_eval_store(hass: HomeAssistant, entry_id: str) -> EvalStore:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, EvalStore] = domain_data.setdefault(EVAL_STORE_KEY, {})
    if entry_id not in stores:
        store = EvalStore(EvalStore.db_path_for_entry(hass, entry_id))
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_eval_store(hass: HomeAssistant, entry_id: str) -> None:
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, EvalStore] = domain_data.get(EVAL_STORE_KEY, {})
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()


async def async_get_eval_store(hass: HomeAssistant, entry_id: str) -> EvalStore:
    return get_eval_store(hass, entry_id)

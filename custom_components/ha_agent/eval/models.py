"""Data models for the HA Agent eval system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVAL_TASKS = ("chat", "action", "email", "news", "classifier")


@dataclass(slots=True)
class EvalCase:
    """One deterministic benchmark prompt for a task route."""

    id: str
    task: str
    user_text: str
    exposed_entities: list[dict[str, Any]] = field(default_factory=list)
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] | None = None
    expected_text_contains: list[str] = field(default_factory=list)
    expected_playbook_route: str | None = None
    mock_mcp_responses: list[str] = field(default_factory=list)
    max_iterations: int = 6


@dataclass(slots=True)
class EvalCaseScore:
    """Score for one model on one benchmark case."""

    case_id: str
    task: str
    model: str
    score: float
    passed: bool
    latency_ms: float | None = None
    iterations: int = 0
    outcome: str = ""
    tool_match: bool = False
    text_match: bool = False
    details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalTaskScore:
    """Aggregate score for one model on one task."""

    task: str
    model: str
    score: float
    case_count: int
    passed_count: int
    avg_latency_ms: float | None = None


@dataclass(slots=True)
class SettingsRecommendation:
    """LLM-derived llama.cpp server tuning recommendation."""

    summary: str
    recommendations: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRun:
    """One eval suite execution."""

    id: str
    entry_id: str
    status: str
    started_at: float
    finished_at: float | None = None
    server_capabilities: dict[str, Any] = field(default_factory=dict)
    settings_recommendation: dict[str, Any] = field(default_factory=dict)
    task_scores: list[EvalTaskScore] = field(default_factory=list)
    case_scores: list[EvalCaseScore] = field(default_factory=list)
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRunState:
    """In-memory status for a background eval run."""

    run: EvalRun
    cancel_requested: bool = False


@dataclass(slots=True)
class DiscoverRun:
    """One phase-3 discover/download/trial pipeline execution."""

    id: str
    entry_id: str
    status: str
    started_at: float
    finished_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    trial_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class DiscoverRunState:
    """In-memory status for a background discover pipeline."""

    run: DiscoverRun
    cancel_requested: bool = False
    download_approval_ready: bool = False
    approved_download_ids: list[str] = field(default_factory=list)
    trial_approval_ready: bool = False
    trial_approved: bool | None = None
    pending_trial_model_id: str | None = None

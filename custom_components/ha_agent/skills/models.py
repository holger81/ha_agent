"""Data models for HA Agent skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillSlot:
    """A fillable parameter in a parameterized skill workflow."""

    name: str
    description: str = ""
    source: str = "user"
    default: str | None = None


@dataclass(slots=True)
class Skill:
    """A reusable workflow learned from a successful multi-step turn."""

    id: str
    slug: str
    title: str
    description: str
    triggers: list[str]
    body: str
    tool_steps: list[dict[str, Any]]
    enabled: bool = True
    created_at: float = 0.0
    last_used_at: float | None = None
    use_count: int = 0
    success_count: int = 0
    last_improved_at: float | None = None
    last_evaluation_at: float | None = None
    version: int = 1
    slots: list[SkillSlot] = field(default_factory=list)
    preconditions: str = ""
    parent_id: str | None = None
    route_scope: str | None = None
    score: float = 1.0
    is_builtin: bool = False


@dataclass(slots=True)
class SkillIndexRow:
    """Lightweight skill row returned from FTS discovery."""

    id: str
    slug: str
    title: str
    description: str
    rank: float


@dataclass(slots=True)
class TurnTrace:
    """Captured metrics for one Assist agent turn."""

    user_text: str
    history_len: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: int = 0
    iterations: int = 0
    fallback: bool = False
    assistant_text: str = ""
    matched_skill_ids: list[str] = field(default_factory=list)
    controlled_entity_ids: list[str] = field(default_factory=list)
    conversation_id: str | None = None
    outcome: str = ""
    verification_notes: list[str] = field(default_factory=list)
    route: str = ""
    exposed_entities: list[dict[str, Any]] = field(default_factory=list)
    complexity: str = "simple"
    slot_bindings: dict[str, str] = field(default_factory=dict)
    verifier_verdict: str = ""
    verifier_detail: str = ""
    subtask_results: list[dict[str, Any]] = field(default_factory=list)
    orchestration_plan: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SkillRunResult:
    """Outcome of executing a turn with matched skills."""

    skill_id: str
    iterations: int
    tool_errors: int
    followed_steps: bool
    succeeded: bool


@dataclass(slots=True)
class PendingSkillDraft:
    """Draft waiting for user confirmation before saving."""

    entry_id: str
    conversation_id: str
    trace: TurnTrace
    history: list[dict[str, str]]
    skill_draft: SkillDraft | None = None
    observer_reason: str = ""


@dataclass(slots=True)
class SkillDraft:
    """LLM-distilled skill payload before persistence."""

    title: str
    description: str
    triggers: list[str]
    body: str
    tool_steps: list[dict[str, Any]]
    slots: list[SkillSlot] = field(default_factory=list)
    preconditions: str = ""
    parent_id: str | None = None
    route_scope: str | None = None

"""Data models for HA Agent skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Hard caps for learned skill text that ends up in the system prompt.
MAX_SKILL_TITLE_CHARS = 64
MAX_SKILL_DESCRIPTION_CHARS = 512
MAX_SKILL_BODY_CHARS = 4000
MAX_SKILL_TRIGGERS = 12
MAX_SKILL_TRIGGER_CHARS = 80
MAX_ADDITIONAL_WORKFLOW_SECTIONS = 3

# Lines that look like chat-role / prompt-injection markers are dropped from
# skill bodies before they can be rendered into the system prompt.
_INSTRUCTION_MARKER_LINE = re.compile(
    r"^\s*(?:"
    r"(?:system|assistant|user|developer|tool)\s*:"
    r"|<\|"
    r"|\[/?inst\]"
    r"|<<\s*/?sys\s*>>"
    r"|system\s*\(internal"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"
    r")",
    re.IGNORECASE,
)


def cap_triggers(raw: Any, *, limit: int = MAX_SKILL_TRIGGERS) -> list[str]:
    """Return deduplicated, length-capped trigger phrases (at most ``limit``)."""
    if isinstance(raw, str):
        items: list[Any] = [raw]
    elif isinstance(raw, list | tuple):
        items = list(raw)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = " ".join(str(item).split())[:MAX_SKILL_TRIGGER_CHARS].strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def sanitize_skill_body(body: Any) -> str:
    """Strip role/instruction marker lines and cap the body length."""
    text = str(body or "")
    kept = [
        line for line in text.splitlines() if not _INSTRUCTION_MARKER_LINE.match(line)
    ]
    cleaned = "\n".join(kept).strip()
    return cleaned[:MAX_SKILL_BODY_CHARS].rstrip()


def validate_skill_fields(skill: Skill) -> Skill:
    """Normalize and cap skill text fields in place before persisting.

    Title/description/body are coerced to stripped strings and truncated;
    triggers are deduplicated and capped; role-marker lines are removed from
    the body. Returns the same ``Skill`` for chaining.
    """
    skill.title = " ".join(str(skill.title or "").split())[:MAX_SKILL_TITLE_CHARS]
    skill.description = str(skill.description or "").strip()[
        :MAX_SKILL_DESCRIPTION_CHARS
    ]
    skill.body = sanitize_skill_body(skill.body)
    skill.triggers = cap_triggers(skill.triggers)
    skill.tool_steps = [
        step for step in (skill.tool_steps or []) if isinstance(step, dict)
    ]
    skill.preconditions = str(skill.preconditions or "")[:MAX_SKILL_BODY_CHARS]
    return skill


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
    llm_model: str | None = None
    llm_base_url: str | None = None
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
    domain_hint: str | None = None
    route_method: str = ""
    classifier_summary: str = ""
    stuck_kind: str = ""
    reasoning_stalls: int = 0
    empty_responses: int = 0
    exposed_entities: list[dict[str, Any]] = field(default_factory=list)
    complexity: str = "simple"
    slot_bindings: dict[str, str] = field(default_factory=dict)
    verifier_verdict: str = ""
    verifier_detail: str = ""
    subtask_results: list[dict[str, Any]] = field(default_factory=list)
    orchestration_plan: list[dict[str, Any]] = field(default_factory=list)
    matched_learned_skill_ids: list[str] = field(default_factory=list)
    skill_followed: bool | None = None
    skill_plan_override: bool = False
    skill_plan_override_reason: str = ""
    explore_mode: bool = False
    recovery_hints: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    plan_progress: list[dict[str, str]] = field(default_factory=list)
    agent_user_id: str = ""
    agent_user_display_name: str = ""
    agent_user_kind: str = ""
    identity_source: str = ""
    identity_ha_user_id: str | None = None
    identity_override_by_ha_user_id: str | None = None
    identity_speaker_confidence: float | None = None
    identity_original_user_id: str = ""
    identity_original_display_name: str = ""
    identity_corrected_by_ha_user_id: str | None = None


@dataclass(slots=True)
class SkillRevision:
    """Snapshot of a skill before an auto-repair or manual edit."""

    id: str
    skill_id: str
    version: int
    snapshot_json: str
    reason: str
    created_at: float


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
    update_skill_id: str | None = None


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
    llm_model: str | None = None
    llm_base_url: str | None = None

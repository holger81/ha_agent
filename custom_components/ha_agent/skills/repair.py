"""Detect and apply deterministic skill repairs from turn traces."""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import LOGGER
from .body import normalize_skill
from .defaults import apply_route_defaults, default_slots_for_route
from .files import mirror_skill_to_file
from .models import Skill, SkillSlot, TurnTrace
from .observer import is_discovery_tool
from .store import get_skill_store

_IMAP_TOOL = re.compile(r"imap|mail_mcp", re.IGNORECASE)
_REPAIR_COOLDOWN_SECONDS = 300
_last_repair_at: dict[str, float] = {}


@dataclass(slots=True)
class RepairIssue:
    """One repairable problem detected in a skill turn."""

    kind: str
    detail: str
    fields: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Outcome of an auto-repair."""

    skill: Skill
    reason: str
    from_version: int
    revision_id: str


def detect_repairable_issues(trace: TurnTrace, skill: Skill) -> list[RepairIssue]:
    """Return issues that can be repaired from a turn trace."""
    if trace.skill_plan_override:
        return []

    issues: list[RepairIssue] = []
    missing: set[str] = set()
    for call in trace.tool_calls:
        for field_name in call.get("missing_fields") or []:
            missing.add(str(field_name))
        if call.get("error_kind") == "param" and call.get("error"):
            match = re.search(r"missing field ['\"]?(\w+)", str(call["error"]), re.I)
            if match:
                missing.add(match.group(1))

    if missing:
        issues.append(
            RepairIssue(
                kind="missing_param",
                detail=f"Required parameters missing: {', '.join(sorted(missing))}",
                fields=sorted(missing),
            )
        )

    if trace.skill_followed is False:
        issues.append(
            RepairIssue(
                kind="not_followed",
                detail=(
                    "Skill workflow was not followed or verifier rejected adherence."
                ),
            )
        )

    for call in trace.tool_calls:
        err = str(call.get("error") or "").lower()
        if "unknown tool" in err or "not found" in err:
            issues.append(
                RepairIssue(
                    kind="bad_tool_name",
                    detail="Tool call failed because the tool name was wrong.",
                )
            )
            break

    concrete_steps = [
        step
        for step in skill.tool_steps
        if str(step.get("toolName") or step.get("name") or "").strip()
        and not is_discovery_tool(str(step.get("toolName") or step.get("name") or ""))
    ]
    if len(concrete_steps) >= 2:
        for call in trace.tool_calls:
            name = str(call.get("toolName") or call.get("name") or "")
            if is_discovery_tool(name):
                issues.append(
                    RepairIssue(
                        kind="discovery_instead_of_skill",
                        detail="Discovery tools ran despite concrete skill steps.",
                    )
                )
                break

    return issues


def _ensure_mailbox_on_imap_steps(skill: Skill) -> tuple[Skill, str | None]:
    """Add mailbox slot and {{mailbox}} to IMAP tool steps."""
    skill = copy.deepcopy(skill)
    apply_route_defaults(skill)
    has_mailbox_slot = any(s.name == "mailbox" for s in skill.slots)
    if not has_mailbox_slot:
        skill.slots.append(
            SkillSlot(
                name="mailbox",
                description="IMAP mailbox folder",
                source="default",
                default="INBOX",
            )
        )

    changed = False
    new_steps: list[dict[str, Any]] = []
    for step in skill.tool_steps:
        step_copy = dict(step)
        tool_name = str(step_copy.get("toolName") or step_copy.get("name") or "")
        if _IMAP_TOOL.search(tool_name):
            args = step_copy.get("arguments")
            if not isinstance(args, dict):
                args = {}
            if "mailbox" not in args:
                args = {**args, "mailbox": "{{mailbox}}"}
                step_copy["arguments"] = args
                changed = True
        new_steps.append(step_copy)
    skill.tool_steps = new_steps

    if "mailbox" not in skill.body.lower() and changed:
        skill.body = (
            skill.body.rstrip()
            + "\n\nAlways pass `mailbox` (default INBOX) to IMAP tools."
        )
        changed = True

    if not changed and not has_mailbox_slot:
        return skill, None
    return skill, "added mailbox parameter to IMAP tool steps"


def _merge_successful_retry_args(
    skill: Skill, trace: TurnTrace
) -> tuple[Skill, str | None]:
    """Promote arguments from successful retries into tool_steps templates."""
    skill = copy.deepcopy(skill)
    by_tool: dict[str, dict[str, Any]] = {}
    for call in trace.tool_calls:
        if not call.get("succeeded"):
            continue
        name = str(call.get("toolName") or call.get("name") or "")
        args = call.get("arguments")
        if name and isinstance(args, dict) and args:
            by_tool[name] = args

    if not by_tool:
        return skill, None

    changed = False
    new_steps: list[dict[str, Any]] = []
    for step in skill.tool_steps:
        step_copy = dict(step)
        tool_name = str(step_copy.get("toolName") or step_copy.get("name") or "")
        if tool_name in by_tool:
            retry_args = by_tool[tool_name]
            existing = step_copy.get("arguments")
            if not isinstance(existing, dict):
                existing = {}
            merged = {**existing}
            for key, value in retry_args.items():
                if key not in merged or merged[key] in ("", "{{" + key + "}}"):
                    merged[key] = value
                    changed = True
            step_copy["arguments"] = merged
        new_steps.append(step_copy)
    skill.tool_steps = new_steps
    if not changed:
        return skill, None
    return skill, "merged successful retry arguments into tool steps"


def _strip_discovery_from_steps(skill: Skill) -> tuple[Skill, str | None]:
    """Remove discovery tools from skill tool_steps."""
    skill = copy.deepcopy(skill)
    filtered = [
        step
        for step in skill.tool_steps
        if not is_discovery_tool(str(step.get("toolName") or step.get("name") or ""))
    ]
    if len(filtered) == len(skill.tool_steps):
        return skill, None
    skill.tool_steps = filtered
    return skill, "removed discovery tools from skill steps"


def _canonicalize_tool_names(skill: Skill) -> tuple[Skill, str | None]:
    """Rewrite shorthand or mistyped MCP tool names in body and steps."""
    before = (
        skill.body,
        tuple(str(step.get("toolName") or "") for step in skill.tool_steps),
    )
    normalize_skill(skill)
    after = (
        skill.body,
        tuple(str(step.get("toolName") or "") for step in skill.tool_steps),
    )
    if before == after:
        return skill, None
    return skill, "canonicalized MCP tool names in skill workflow"


def repair_skill_from_trace(skill: Skill, trace: TurnTrace) -> tuple[Skill, str] | None:
    """Apply deterministic repairs; return updated skill and reason."""
    issues = detect_repairable_issues(trace, skill)
    if not issues:
        return None

    working = copy.deepcopy(skill)
    reasons: list[str] = []
    kinds = {issue.kind for issue in issues}

    if "missing_param" in kinds or any(
        "mailbox" in issue.fields for issue in issues if issue.fields
    ):
        working, reason = _ensure_mailbox_on_imap_steps(working)
        if reason:
            reasons.append(reason)

    if "bad_tool_name" in kinds:
        working, reason = _canonicalize_tool_names(working)
        if reason:
            reasons.append(reason)

    working, reason = _merge_successful_retry_args(working, trace)
    if reason:
        reasons.append(reason)

    if "discovery_instead_of_skill" in kinds:
        working, reason = _strip_discovery_from_steps(working)
        if reason:
            reasons.append(reason)

    route = trace.route or working.route_scope or ""
    if not working.slots and route:
        defaults = default_slots_for_route(route)
        if defaults:
            working.slots = list(defaults)
            reasons.append(f"added default slots for {route} route")

    if not reasons:
        return None
    return working, "; ".join(reasons)


def can_auto_repair(skill_id: str) -> bool:
    """Rate-limit auto-repair to once per skill every few minutes."""
    last = _last_repair_at.get(skill_id, 0.0)
    return (time.time() - last) >= _REPAIR_COOLDOWN_SECONDS


def auto_repair_skill(
    hass: HomeAssistant,
    entry_id: str,
    skill: Skill,
    trace: TurnTrace,
) -> RepairResult | None:
    """Save revision, apply repair, persist skill. Runs in executor context."""
    if skill.is_builtin:
        return None
    if not can_auto_repair(skill.id):
        return None

    patched = repair_skill_from_trace(skill, trace)
    if patched is None:
        return None

    updated, reason = patched
    store = get_skill_store(hass, entry_id)
    from_version = skill.version
    revision_id = store.save_revision(skill, reason=reason)
    updated.version = skill.version + 1
    updated.last_improved_at = time.time()
    saved = store.update_skill(updated)
    mirror_skill_to_file(hass, entry_id, saved)
    _last_repair_at[skill.id] = time.time()
    LOGGER.info(
        "Auto-repaired skill %s v%s→v%s: %s",
        saved.title,
        from_version,
        saved.version,
        reason,
    )
    return RepairResult(
        skill=saved,
        reason=reason,
        from_version=from_version,
        revision_id=revision_id,
    )

"""Propose and apply skill generalization (merge similar learned skills)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .models import Skill, SkillDraft, SkillSlot
from .observer import is_discovery_tool
from .tool_names import canonicalize_tool_name

_ENTITY_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$", re.IGNORECASE)
_MIN_CLUSTER = 2
_NEAR_TOOL_JACCARD = 0.8


@dataclass(slots=True)
class SkillGeneralizeCluster:
    """One merge candidate: similar skills that can collapse into a parent."""

    key: str
    route_scope: str | None
    tool_names: list[str]
    skill_ids: list[str]
    survivor_id: str
    draft: SkillDraft
    reason: str
    member_titles: list[str] = field(default_factory=list)


def _tool_name(raw: Any) -> str:
    name = str(raw or "").strip()
    if not name:
        return ""
    return canonicalize_tool_name(name)


def skill_tool_fingerprint(skill: Skill) -> frozenset[str]:
    """Non-discovery tool names for clustering."""
    names: set[str] = set()
    for step in skill.tool_steps or []:
        if not isinstance(step, dict):
            continue
        name = _tool_name(step.get("toolName") or step.get("name"))
        if name and not is_discovery_tool(name):
            names.add(name.lower())
    return frozenset(names)


def _scope_key(skill: Skill) -> str:
    return (skill.route_scope or "").strip().lower() or "_"


def _pick_survivor(skills: list[Skill]) -> Skill:
    return max(
        skills,
        key=lambda skill: (
            int(skill.use_count or 0),
            float(skill.score or 0.0),
            float(skill.created_at or 0.0),
        ),
    )


def _merge_slots(skills: list[Skill]) -> list[SkillSlot]:
    by_name: dict[str, SkillSlot] = {}
    for skill in skills:
        for slot in skill.slots or []:
            if slot.name not in by_name:
                by_name[slot.name] = SkillSlot(
                    name=slot.name,
                    description=slot.description,
                    source=slot.source,
                    default=None if slot.name == "entity_id" else slot.default,
                )
    return list(by_name.values())


def _union_tool_steps(skills: list[Skill]) -> list[dict[str, Any]]:
    """Keep first occurrence of each tool; prefer slotted / sparse arguments."""
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for skill in sorted(
        skills,
        key=lambda item: (-int(item.use_count or 0), item.slug),
    ):
        for step in skill.tool_steps or []:
            if not isinstance(step, dict):
                continue
            name = _tool_name(step.get("toolName") or step.get("name"))
            if not name or is_discovery_tool(name) or name.lower() in seen:
                continue
            seen.add(name.lower())
            step_copy = dict(step)
            step_copy["toolName"] = name
            args = step_copy.get("arguments")
            if isinstance(args, dict):
                cleaned: dict[str, Any] = {}
                for key, value in args.items():
                    if key == "entity_id":
                        cleaned[key] = "{{entity_id}}"
                    elif key == "mailbox" and isinstance(value, str) and value.strip():
                        cleaned[key] = "{{mailbox}}"
                    elif isinstance(value, str) and value.startswith("{{"):
                        cleaned[key] = value
                    elif isinstance(value, str) and _ENTITY_ID.match(value):
                        continue
                    else:
                        cleaned[key] = value
                step_copy["arguments"] = cleaned
            ordered.append(step_copy)
    return ordered


def _merge_triggers(skills: list[Skill], *, limit: int = 12) -> list[str]:
    triggers: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        for trigger in skill.triggers or []:
            key = trigger.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            triggers.append(trigger.strip())
            if len(triggers) >= limit:
                return triggers
    return triggers


def _general_title(skills: list[Skill], scope: str, tools: list[str]) -> str:
    if scope and scope != "_":
        label = scope.replace("_", " ").title()
        return f"Generalized {label} workflow"
    if tools:
        short = tools[0].split("__")[-1].replace("_", " ")
        return f"Generalized {short} skill"
    titles = [skill.title for skill in skills if skill.title]
    if titles:
        return f"Generalized: {titles[0]}"
    return "Generalized skill"


def _ensure_parameter_slots(draft: SkillDraft) -> SkillDraft:
    """Add entity_id / mailbox slots when tool steps reference them."""
    slots = list(draft.slots)
    names = {slot.name for slot in slots}
    needs_entity = False
    needs_mailbox = False
    for step in draft.tool_steps:
        if not isinstance(step, dict):
            continue
        args = step.get("arguments")
        if not isinstance(args, dict):
            continue
        if "entity_id" in args:
            needs_entity = True
        if "mailbox" in args:
            needs_mailbox = True
    if needs_entity and "entity_id" not in names:
        slots.append(
            SkillSlot(
                name="entity_id",
                description="Home Assistant entity to control",
                source="user",
                default=None,
            )
        )
    if needs_mailbox and "mailbox" not in names:
        slots.append(
            SkillSlot(
                name="mailbox",
                description="IMAP mailbox folder",
                source="default",
                default="INBOX",
            )
        )
    draft.slots = slots
    return draft


def build_generalized_draft(skills: list[Skill]) -> SkillDraft:
    """Build a slotted parent draft from a cluster of similar skills."""
    if not skills:
        raise ValueError("Need at least one skill to generalize")
    survivor = _pick_survivor(skills)
    scope = (survivor.route_scope or "").strip() or None
    tools = sorted(skill_tool_fingerprint(survivor))
    titles = [skill.title for skill in skills]
    body_lines = [
        f"# {_general_title(skills, scope or '_', tools)}",
        "",
        "Merged from similar learned skills:",
        *[f"- {title}" for title in titles],
        "",
        "Use slots for values that vary per request "
        "(entity, mailbox, etc.). Follow the tool sequence below.",
        "",
    ]
    survivor_body = (survivor.body or "").strip()
    if survivor_body:
        body_lines.extend(["## Primary workflow", "", survivor_body, ""])
    for skill in skills:
        if skill.id == survivor.id:
            continue
        extra = (skill.body or "").strip()
        if extra and extra not in survivor_body:
            body_lines.extend(
                [f"## From {skill.title}", "", extra[:1200], ""],
            )

    draft = SkillDraft(
        title=_general_title(skills, scope or "_", tools)[:64],
        description=(
            f"Generalized workflow merged from {len(skills)} similar skills. "
            "Parameters vary via slots."
        )[:512],
        triggers=_merge_triggers(skills),
        body="\n".join(body_lines).strip(),
        tool_steps=_union_tool_steps(skills),
        slots=_merge_slots(skills),
        preconditions=survivor.preconditions,
        parent_id=None,
        route_scope=scope,
        llm_model=next(
            (skill.llm_model for skill in skills if (skill.llm_model or "").strip()),
            None,
        ),
        llm_base_url=next(
            (
                skill.llm_base_url
                for skill in skills
                if (skill.llm_base_url or "").strip()
            ),
            None,
        ),
    )
    return _ensure_parameter_slots(draft)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cluster_skills_for_generalize(
    skills: list[Skill],
    *,
    min_cluster: int = _MIN_CLUSTER,
) -> list[SkillGeneralizeCluster]:
    """Group enabled learned skills that share scope and tool fingerprints."""
    candidates = [
        skill
        for skill in skills
        if not skill.is_builtin and skill.enabled and skill_tool_fingerprint(skill)
    ]
    buckets: dict[tuple[str, frozenset[str]], list[Skill]] = defaultdict(list)
    for skill in candidates:
        buckets[(_scope_key(skill), skill_tool_fingerprint(skill))].append(skill)

    clusters: list[SkillGeneralizeCluster] = []
    used_ids: set[str] = set()

    for (scope, fingerprint), members in sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]), item[0][0]),
    ):
        if len(members) < min_cluster:
            continue
        member_ids = {skill.id for skill in members}
        if member_ids & used_ids:
            continue
        survivor = _pick_survivor(members)
        tools = sorted(fingerprint)
        key = f"{scope}|{'+'.join(tools)}"
        draft = build_generalized_draft(members)
        clusters.append(
            SkillGeneralizeCluster(
                key=key,
                route_scope=None if scope == "_" else scope,
                tool_names=tools,
                skill_ids=[skill.id for skill in members],
                survivor_id=survivor.id,
                draft=draft,
                reason=(
                    f"{len(members)} skills share route_scope={scope!r} and "
                    f"tools [{', '.join(tools)}]."
                ),
                member_titles=[skill.title for skill in members],
            )
        )
        used_ids |= member_ids

    remaining = [skill for skill in candidates if skill.id not in used_ids]
    by_scope: dict[str, list[Skill]] = defaultdict(list)
    for skill in remaining:
        by_scope[_scope_key(skill)].append(skill)

    for scope, group in by_scope.items():
        if len(group) < min_cluster:
            continue
        pending = list(group)
        while len(pending) >= min_cluster:
            seed = pending.pop(0)
            if seed.id in used_ids:
                continue
            seed_fp = skill_tool_fingerprint(seed)
            cohort = [seed]
            still: list[Skill] = []
            for other in pending:
                if other.id in used_ids:
                    continue
                if (
                    _jaccard(seed_fp, skill_tool_fingerprint(other))
                    >= _NEAR_TOOL_JACCARD
                ):
                    cohort.append(other)
                else:
                    still.append(other)
            pending = still
            if len(cohort) < min_cluster:
                continue
            survivor = _pick_survivor(cohort)
            tools = sorted(skill_tool_fingerprint(survivor))
            key = f"{scope}|near|{'+'.join(tools)}|{survivor.id[:8]}"
            draft = build_generalized_draft(cohort)
            clusters.append(
                SkillGeneralizeCluster(
                    key=key,
                    route_scope=None if scope == "_" else scope,
                    tool_names=tools,
                    skill_ids=[skill.id for skill in cohort],
                    survivor_id=survivor.id,
                    draft=draft,
                    reason=(
                        f"{len(cohort)} skills share route_scope={scope!r} with "
                        f"overlapping tools (Jaccard ≥ {_NEAR_TOOL_JACCARD})."
                    ),
                    member_titles=[skill.title for skill in cohort],
                )
            )
            used_ids |= {skill.id for skill in cohort}

    return clusters


def cluster_to_dict(cluster: SkillGeneralizeCluster) -> dict[str, Any]:
    """Serialize a propose payload for the console."""
    return {
        "key": cluster.key,
        "route_scope": cluster.route_scope,
        "tool_names": list(cluster.tool_names),
        "skill_ids": list(cluster.skill_ids),
        "survivor_id": cluster.survivor_id,
        "member_titles": list(cluster.member_titles),
        "reason": cluster.reason,
        "draft": {
            "title": cluster.draft.title,
            "description": cluster.draft.description,
            "triggers": list(cluster.draft.triggers),
            "body": cluster.draft.body,
            "tool_steps": list(cluster.draft.tool_steps),
            "slots": [
                {
                    "name": slot.name,
                    "description": slot.description,
                    "source": slot.source,
                    "default": slot.default,
                }
                for slot in cluster.draft.slots
            ],
            "route_scope": cluster.draft.route_scope,
            "llm_model": cluster.draft.llm_model,
            "llm_base_url": cluster.draft.llm_base_url,
        },
    }

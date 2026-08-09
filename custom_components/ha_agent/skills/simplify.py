"""LLM-assisted skill simplification and combination (propose → apply → undo)."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend, get_llm_backend
from ..const import DATA_KEY, LOGGER
from ..llm_client import LlmClient
from .body import normalize_skill_draft
from .models import Skill, SkillDraft, SkillSlot

SIMPLIFY_STATE_KEY = "skill_simplify_state"
_MAX_CATALOG = 30
_BODY_LIMIT = 900
_MAX_PROPOSALS = 8

_SIMPLIFY_PROMPT = (
    "You are simplifying a Home Assistant agent skill library.\n"
    "Given LEARNED SKILLS, propose consolidations that reduce redundancy "
    "and shorten overly specific or verbose skills.\n"
    "Return ONLY valid JSON:\n"
    "{\n"
    '  "summary": "short overview",\n'
    '  "proposals": [\n'
    "    {\n"
    '      "action": "combine" | "simplify",\n'
    '      "skill_ids": ["id", ...],\n'
    '      "survivor_id": "id to update",\n'
    '      "reason": "why this helps",\n'
    '      "draft": {\n'
    '        "title": "...",\n'
    '        "description": "...",\n'
    '        "triggers": ["..."],\n'
    '        "body": "markdown workflow",\n'
    '        "tool_steps": [{"toolName": "...", "arguments": {}}],\n'
    '        "slots": [{"name": "...", "description": "", "source": "user", '
    '"default": null}],\n'
    '        "route_scope": "action|chat|email|news|... or null",\n'
    '        "preconditions": ""\n'
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Prefer fewer, clearer skills with slots for varying values "
    "(entity_id, mailbox, etc.).\n"
    "- action=combine when 2+ skills should merge; include all member skill_ids.\n"
    "- action=simplify when one skill should be rewritten (skill_ids length 1).\n"
    "- survivor_id must be one of skill_ids (prefer highest use_count).\n"
    "- Keep working tool_steps; drop discovery-only tools; use "
    "{{slot}} placeholders for varying args.\n"
    "- Do not invent tools that none of the members used.\n"
    "- Return proposals: [] when nothing useful can be improved.\n"
    "- At most 8 proposals. Be conservative: only clear wins."
)


@dataclass(slots=True)
class SkillSimplifyProposal:
    """One preview-only simplify/combine proposal."""

    proposal_id: str
    action: str
    skill_ids: list[str]
    survivor_id: str
    member_titles: list[str]
    reason: str
    draft: SkillDraft
    model_used: str = ""


@dataclass(slots=True)
class SkillSimplifyUndo:
    """Snapshot needed to reverse the last applied simplification."""

    survivor_id: str
    revision_id: str
    archived: list[dict[str, Any]] = field(default_factory=list)
    proposal_id: str = ""
    summary: str = ""
    created_at: float = 0.0


@dataclass(slots=True)
class SkillSimplifyState:
    """Per-entry propose/apply/undo state."""

    proposals: list[SkillSimplifyProposal] = field(default_factory=list)
    summary: str = ""
    model_used: str = ""
    undo: SkillSimplifyUndo | None = None


def _state_map(hass: HomeAssistant) -> dict[str, SkillSimplifyState]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(SIMPLIFY_STATE_KEY, {})


def get_simplify_state(hass: HomeAssistant, entry_id: str) -> SkillSimplifyState:
    states = _state_map(hass)
    if entry_id not in states:
        states[entry_id] = SkillSimplifyState()
    return states[entry_id]


def clear_simplify_proposals(hass: HomeAssistant, entry_id: str) -> None:
    state = get_simplify_state(hass, entry_id)
    state.proposals = []
    state.summary = ""


def _strip_json_fence(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def skill_to_catalog_entry(skill: Skill) -> dict[str, Any]:
    """Compact skill payload for the simplify prompt."""
    body = (skill.body or "").strip()
    if len(body) > _BODY_LIMIT:
        body = body[:_BODY_LIMIT] + "…"
    return {
        "id": skill.id,
        "slug": skill.slug,
        "title": skill.title,
        "description": skill.description,
        "triggers": list(skill.triggers or [])[:12],
        "body": body,
        "tool_steps": list(skill.tool_steps or [])[:12],
        "slots": [
            {
                "name": slot.name,
                "description": slot.description,
                "source": slot.source,
                "default": slot.default,
            }
            for slot in (skill.slots or [])
        ],
        "route_scope": skill.route_scope,
        "use_count": int(skill.use_count or 0),
        "score": float(skill.score or 0.0),
    }


def _parse_draft(data: dict[str, Any], *, fallback: Skill) -> SkillDraft | None:
    title = str(data.get("title") or fallback.title).strip()
    description = str(data.get("description") or fallback.description).strip()
    body = str(data.get("body") or fallback.body).strip()
    if not title or not body:
        return None
    triggers_raw = data.get("triggers", fallback.triggers)
    triggers = (
        [str(item).strip() for item in triggers_raw if str(item).strip()]
        if isinstance(triggers_raw, list)
        else list(fallback.triggers or [])
    )
    if not triggers:
        triggers = [title]
    tool_steps_raw = data.get("tool_steps", fallback.tool_steps)
    tool_steps = (
        [
            item
            for item in tool_steps_raw
            if isinstance(item, dict)
            and str(item.get("toolName") or item.get("name") or "").strip()
        ]
        if isinstance(tool_steps_raw, list)
        else list(fallback.tool_steps or [])
    )
    slots: list[SkillSlot] = []
    slots_raw = data.get("slots")
    if isinstance(slots_raw, list):
        for item in slots_raw:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            slots.append(
                SkillSlot(
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                    source=str(item.get("source", "user")),
                    default=item.get("default"),
                )
            )
    elif fallback.slots:
        slots = list(fallback.slots)
    route_scope = data.get("route_scope", fallback.route_scope)
    if isinstance(route_scope, str):
        route_scope = route_scope.strip() or None
    else:
        route_scope = fallback.route_scope
    return normalize_skill_draft(
        SkillDraft(
            title=title[:64],
            description=(description or title)[:512],
            triggers=triggers[:16],
            body=body,
            tool_steps=tool_steps,
            slots=slots,
            preconditions=str(
                data.get("preconditions", fallback.preconditions) or ""
            ),
            parent_id=None,
            route_scope=route_scope,
            llm_model=fallback.llm_model,
            llm_base_url=fallback.llm_base_url,
        ),
        explicit_tool_steps=bool(tool_steps),
    )


def parse_simplify_response(
    content: str,
    *,
    skills_by_id: dict[str, Skill],
    model_used: str = "",
) -> tuple[str, list[SkillSimplifyProposal]]:
    """Parse LLM JSON into validated proposals."""
    try:
        data = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        return "", []
    if not isinstance(data, dict):
        return "", []
    summary = str(data.get("summary") or "").strip()
    raw_proposals = data.get("proposals")
    if not isinstance(raw_proposals, list):
        return summary, []

    proposals: list[SkillSimplifyProposal] = []
    for item in raw_proposals:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        if action not in {"combine", "simplify"}:
            continue
        ids_raw = item.get("skill_ids")
        if not isinstance(ids_raw, list):
            continue
        skill_ids = [str(sid).strip() for sid in ids_raw if str(sid).strip()]
        skill_ids = [sid for sid in skill_ids if sid in skills_by_id]
        if not skill_ids:
            continue
        if action == "combine" and len(skill_ids) < 2:
            continue
        if action == "simplify":
            skill_ids = skill_ids[:1]
        survivor_id = str(item.get("survivor_id") or skill_ids[0]).strip()
        if survivor_id not in skill_ids:
            survivor_id = skill_ids[0]
        draft_raw = item.get("draft")
        if not isinstance(draft_raw, dict):
            continue
        draft = _parse_draft(draft_raw, fallback=skills_by_id[survivor_id])
        if draft is None:
            continue
        proposals.append(
            SkillSimplifyProposal(
                proposal_id=str(uuid.uuid4()),
                action=action,
                skill_ids=skill_ids,
                survivor_id=survivor_id,
                member_titles=[skills_by_id[sid].title for sid in skill_ids],
                reason=str(item.get("reason") or "").strip()
                or f"{action.title()} proposed by model.",
                draft=draft,
                model_used=model_used,
            )
        )
        if len(proposals) >= _MAX_PROPOSALS:
            break
    return summary, proposals


def proposal_to_dict(proposal: SkillSimplifyProposal) -> dict[str, Any]:
    """Serialize a proposal for the console."""
    draft = proposal.draft
    return {
        "proposal_id": proposal.proposal_id,
        "action": proposal.action,
        "skill_ids": list(proposal.skill_ids),
        "survivor_id": proposal.survivor_id,
        "member_titles": list(proposal.member_titles),
        "reason": proposal.reason,
        "model_used": proposal.model_used,
        "draft": {
            "title": draft.title,
            "description": draft.description,
            "triggers": list(draft.triggers),
            "body": draft.body,
            "tool_steps": list(draft.tool_steps),
            "slots": [
                {
                    "name": slot.name,
                    "description": slot.description,
                    "source": slot.source,
                    "default": slot.default,
                }
                for slot in draft.slots
            ],
            "route_scope": draft.route_scope,
            "preconditions": draft.preconditions,
            "llm_model": draft.llm_model,
            "llm_base_url": draft.llm_base_url,
        },
    }


def undo_to_dict(undo: SkillSimplifyUndo | None) -> dict[str, Any] | None:
    if undo is None:
        return None
    return {
        "survivor_id": undo.survivor_id,
        "revision_id": undo.revision_id,
        "archived_skill_ids": [item["id"] for item in undo.archived],
        "proposal_id": undo.proposal_id,
        "summary": undo.summary,
        "created_at": undo.created_at,
    }


async def resolve_strong_simplify_backend(
    hass: HomeAssistant,
    entry_id: str,
) -> tuple[LlmBackend, str]:
    """Prefer the top-ranked eval model, else the configured chat model."""
    from ..api.helpers import get_entry

    entry = get_entry(hass, entry_id)
    base = get_llm_backend(entry)
    backend = replace(
        base,
        max_tokens=max(int(base.max_tokens), 4096),
        temperature=min(float(base.temperature), 0.2),
    )
    label = f"chat model ({backend.model})"
    try:
        from ..eval.store import get_eval_store

        store = get_eval_store(hass, entry_id)
        scores = await hass.async_add_executor_job(store.list_model_scores)
        if scores:
            top = str(scores[0].get("model_id") or "").strip()
            if top:
                backend = replace(backend, model=top)
                label = f"top eval model ({top})"
    except Exception as err:
        LOGGER.debug("Simplify strong-model lookup skipped: %s", err)
    return backend, label


async def propose_skill_simplification(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    skills: list[Skill],
    model_label: str = "",
) -> dict[str, Any]:
    """Ask the model for simplify/combine proposals and stash them for preview."""
    candidates = [
        skill
        for skill in skills
        if not skill.is_builtin and skill.enabled
    ][:_MAX_CATALOG]
    if not candidates:
        state = get_simplify_state(hass, entry_id)
        state.proposals = []
        state.summary = "No enabled learned skills to simplify."
        state.model_used = model_label or backend.model
        return {
            "proposals": [],
            "count": 0,
            "summary": state.summary,
            "model_used": state.model_used,
            "undo": undo_to_dict(state.undo),
        }

    catalog = [skill_to_catalog_entry(skill) for skill in candidates]
    messages = [
        {"role": "system", "content": _SIMPLIFY_PROMPT},
        {
            "role": "user",
            "content": json.dumps({"skills": catalog}, ensure_ascii=True),
        },
    ]
    model_used = model_label or backend.model
    try:
        result = await llm.chat(messages, backend, tools=[])
        content = result.content or ""
    except Exception as err:
        LOGGER.warning("Skill simplify propose failed: %s", err)
        raise

    by_id = {skill.id: skill for skill in candidates}
    summary, proposals = parse_simplify_response(
        content,
        skills_by_id=by_id,
        model_used=model_used,
    )
    state = get_simplify_state(hass, entry_id)
    state.proposals = proposals
    state.summary = summary or (
        f"Found {len(proposals)} proposal(s)."
        if proposals
        else "Model found no useful simplifications."
    )
    state.model_used = model_used
    return {
        "proposals": [proposal_to_dict(item) for item in proposals],
        "count": len(proposals),
        "summary": state.summary,
        "model_used": model_used,
        "undo": undo_to_dict(state.undo),
    }

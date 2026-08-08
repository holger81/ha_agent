"""Distill learned skills from successful agent turns."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..llm_client import LlmClient
from .body import normalize_skill_draft
from .files import async_mirror_skill_to_file
from .learning_policy import prepare_learned_draft
from .models import Skill, SkillDraft, TurnTrace
from .observer import observe_skill_candidate
from .store import get_skill_store


async def save_skill_from_draft(
    hass: HomeAssistant,
    entry_id: str,
    draft: SkillDraft,
    *,
    update_existing: Skill | None = None,
    revision_reason: str = "Skill learning update",
) -> Skill:
    """Persist a distilled skill, optionally updating a duplicate."""
    store = get_skill_store(hass, entry_id)

    def _save() -> Skill:
        if update_existing is not None:
            store.save_revision(update_existing, reason=revision_reason)
            skill = update_existing
            skill.title = draft.title
            skill.description = draft.description
            skill.triggers = draft.triggers
            skill.body = draft.body
            skill.tool_steps = draft.tool_steps
            skill.slots = draft.slots
            skill.preconditions = draft.preconditions
            skill.parent_id = draft.parent_id
            skill.route_scope = draft.route_scope
            skill.version += 1
            return store.update_skill(skill)
        return store.insert_skill(
            title=draft.title,
            description=draft.description,
            triggers=draft.triggers,
            body=draft.body,
            tool_steps=draft.tool_steps,
            slots=draft.slots,
            preconditions=draft.preconditions,
            parent_id=draft.parent_id,
            route_scope=draft.route_scope,
        )

    skill = await hass.async_add_executor_job(_save)
    await async_mirror_skill_to_file(hass, entry_id, skill)
    return skill


async def persist_skill_draft(
    hass: HomeAssistant,
    entry_id: str,
    draft: SkillDraft,
    *,
    trace: TurnTrace | None = None,
    update_skill_id: str | None = None,
    revision_reason: str = "Skill learning update",
    apply_quality_gate: bool = True,
) -> Skill | None:
    """Persist a draft with shared confirm/auto-save rules.

    - If ``update_skill_id`` is set, update that skill only (no FTS dedupe).
    - Else if a strong ``find_duplicate`` hit exists, update that skill.
    - Else insert a new skill.
    """
    working = draft
    if apply_quality_gate:
        if trace is None:
            return None
        prepared = prepare_learned_draft(working, trace)
        if prepared is None:
            return None
        working = prepared

    normalized = normalize_skill_draft(
        working,
        explicit_tool_steps=bool(working.tool_steps),
    )
    store = get_skill_store(hass, entry_id)

    update_existing: Skill | None = None
    if update_skill_id:

        def _load_target() -> Skill | None:
            return store.get_skill(update_skill_id)

        update_existing = await hass.async_add_executor_job(_load_target)
    else:

        def _find_dup() -> Skill | None:
            return store.find_duplicate(
                normalized.triggers,
                tool_steps=normalized.tool_steps,
                route_scope=normalized.route_scope,
            )

        update_existing = await hass.async_add_executor_job(_find_dup)

    return await save_skill_from_draft(
        hass,
        entry_id,
        normalized,
        update_existing=update_existing,
        revision_reason=revision_reason,
    )


async def create_skill_from_trace(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    trace: TurnTrace,
    history: list[dict[str, str]],
    manual_save: bool = False,
    draft: SkillDraft | None = None,
    update_skill_id: str | None = None,
) -> Skill | None:
    """Observe, distill, and save a skill from a turn trace."""
    if draft is None:
        observed = await observe_skill_candidate(
            llm,
            backend,
            trace=trace,
            history=history,
            manual_save=manual_save,
        )
        if not observed.learn or observed.draft is None:
            return None
        draft = observed.draft

    return await persist_skill_draft(
        hass,
        entry_id,
        draft,
        trace=trace,
        update_skill_id=update_skill_id,
    )

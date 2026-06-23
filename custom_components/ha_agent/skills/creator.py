"""Distill learned skills from successful agent turns."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..llm_client import LlmClient
from .body import normalize_skill_draft
from .models import Skill, SkillDraft, TurnTrace
from .observer import observe_skill_candidate
from .store import get_skill_store


async def save_skill_from_draft(
    hass: HomeAssistant,
    entry_id: str,
    draft: SkillDraft,
    *,
    update_existing: Skill | None = None,
) -> Skill:
    """Persist a distilled skill, optionally updating a duplicate."""
    store = get_skill_store(hass, entry_id)

    def _save() -> Skill:
        if update_existing is not None:
            skill = update_existing
            skill.title = draft.title
            skill.description = draft.description
            skill.triggers = draft.triggers
            skill.body = draft.body
            skill.tool_steps = draft.tool_steps
            skill.version += 1
            return store.update_skill(skill)
        return store.insert_skill(
            title=draft.title,
            description=draft.description,
            triggers=draft.triggers,
            body=draft.body,
            tool_steps=draft.tool_steps,
        )

    return await hass.async_add_executor_job(_save)


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

    store = get_skill_store(hass, entry_id)

    def _find_dup() -> Skill | None:
        return store.find_duplicate(draft.triggers)

    duplicate = await hass.async_add_executor_job(_find_dup)
    normalized = normalize_skill_draft(
        draft,
        explicit_tool_steps=bool(draft.tool_steps),
    )
    return await save_skill_from_draft(
        hass,
        entry_id,
        normalized,
        update_existing=duplicate,
    )

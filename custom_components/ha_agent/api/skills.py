"""Skills API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..config_helpers import get_llm_backend
from ..llm_client import LlmClient
from ..skills.body import normalize_skill, normalize_skill_draft
from ..skills.creator import create_skill_from_trace, save_skill_from_draft
from ..skills.models import Skill, SkillDraft
from ..skills.runtime import get_pending_draft as runtime_get_pending_draft
from ..skills.runtime import pop_pending_draft
from ..skills.store import get_skill_store
from .helpers import get_entry
from .serialize import pending_draft_to_dict, skill_to_dict


async def list_skills(
    hass: HomeAssistant,
    entry_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return paginated skills for an entry."""
    store = get_skill_store(hass, entry_id)

    def _load() -> tuple[list[Skill], int]:
        skills = store.list_recent(limit=limit + offset)
        total = store.count_skills()
        return skills[offset : offset + limit], total

    skills, total = await hass.async_add_executor_job(_load)
    return {
        "skills": [skill_to_dict(skill) for skill in skills],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


async def search_skills(
    hass: HomeAssistant,
    entry_id: str,
    query: str,
    *,
    limit: int = 20,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    """FTS search returning full skill records."""
    store = get_skill_store(hass, entry_id)

    def _search() -> list[Skill]:
        rows = store.search(query, limit=limit, enabled_only=enabled_only)
        ids = [row.id for row in rows]
        skills = store.load_skills_by_ids(ids)
        by_id = {skill.id: skill for skill in skills}
        return [by_id[sid] for sid in ids if sid in by_id]

    skills = await hass.async_add_executor_job(_search)
    return [skill_to_dict(skill) for skill in skills]


async def get_skill(
    hass: HomeAssistant, entry_id: str, skill_id: str
) -> dict[str, Any]:
    """Return one skill by id."""
    store = get_skill_store(hass, entry_id)

    def _get() -> Skill | None:
        return store.get_skill(skill_id)

    skill = await hass.async_add_executor_job(_get)
    if skill is None:
        raise HomeAssistantError(f"Skill not found: {skill_id}")
    return skill_to_dict(skill)


async def set_skill_enabled(
    hass: HomeAssistant,
    entry_id: str,
    skill_id: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a skill."""
    store = get_skill_store(hass, entry_id)

    def _set() -> Skill | None:
        return store.set_enabled(skill_id, enabled)

    skill = await hass.async_add_executor_job(_set)
    if skill is None:
        raise HomeAssistantError(f"Skill not found: {skill_id}")
    return skill_to_dict(skill)


async def delete_skill(hass: HomeAssistant, entry_id: str, skill_id: str) -> bool:
    """Delete a skill."""
    store = get_skill_store(hass, entry_id)

    def _delete() -> bool:
        return store.delete_skill(skill_id)

    deleted = await hass.async_add_executor_job(_delete)
    if not deleted:
        raise HomeAssistantError(f"Skill not found: {skill_id}")
    return True


async def create_skill(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a skill from console form data."""
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    body = str(payload.get("body", "")).strip()
    triggers = payload.get("triggers", [])
    explicit_tool_steps = "tool_steps" in payload
    tool_steps = payload.get("tool_steps", []) if explicit_tool_steps else []
    if not title or not description or not body:
        raise HomeAssistantError("title, description, and body are required")
    if not isinstance(triggers, list) or not triggers:
        raise HomeAssistantError("At least one trigger is required")

    draft = normalize_skill_draft(
        SkillDraft(
            title=title,
            description=description,
            triggers=[str(t).strip() for t in triggers if str(t).strip()],
            body=body,
            tool_steps=[step for step in tool_steps if isinstance(step, dict)],
        ),
        explicit_tool_steps=explicit_tool_steps,
    )
    skill = await save_skill_from_draft(hass, entry_id, draft)
    return skill_to_dict(skill)


async def update_skill(
    hass: HomeAssistant,
    entry_id: str,
    skill_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update an existing skill."""
    store = get_skill_store(hass, entry_id)

    def _update() -> Skill:
        skill = store.get_skill(skill_id)
        if skill is None:
            raise HomeAssistantError(f"Skill not found: {skill_id}")
        if "title" in payload:
            skill.title = str(payload["title"]).strip()
        if "description" in payload:
            skill.description = str(payload["description"]).strip()
        if "body" in payload:
            skill.body = str(payload["body"]).strip()
        if "triggers" in payload:
            triggers = payload["triggers"]
            if not isinstance(triggers, list) or not triggers:
                raise HomeAssistantError("At least one trigger is required")
            skill.triggers = [str(t).strip() for t in triggers if str(t).strip()]
        explicit_tool_steps = "tool_steps" in payload
        if explicit_tool_steps:
            steps = payload["tool_steps"]
            skill.tool_steps = (
                [step for step in steps if isinstance(step, dict)]
                if isinstance(steps, list)
                else []
            )
        if "enabled" in payload:
            skill.enabled = bool(payload["enabled"])
        normalize_skill(skill, explicit_tool_steps=explicit_tool_steps)
        skill.version += 1
        return store.update_skill(skill)

    skill = await hass.async_add_executor_job(_update)
    return skill_to_dict(skill)


async def fetch_pending_draft(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> dict[str, Any] | None:
    """Return pending skill draft for a conversation."""
    draft = runtime_get_pending_draft(hass, conversation_id)
    if draft is None or draft.entry_id != entry_id:
        return None
    return pending_draft_to_dict(draft)


async def confirm_pending_draft(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    """Confirm and save a pending skill draft."""
    draft = runtime_get_pending_draft(hass, conversation_id)
    if draft is None or draft.entry_id != entry_id:
        raise HomeAssistantError("No pending skill draft for this conversation")
    session = async_get_clientsession(hass)
    llm = LlmClient(session)
    entry = get_entry(hass, entry_id)
    backend = get_llm_backend(entry)
    skill = await create_skill_from_trace(
        hass,
        entry_id,
        llm,
        backend,
        trace=draft.trace,
        history=draft.history,
        manual_save=True,
        draft=draft.skill_draft,
    )
    if skill is None:
        raise HomeAssistantError(
            "Failed to save skill from draft. Check HA Agent logs and LLM connectivity."
        )
    pop_pending_draft(hass, conversation_id)
    return skill_to_dict(skill)


def dismiss_pending_draft(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> None:
    """Dismiss a pending skill draft."""
    draft = runtime_get_pending_draft(hass, conversation_id)
    if draft is None or draft.entry_id != entry_id:
        raise HomeAssistantError("No pending skill draft for this conversation")
    pop_pending_draft(hass, conversation_id)


async def export_skills(hass: HomeAssistant, entry_id: str) -> list[dict[str, Any]]:
    """Export all skills as JSON-serializable dicts."""
    store = get_skill_store(hass, entry_id)

    def _export() -> list[Skill]:
        total = store.count_skills()
        return store.list_recent(limit=max(total, 1))

    skills = await hass.async_add_executor_job(_export)
    return [skill_to_dict(skill) for skill in skills]


async def import_skills(
    hass: HomeAssistant,
    entry_id: str,
    skills_payload: list[dict[str, Any]],
) -> int:
    """Import skills from a JSON bundle. Returns count imported."""
    count = 0
    for item in skills_payload:
        if not isinstance(item, dict):
            continue
        try:
            await create_skill(hass, entry_id, item)
            count += 1
        except HomeAssistantError:
            continue
    return count

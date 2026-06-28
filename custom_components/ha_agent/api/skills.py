"""Skills API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..config_helpers import get_llm_backend
from ..llm_client import LlmClient
from ..skills.body import (
    derive_tool_steps_from_body,
    normalize_skill,
    normalize_skill_draft,
)
from ..skills.creator import create_skill_from_trace, save_skill_from_draft
from ..skills.files import (
    async_mirror_skill_to_file,
    async_sync_skill_files,
    delete_skill_file,
    new_skill_markdown,
    skill_file_path,
    skills_directory,
)
from ..skills.markdown import (
    apply_draft_to_skill,
    draft_from_markdown,
    skill_to_markdown,
)
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
    payload = skill_to_dict(skill)
    directory = skills_directory(hass, entry_id)
    payload["markdown"] = skill_to_markdown(skill)
    payload["file_path"] = str(skill_file_path(directory, skill.slug))
    return payload


async def derive_skill_tool_steps(body: str) -> list[dict[str, Any]]:
    """Derive tool steps from a workflow body (console preview / recreate)."""
    return derive_tool_steps_from_body(str(body or ""))


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
    directory = skills_directory(hass, entry_id)

    def _delete() -> tuple[bool, str | None]:
        skill = store.get_skill(skill_id)
        if skill is None:
            return False, None
        slug = skill.slug
        deleted = store.delete_skill(skill_id)
        return deleted, slug if deleted else None

    deleted, slug = await hass.async_add_executor_job(_delete)
    if not deleted:
        raise HomeAssistantError(f"Skill not found: {skill_id}")
    if slug:
        await hass.async_add_executor_job(delete_skill_file, directory, slug)
    return True


async def create_skill(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a skill from markdown or legacy form fields."""
    if markdown := str(payload.get("markdown", "")).strip():
        draft, slug, explicit_tool_steps = draft_from_markdown(markdown)
        draft = normalize_skill_draft(draft, explicit_tool_steps=explicit_tool_steps)
        store = get_skill_store(hass, entry_id)

        def _insert() -> Skill:
            if slug and store.get_skill_by_slug(slug):
                raise HomeAssistantError(f"Skill already exists for slug: {slug}")
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
                slug=slug,
                enabled=bool(payload.get("enabled", True)),
            )

        skill = await hass.async_add_executor_job(_insert)
        await async_mirror_skill_to_file(hass, entry_id, skill)
        result = skill_to_dict(skill)
        result["markdown"] = skill_to_markdown(skill)
        result["file_path"] = str(
            skill_file_path(skills_directory(hass, entry_id), skill.slug)
        )
        return result

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
    result = skill_to_dict(skill)
    result["markdown"] = skill_to_markdown(skill)
    result["file_path"] = str(
        skill_file_path(skills_directory(hass, entry_id), skill.slug)
    )
    return result


async def update_skill(
    hass: HomeAssistant,
    entry_id: str,
    skill_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update an existing skill."""
    store = get_skill_store(hass, entry_id)
    directory = skills_directory(hass, entry_id)

    def _update() -> Skill:
        skill = store.get_skill(skill_id)
        if skill is None:
            raise HomeAssistantError(f"Skill not found: {skill_id}")
        if skill.is_builtin:
            raise HomeAssistantError("Built-in route skills cannot be edited")

        old_slug = skill.slug
        if markdown := str(payload.get("markdown", "")).strip():
            draft, slug, explicit_tool_steps = draft_from_markdown(
                markdown,
                filename_slug=skill.slug,
            )
            apply_draft_to_skill(skill, draft)
            normalize_skill(skill, explicit_tool_steps=explicit_tool_steps)
            if slug and slug != skill.slug:
                if store.get_skill_by_slug(slug):
                    raise HomeAssistantError(f"Skill already exists for slug: {slug}")
                skill.slug = slug
            if "enabled" in payload:
                skill.enabled = bool(payload["enabled"])
        else:
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
        updated = store.update_skill(skill)
        if old_slug != updated.slug:
            delete_skill_file(directory, old_slug)
        return updated

    skill = await hass.async_add_executor_job(_update)
    await async_mirror_skill_to_file(hass, entry_id, skill)
    result = skill_to_dict(skill)
    result["markdown"] = skill_to_markdown(skill)
    result["file_path"] = str(skill_file_path(directory, skill.slug))
    return result


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
    directory = skills_directory(hass, entry_id)

    def _export() -> list[Skill]:
        total = store.count_skills()
        return store.list_recent(limit=max(total, 1))

    skills = await hass.async_add_executor_job(_export)
    payload: list[dict[str, Any]] = []
    for skill in skills:
        item = skill_to_dict(skill)
        if not skill.is_builtin:
            item["markdown"] = skill_to_markdown(skill)
            item["file_path"] = str(skill_file_path(directory, skill.slug))
        payload.append(item)
    return payload


async def import_skills(
    hass: HomeAssistant,
    entry_id: str,
    skills_payload: list[dict[str, Any]],
) -> int:
    """Import skills from JSON bundles or markdown strings."""
    count = 0
    for item in skills_payload:
        if not isinstance(item, dict):
            continue
        try:
            if item.get("markdown"):
                await create_skill(hass, entry_id, {"markdown": item["markdown"]})
            else:
                await create_skill(hass, entry_id, item)
            count += 1
        except HomeAssistantError:
            continue
    return count


async def list_skill_revisions(
    hass: HomeAssistant,
    entry_id: str,
    skill_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return revision history for a skill."""
    store = get_skill_store(hass, entry_id)

    def _load():
        return store.list_revisions(skill_id, limit=limit)

    revisions = await hass.async_add_executor_job(_load)
    return [
        {
            "id": rev.id,
            "skill_id": rev.skill_id,
            "version": rev.version,
            "reason": rev.reason,
            "created_at": rev.created_at,
        }
        for rev in revisions
    ]


async def restore_skill_revision(
    hass: HomeAssistant,
    entry_id: str,
    revision_id: str,
) -> dict[str, Any]:
    """Restore a skill from a saved revision."""
    store = get_skill_store(hass, entry_id)

    def _restore() -> Skill | None:
        return store.restore_revision(revision_id)

    skill = await hass.async_add_executor_job(_restore)
    if skill is None:
        raise HomeAssistantError(f"Revision not found: {revision_id}")
    await async_mirror_skill_to_file(hass, entry_id, skill)
    result = skill_to_dict(skill)
    result["markdown"] = skill_to_markdown(skill)
    result["file_path"] = str(
        skill_file_path(skills_directory(hass, entry_id), skill.slug)
    )
    return result


async def sync_skill_files(hass: HomeAssistant, entry_id: str) -> dict[str, Any]:
    """Import markdown skill files from disk and backfill missing files."""
    result = await async_sync_skill_files(hass, entry_id)
    return {
        "directory": result.directory,
        "imported": result.imported,
        "written": result.written,
        "skipped": result.skipped,
    }


async def get_skills_directory(hass: HomeAssistant, entry_id: str) -> dict[str, str]:
    """Return the on-disk skills directory and starter template."""
    directory = skills_directory(hass, entry_id)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "directory": str(directory),
        "template": new_skill_markdown(),
    }

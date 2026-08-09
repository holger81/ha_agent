"""Skills API for the HA Agent console."""

from __future__ import annotations

import time
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
from ..skills.creator import (
    create_skill_from_trace,
    persist_skill_draft,
    save_skill_from_draft,
)
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
from ..skills.store import get_skill_store, revision_snapshot_summary
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
                llm_model=draft.llm_model,
                llm_base_url=draft.llm_base_url,
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

        revision_reason = str(
            payload.get("revision_reason") or "Manual update"
        ).strip()[:512]
        store.save_revision(skill, reason=revision_reason or "Manual update")

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

    if draft.skill_draft is not None:
        skill = await persist_skill_draft(
            hass,
            entry_id,
            draft.skill_draft,
            trace=draft.trace,
            update_skill_id=draft.update_skill_id,
        )
    else:
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
            update_skill_id=draft.update_skill_id,
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
            **revision_snapshot_summary(rev.snapshot_json),
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
        "repaired": result.repaired,
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


async def propose_skill_generalize(
    hass: HomeAssistant,
    entry_id: str,
    *,
    min_cluster: int = 2,
) -> dict[str, Any]:
    """Return merge clusters for similar learned skills (preview only)."""
    from ..skills.generalize import cluster_skills_for_generalize, cluster_to_dict

    store = get_skill_store(hass, entry_id)

    def _load() -> list[Skill]:
        total = max(store.count_skills(), 1)
        return store.list_recent(limit=total)

    skills = await hass.async_add_executor_job(_load)
    clusters = cluster_skills_for_generalize(skills, min_cluster=min_cluster)
    return {
        "clusters": [cluster_to_dict(cluster) for cluster in clusters],
        "count": len(clusters),
    }


async def apply_skill_generalize(
    hass: HomeAssistant,
    entry_id: str,
    *,
    skill_ids: list[str],
    survivor_id: str | None = None,
    archive_others: bool = True,
) -> dict[str, Any]:
    """Merge similar skills into one survivor and optionally disable the rest."""
    from ..skills.generalize import build_generalized_draft

    if len(skill_ids) < 2:
        raise HomeAssistantError("Need at least two skill ids to generalize.")

    store = get_skill_store(hass, entry_id)

    def _load_members() -> list[Skill]:
        members: list[Skill] = []
        for skill_id in skill_ids:
            skill = store.get_skill(skill_id)
            if skill is None:
                raise HomeAssistantError(f"Skill not found: {skill_id}")
            if skill.is_builtin:
                raise HomeAssistantError(
                    f"Cannot merge built-in skill: {skill.slug}"
                )
            members.append(skill)
        return members

    members = await hass.async_add_executor_job(_load_members)
    by_id = {skill.id: skill for skill in members}
    if survivor_id and survivor_id not in by_id:
        raise HomeAssistantError(f"Survivor not in skill_ids: {survivor_id}")
    survivor = by_id[survivor_id] if survivor_id else max(
        members,
        key=lambda skill: (
            int(skill.use_count or 0),
            float(skill.score or 0.0),
            float(skill.created_at or 0.0),
        ),
    )

    draft = normalize_skill_draft(
        build_generalized_draft(members),
        explicit_tool_steps=True,
    )
    updated = await save_skill_from_draft(
        hass,
        entry_id,
        draft,
        update_existing=survivor,
        revision_reason="Skill generalization merge",
    )

    archived: list[str] = []
    if archive_others:

        def _archive() -> list[str]:
            done: list[str] = []
            for skill in members:
                if skill.id == updated.id:
                    continue
                skill.parent_id = updated.id
                skill.enabled = False
                store.update_skill(skill)
                done.append(skill.id)
            return done

        archived = await hass.async_add_executor_job(_archive)
        for skill_id in archived:
            child = await hass.async_add_executor_job(store.get_skill, skill_id)
            if child is not None:
                await async_mirror_skill_to_file(hass, entry_id, child)

    result = skill_to_dict(updated)
    result["markdown"] = skill_to_markdown(updated)
    result["file_path"] = str(
        skill_file_path(skills_directory(hass, entry_id), updated.slug)
    )
    return {
        "survivor": result,
        "archived_skill_ids": archived,
        "merged_count": len(members),
    }


async def propose_skill_simplify(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Ask a strong model how to simplify/combine skills (preview only)."""
    from ..skills.simplify import (
        propose_skill_simplification,
        resolve_strong_simplify_backend,
    )

    store = get_skill_store(hass, entry_id)

    def _load() -> list[Skill]:
        total = max(store.count_skills(), 1)
        return store.list_recent(limit=total)

    skills = await hass.async_add_executor_job(_load)
    session = async_get_clientsession(hass)
    llm = LlmClient(session)
    backend, model_label = await resolve_strong_simplify_backend(hass, entry_id)
    try:
        return await propose_skill_simplification(
            hass,
            entry_id,
            llm,
            backend,
            skills=skills,
            model_label=model_label,
        )
    except HomeAssistantError:
        raise
    except Exception as err:
        raise HomeAssistantError(
            f"Skill simplification propose failed: {err}"
        ) from err


async def apply_skill_simplify(
    hass: HomeAssistant,
    entry_id: str,
    *,
    proposal_id: str,
) -> dict[str, Any]:
    """Apply a previously proposed simplify/combine draft."""
    from ..skills.simplify import (
        SkillSimplifyUndo,
        get_simplify_state,
        undo_to_dict,
    )

    state = get_simplify_state(hass, entry_id)
    proposal = next(
        (item for item in state.proposals if item.proposal_id == proposal_id),
        None,
    )
    if proposal is None:
        raise HomeAssistantError(
            "Simplify proposal not found. Run Simplify skills again."
        )

    store = get_skill_store(hass, entry_id)

    def _load_members() -> list[Skill]:
        members: list[Skill] = []
        for skill_id in proposal.skill_ids:
            skill = store.get_skill(skill_id)
            if skill is None:
                raise HomeAssistantError(f"Skill not found: {skill_id}")
            if skill.is_builtin:
                raise HomeAssistantError(
                    f"Cannot simplify built-in skill: {skill.slug}"
                )
            members.append(skill)
        return members

    members = await hass.async_add_executor_job(_load_members)
    by_id = {skill.id: skill for skill in members}
    survivor = by_id.get(proposal.survivor_id) or members[0]

    draft = normalize_skill_draft(proposal.draft, explicit_tool_steps=True)
    updated = await save_skill_from_draft(
        hass,
        entry_id,
        draft,
        update_existing=survivor,
        revision_reason=f"Skill {proposal.action}",
    )

    def _latest_revision_id() -> str | None:
        revisions = store.list_revisions(updated.id, limit=1)
        return revisions[0].id if revisions else None

    revision_id = await hass.async_add_executor_job(_latest_revision_id)
    if not revision_id:
        raise HomeAssistantError("Could not record revision for undo.")

    archived_meta: list[dict[str, Any]] = []
    if proposal.action == "combine" and len(members) > 1:

        def _archive() -> list[dict[str, Any]]:
            done: list[dict[str, Any]] = []
            for skill in members:
                if skill.id == updated.id:
                    continue
                done.append(
                    {
                        "id": skill.id,
                        "enabled": bool(skill.enabled),
                        "parent_id": skill.parent_id,
                    }
                )
                skill.parent_id = updated.id
                skill.enabled = False
                store.update_skill(skill)
            return done

        archived_meta = await hass.async_add_executor_job(_archive)
        for item in archived_meta:
            child = await hass.async_add_executor_job(store.get_skill, item["id"])
            if child is not None:
                await async_mirror_skill_to_file(hass, entry_id, child)

    state.undo = SkillSimplifyUndo(
        survivor_id=updated.id,
        revision_id=revision_id,
        archived=archived_meta,
        proposal_id=proposal.proposal_id,
        summary=(
            f"{proposal.action.title()}: {updated.title}"
            + (
                f" (archived {len(archived_meta)})"
                if archived_meta
                else ""
            )
        ),
        created_at=time.time(),
    )
    state.proposals = [
        item for item in state.proposals if item.proposal_id != proposal_id
    ]

    result = skill_to_dict(updated)
    result["markdown"] = skill_to_markdown(updated)
    result["file_path"] = str(
        skill_file_path(skills_directory(hass, entry_id), updated.slug)
    )
    return {
        "survivor": result,
        "archived_skill_ids": [item["id"] for item in archived_meta],
        "action": proposal.action,
        "undo": undo_to_dict(state.undo),
        "remaining_proposals": len(state.proposals),
    }


async def undo_skill_simplify(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Reverse the last applied simplification (restore survivor + siblings)."""
    from ..skills.simplify import get_simplify_state, undo_to_dict

    state = get_simplify_state(hass, entry_id)
    undo = state.undo
    if undo is None:
        raise HomeAssistantError("Nothing to undo.")

    store = get_skill_store(hass, entry_id)
    restored = await restore_skill_revision(hass, entry_id, undo.revision_id)

    def _restore_archived() -> list[str]:
        restored_ids: list[str] = []
        for item in undo.archived:
            skill = store.get_skill(str(item["id"]))
            if skill is None:
                continue
            skill.enabled = bool(item.get("enabled", True))
            skill.parent_id = item.get("parent_id")
            store.update_skill(skill)
            restored_ids.append(skill.id)
        return restored_ids

    restored_ids = await hass.async_add_executor_job(_restore_archived)
    for skill_id in restored_ids:
        child = await hass.async_add_executor_job(store.get_skill, skill_id)
        if child is not None:
            await async_mirror_skill_to_file(hass, entry_id, child)

    summary = undo.summary
    state.undo = None
    return {
        "survivor": restored,
        "restored_skill_ids": restored_ids,
        "summary": summary,
        "undo": undo_to_dict(state.undo),
    }


async def get_skill_simplify_status(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Return current simplify proposals and undo availability."""
    from ..skills.simplify import get_simplify_state, proposal_to_dict, undo_to_dict

    state = get_simplify_state(hass, entry_id)
    return {
        "proposals": [proposal_to_dict(item) for item in state.proposals],
        "count": len(state.proposals),
        "summary": state.summary,
        "model_used": state.model_used,
        "undo": undo_to_dict(state.undo),
    }

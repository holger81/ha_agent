"""Filesystem-backed skill markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from homeassistant.core import HomeAssistant

from ..const import LOGGER
from .body import normalize_skill
from .bundled import (
    apply_bundled_skill,
    email_skill_needs_refresh,
    seed_missing_bundled_skills,
)
from .markdown import (
    NEW_SKILL_MARKDOWN,
    apply_draft_to_skill,
    draft_from_markdown,
    skill_to_markdown,
)
from .models import Skill
from .store import SkillStore, get_skill_store


@dataclass(slots=True)
class SkillFileSyncResult:
    """Outcome of syncing markdown skill files with the database."""

    directory: str
    imported: int = 0
    written: int = 0
    repaired: int = 0
    skipped: int = 0


def skills_directory(hass: HomeAssistant, entry_id: str) -> Path:
    """Return the config directory where skill .md files live."""
    return Path(hass.config.path(f"ha_agent/skills/{entry_id}"))


def skill_file_path(directory: Path, slug: str) -> Path:
    """Return the markdown path for a skill slug."""
    safe = slug.strip().replace("/", "-").replace("\\", "-")
    return directory / f"{safe}.md"


def write_skill_file(directory: Path, skill: Skill) -> Path:
    """Write a skill to its markdown file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = skill_file_path(directory, skill.slug)
    path.write_text(skill_to_markdown(skill), encoding="utf-8")
    return path


def delete_skill_file(directory: Path, slug: str) -> None:
    """Remove a skill markdown file if present."""
    path = skill_file_path(directory, slug)
    if path.is_file():
        path.unlink()


def mirror_skill_to_file(
    hass: HomeAssistant, entry_id: str, skill: Skill
) -> Path | None:
    """Persist a learned skill as markdown. Builtins are not mirrored."""
    if skill.is_builtin:
        return None
    directory = skills_directory(hass, entry_id)
    return write_skill_file(directory, skill)


def _import_file(store: SkillStore, path: Path) -> bool:
    slug_hint = path.stem
    text = path.read_text(encoding="utf-8")
    draft, slug_override, _explicit = draft_from_markdown(
        text,
        filename_slug=slug_hint,
    )
    slug = slug_override or slug_hint
    existing = store.get_skill_by_slug(slug)
    if existing is not None:
        if existing.is_builtin:
            LOGGER.warning("Skipping skill file %s: slug matches builtin skill", path)
            return False
        apply_draft_to_skill(existing, draft)
        normalize_skill(existing)
        store.update_skill(existing)
        return True

    skill = store.insert_skill(
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
    )
    if skill.slug != slug:
        write_skill_file(path.parent, skill)
        delete_skill_file(path.parent, slug)
    return True


def sync_skill_files(hass: HomeAssistant, entry_id: str) -> SkillFileSyncResult:
    """Import .md files into SQLite and backfill missing files for DB skills."""
    store = get_skill_store(hass, entry_id)
    directory = skills_directory(hass, entry_id)
    directory.mkdir(parents=True, exist_ok=True)

    result = SkillFileSyncResult(directory=str(directory))
    seeded = seed_missing_bundled_skills(store, directory)
    result.imported += seeded

    for path in sorted(directory.glob("*.md")):
        try:
            if _import_file(store, path):
                result.imported += 1
            else:
                result.skipped += 1
        except Exception as err:
            result.skipped += 1
            LOGGER.warning("Could not import skill file %s: %s", path, err)

    total = store.count_skills()
    for skill in store.list_recent(limit=max(total, 1)):
        if skill.is_builtin:
            continue
        before = skill_to_markdown(skill)
        if email_skill_needs_refresh(skill):
            apply_bundled_skill(skill)
        normalize_skill(skill)
        after = skill_to_markdown(skill)
        file_path = skill_file_path(directory, skill.slug)
        if after != before:
            store.update_skill(skill)
            write_skill_file(directory, skill)
            result.repaired += 1
        elif not file_path.is_file():
            write_skill_file(directory, skill)
            result.written += 1

    return result


async def async_sync_skill_files(
    hass: HomeAssistant,
    entry_id: str,
) -> SkillFileSyncResult:
    """Async wrapper for filesystem skill sync."""
    return await hass.async_add_executor_job(sync_skill_files, hass, entry_id)


async def async_mirror_skill_to_file(
    hass: HomeAssistant,
    entry_id: str,
    skill: Skill,
) -> Path | None:
    """Async wrapper for writing one skill markdown file."""
    return await hass.async_add_executor_job(
        mirror_skill_to_file,
        hass,
        entry_id,
        skill,
    )


def new_skill_markdown() -> str:
    """Return a starter markdown template for new skills."""
    return NEW_SKILL_MARKDOWN

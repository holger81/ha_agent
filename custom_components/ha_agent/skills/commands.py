"""Chat commands and HA services for skill administration."""

from __future__ import annotations

import re
from typing import Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse, callback

from ..const import DATA_KEY, DOMAIN, LOGGER
from ..context import is_affirmative
from .creator import create_skill_from_trace
from .models import PendingSkillDraft
from .runtime import pop_pending_draft, set_pending_draft
from .store import get_skill_store

_ENABLE = re.compile(
    r"\b(?:enable|turn on|activate)\b.*\bskill\b",
    re.IGNORECASE,
)
_DISABLE = re.compile(
    r"\b(?:disable|turn off|deactivate)\b.*\bskill\b",
    re.IGNORECASE,
)
_DELETE = re.compile(r"\bdelete\b.*\bskill\b", re.IGNORECASE)
_LIST = re.compile(r"\blist\b.*\bskills?\b", re.IGNORECASE)
_MANUAL_SAVE = re.compile(
    r"\b(?:save|remember|store)\b.*\b(?:as a skill|this as a skill|how to do this)\b",
    re.IGNORECASE,
)
_SKILL_NAME = re.compile(
    r"\bskill\s+(?:called\s+)?[\"']?([^\"'.?!]+)[\"']?",
    re.IGNORECASE,
)


@callback
def is_skill_admin_query(query: str) -> bool:
    """Return True when the user is managing skills."""
    return bool(
        _ENABLE.search(query)
        or _DISABLE.search(query)
        or _DELETE.search(query)
        or _LIST.search(query)
        or _MANUAL_SAVE.search(query)
    )


def _extract_skill_query(text: str) -> str:
    """Extract the skill name phrase from a command."""
    if match := _SKILL_NAME.search(text):
        return match.group(1).strip()
    cleaned = re.sub(
        r"\b(?:enable|disable|delete|turn on|turn off|activate|deactivate|"
        r"the|a|my|skill|skills)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


async def _resolve_skill_by_query(
    hass: HomeAssistant,
    entry_id: str,
    query: str,
) -> Any:
    """Resolve a skill from a user phrase via FTS."""
    store = get_skill_store(hass, entry_id)
    name_query = _extract_skill_query(query)
    if not name_query:
        return None

    def _search():
        matches = store.search(name_query, limit=1, enabled_only=False)
        if not matches:
            return None
        return store.get_skill(matches[0].id)

    return await hass.async_add_executor_job(_search)


async def try_handle_skill_command(
    hass: HomeAssistant,
    entry_id: str,
    user_text: str,
) -> str | None:
    """Handle skill admin chat commands. Return reply text or None."""
    if _LIST.search(user_text):
        return await _cmd_list_skills(hass, entry_id)
    if _ENABLE.search(user_text):
        return await _cmd_set_skill_enabled(hass, entry_id, user_text, enabled=True)
    if _DISABLE.search(user_text):
        return await _cmd_set_skill_enabled(hass, entry_id, user_text, enabled=False)
    if _DELETE.search(user_text):
        return await _cmd_delete_skill(hass, entry_id, user_text)
    return None


async def try_confirm_pending_save(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str | None,
    user_text: str,
    *,
    llm,
    backend,
) -> str | None:
    """Save a pending skill draft when the user affirms."""
    if not is_affirmative(user_text):
        return None
    draft = pop_pending_draft(hass, conversation_id)
    if draft is None or draft.entry_id != entry_id:
        return None
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
        set_pending_draft(hass, draft)
        return "I couldn't save that skill. Please try again."
    return f"Saved skill: {skill.title}."


def queue_pending_save(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str | None,
    *,
    trace,
    history: list[dict[str, str]],
    skill_draft=None,
    observer_reason: str = "",
) -> None:
    """Queue a skill draft for user confirmation."""
    if not conversation_id:
        return
    set_pending_draft(
        hass,
        PendingSkillDraft(
            entry_id=entry_id,
            conversation_id=conversation_id,
            trace=trace,
            history=history,
            skill_draft=skill_draft,
            observer_reason=observer_reason,
        ),
    )


async def _cmd_list_skills(hass: HomeAssistant, entry_id: str) -> str:
    store = get_skill_store(hass, entry_id)

    def _list():
        return store.list_recent(limit=10)

    skills = await hass.async_add_executor_job(_list)
    if not skills:
        return "You have no saved skills yet."
    lines = ["Recent skills:"]
    for skill in skills:
        state = "enabled" if skill.enabled else "disabled"
        lines.append(f"- {skill.title} ({state})")
    return " ".join(lines)


async def _cmd_set_skill_enabled(
    hass: HomeAssistant,
    entry_id: str,
    user_text: str,
    *,
    enabled: bool,
) -> str:
    skill = await _resolve_skill_by_query(hass, entry_id, user_text)
    if skill is None:
        return "I couldn't find that skill."
    store = get_skill_store(hass, entry_id)

    def _set():
        return store.set_enabled(skill.id, enabled)

    updated = await hass.async_add_executor_job(_set)
    if updated is None:
        return "I couldn't update that skill."
    verb = "enabled" if enabled else "disabled"
    return f"Skill {updated.title} is now {verb}."


async def _cmd_delete_skill(
    hass: HomeAssistant,
    entry_id: str,
    user_text: str,
) -> str:
    skill = await _resolve_skill_by_query(hass, entry_id, user_text)
    if skill is None:
        return "I couldn't find that skill."
    store = get_skill_store(hass, entry_id)

    def _delete():
        return store.delete_skill(skill.id)

    if await hass.async_add_executor_job(_delete):
        return f"Deleted skill {skill.title}."
    return "I couldn't delete that skill."


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register ha_agent skill services."""
    if hass.data.get(DATA_KEY, {}).get("services_registered"):
        return

    async def enable_skill(call: ServiceCall) -> None:
        entry_id = _entry_id_from_call(hass, call)
        skill_id = call.data.get("skill_id")
        if not entry_id or not skill_id:
            return
        store = get_skill_store(hass, entry_id)
        await hass.async_add_executor_job(store.set_enabled, skill_id, True)

    async def disable_skill(call: ServiceCall) -> None:
        entry_id = _entry_id_from_call(hass, call)
        skill_id = call.data.get("skill_id")
        if not entry_id or not skill_id:
            return
        store = get_skill_store(hass, entry_id)
        await hass.async_add_executor_job(store.set_enabled, skill_id, False)

    async def delete_skill(call: ServiceCall) -> None:
        entry_id = _entry_id_from_call(hass, call)
        skill_id = call.data.get("skill_id")
        if not entry_id or not skill_id:
            return
        store = get_skill_store(hass, entry_id)
        await hass.async_add_executor_job(store.delete_skill, skill_id)

    async def list_skills(call: ServiceCall):
        entry_id = _entry_id_from_call(hass, call)
        if not entry_id:
            return {"skills": []}
        store = get_skill_store(hass, entry_id)
        skills = await hass.async_add_executor_job(store.list_recent, limit=50)
        payload = [
            {
                "id": skill.id,
                "title": skill.title,
                "enabled": skill.enabled,
                "use_count": skill.use_count,
            }
            for skill in skills
        ]
        LOGGER.info(
            "HA Agent skills for %s: %s",
            entry_id,
            ", ".join(skill["title"] for skill in payload) or "none",
        )
        return {"skills": payload}

    hass.services.async_register(
        DOMAIN,
        "enable_skill",
        enable_skill,
        schema=_skill_service_schema(),
    )
    hass.services.async_register(
        DOMAIN,
        "disable_skill",
        disable_skill,
        schema=_skill_service_schema(),
    )
    hass.services.async_register(
        DOMAIN,
        "delete_skill",
        delete_skill,
        schema=_skill_service_schema(),
    )
    hass.services.async_register(
        DOMAIN,
        "list_skills",
        list_skills,
        schema=_list_skills_schema(),
        supports_response=SupportsResponse.ONLY,
    )
    hass.data.setdefault(DATA_KEY, {})["services_registered"] = True


def _list_skills_schema():
    import voluptuous as vol

    return vol.Schema({vol.Optional("entry_id"): str})


def _skill_service_schema():
    import voluptuous as vol

    return vol.Schema(
        {
            vol.Optional("entry_id"): str,
            vol.Optional("skill_id"): str,
        }
    )


def _entry_id_from_call(hass: HomeAssistant, call: ServiceCall) -> str | None:
    if entry_id := call.data.get("entry_id"):
        return entry_id
    entries = hass.config_entries.async_entries(DOMAIN)
    if len(entries) == 1:
        return entries[0].entry_id
    return None

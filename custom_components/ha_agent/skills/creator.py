"""Distill learned skills from successful agent turns."""

from __future__ import annotations

import json
import re

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from .models import Skill, SkillDraft, TurnTrace
from .store import get_skill_store

_DISTILL_PROMPT = (
    "You distill successful Home Assistant assistant workflows into "
    "reusable skills.\n"
    "Return ONLY valid JSON with keys: title, description, triggers, body, "
    "tool_steps.\n"
    "- title: short human name (max 64 chars)\n"
    "- description: third-person WHAT + WHEN for discovery (max 512 chars)\n"
    "- triggers: list of 3-8 example user phrases that should match this skill\n"
    "- body: markdown workflow the agent should follow (steps, entity notes)\n"
    "- tool_steps: list of objects with toolName and arguments from the run\n"
    "Do not invent entity_id values not present in the trace."
)


async def distill_skill_draft(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    trace: TurnTrace,
    history: list[dict[str, str]],
) -> SkillDraft | None:
    """Use the LLM to distill a skill from a successful turn trace."""
    payload = {
        "user_goal": trace.user_text,
        "history": history[-10:],
        "tool_calls": trace.tool_calls,
        "controlled_entity_ids": trace.controlled_entity_ids,
    }
    messages = [
        {"role": "system", "content": _DISTILL_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=True),
        },
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill distillation failed: %s", err)
        return None

    content = (result.content or "").strip()
    if not content:
        return None
    return _parse_skill_draft(content)


def _parse_skill_draft(content: str) -> SkillDraft | None:
    """Parse LLM JSON output into a SkillDraft."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.debug("Skill distillation returned invalid JSON: %s", content[:200])
        return None
    if not isinstance(data, dict):
        return None

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    body = str(data.get("body", "")).strip()
    triggers_raw = data.get("triggers", [])
    tool_steps_raw = data.get("tool_steps", [])
    if not title or not description or not body:
        return None

    triggers = (
        [str(item).strip() for item in triggers_raw if str(item).strip()]
        if isinstance(triggers_raw, list)
        else []
    )
    tool_steps = (
        [item for item in tool_steps_raw if isinstance(item, dict)]
        if isinstance(tool_steps_raw, list)
        else []
    )
    if not triggers:
        triggers = [title]

    return SkillDraft(
        title=title,
        description=description,
        triggers=triggers,
        body=body,
        tool_steps=tool_steps,
    )


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
) -> Skill | None:
    """Distill and save a skill from a turn trace."""
    draft = await distill_skill_draft(
        llm,
        backend,
        trace=trace,
        history=history,
    )
    if draft is None:
        return None

    store = get_skill_store(hass, entry_id)

    def _find_dup() -> Skill | None:
        return store.find_duplicate(draft.triggers)

    duplicate = await hass.async_add_executor_job(_find_dup)
    return await save_skill_from_draft(
        hass,
        entry_id,
        draft,
        update_existing=duplicate,
    )

"""Built-in route-scoped skills from playbooks (unified skill memory)."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from ..playbooks import (
    DEFAULT_PLAYBOOKS,
    async_select_playbook,
    playbook_key_for_route,
)
from .models import Skill

_BUILTIN_SLUG_PREFIX = "builtin-route-"


def route_skill_slug(route_value: str) -> str:
    """Return the slug for a built-in route skill."""
    key = playbook_key_for_route(route_value)
    return f"{_BUILTIN_SLUG_PREFIX}{key}"


def builtin_route_skill(route_value: str, body: str, *, key: str) -> Skill:
    """Build a Skill object from a route playbook body."""
    default = DEFAULT_PLAYBOOKS.get(key) or DEFAULT_PLAYBOOKS["general"]
    return Skill(
        id=f"builtin-{key}",
        slug=route_skill_slug(route_value),
        title=default["title"],
        description=default.get("match", default["title"]),
        triggers=[route_value, key],
        body=body,
        tool_steps=[],
        enabled=True,
        route_scope=route_value,
        is_builtin=True,
        score=0.5,
    )


async def load_route_skill(
    hass: HomeAssistant,
    entry_id: str,
    llm: object,
    backend: object,
    *,
    user_text: str,
    route_value: str,
    history: list[dict[str, str]] | None,
) -> Skill:
    """Return the active route playbook as a built-in Skill."""
    selection = await async_select_playbook(
        hass,
        entry_id,
        llm,
        backend,
        user_text=user_text,
        route_value=route_value,
        history=history,
    )
    key = selection.key
    return builtin_route_skill(route_value, selection.body, key=key)


def merge_route_and_learned_skills(
    route_skill: Skill | None,
    learned: list[Skill],
) -> list[Skill]:
    """Inject route baseline before learned skills; learned skills win on match."""
    if route_skill is None:
        return learned
    if not learned:
        return [route_skill]
    return [route_skill, *learned]

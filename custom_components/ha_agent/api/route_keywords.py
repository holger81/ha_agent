"""Route keyword API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..route_keywords import RouteKeywords, get_route_keyword_store
from .serialize import route_keywords_to_dict


async def list_route_keywords(
    hass: HomeAssistant,
    entry_id: str,
) -> list[dict[str, Any]]:
    """Return editable trigger keyword lists for each built-in route."""
    store = get_route_keyword_store(hass, entry_id)
    items = await hass.async_add_executor_job(store.list_route_keywords)
    return [route_keywords_to_dict(item) for item in items]


async def update_route_keywords(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update a route's editable trigger keywords."""
    store = get_route_keyword_store(hass, entry_id)
    keywords = payload.get("keywords")
    enabled = payload.get("enabled")
    parsed_keywords: list[str] | None = None
    if keywords is not None:
        if not isinstance(keywords, list):
            raise HomeAssistantError("keywords must be a list of strings")
        parsed_keywords = [str(keyword) for keyword in keywords]

    def _update() -> RouteKeywords | None:
        return store.update_route_keywords(
            route,
            keywords=parsed_keywords,
            enabled=None if enabled is None else bool(enabled),
        )

    item = await hass.async_add_executor_job(_update)
    if item is None:
        raise HomeAssistantError(f"Route not found: {route}")
    return route_keywords_to_dict(item)


async def set_route_keywords_enabled(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a route's keyword override."""
    store = get_route_keyword_store(hass, entry_id)

    def _set() -> RouteKeywords | None:
        return store.update_route_keywords(route, enabled=enabled)

    item = await hass.async_add_executor_job(_set)
    if item is None:
        raise HomeAssistantError(f"Route not found: {route}")
    return route_keywords_to_dict(item)


async def reset_route_keywords(
    hass: HomeAssistant,
    entry_id: str,
    route: str,
) -> dict[str, Any]:
    """Restore a route's keywords to the shipped default."""
    store = get_route_keyword_store(hass, entry_id)

    def _reset() -> RouteKeywords | None:
        return store.reset_route_keywords(route)

    item = await hass.async_add_executor_job(_reset)
    if item is None:
        raise HomeAssistantError(f"Route not found: {route}")
    return route_keywords_to_dict(item)

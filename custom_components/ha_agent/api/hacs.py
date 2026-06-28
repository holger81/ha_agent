"""HACS update helpers for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

HACS_REPOSITORY = "holger81/ha_agent"
HACS_DOMAIN = "hacs"
UPDATE_ENTITY_CANDIDATES = (
    "update.ha_agent_update",
    "update.ha_agent_ha_agent_update",
)


def _get_hacs(hass: HomeAssistant) -> Any | None:
    return hass.data.get(HACS_DOMAIN)


def _get_repository(hass: HomeAssistant) -> Any | None:
    hacs = _get_hacs(hass)
    if hacs is None:
        return None
    repositories = getattr(hacs, "repositories", None)
    if repositories is None:
        return None
    getter = getattr(repositories, "get_by_full_name", None)
    if callable(getter):
        repo = getter(HACS_REPOSITORY)
        if repo is not None:
            return repo
    for repo in getattr(repositories, "list_downloaded", []) or []:
        data = getattr(repo, "data", None)
        full_name = getattr(data, "full_name", "") if data is not None else ""
        if str(full_name).lower() == HACS_REPOSITORY.lower():
            return repo
    return None


def _find_update_entity(hass: HomeAssistant) -> str | None:
    registry = er.async_get(hass)
    for entity_id in UPDATE_ENTITY_CANDIDATES:
        if registry.async_get(entity_id) is not None:
            return entity_id
    for entry in registry.entities.values():
        if entry.domain != "update" or entry.platform != HACS_DOMAIN:
            continue
        state = hass.states.get(entry.entity_id)
        if state is None:
            continue
        release_url = str(state.attributes.get("release_url") or "")
        if HACS_REPOSITORY.lower() in release_url.lower():
            return entry.entity_id
    return None


def _version_from_repo(repo: Any) -> tuple[str | None, str | None]:
    data = getattr(repo, "data", None)
    if data is None:
        return None, None
    installed = getattr(data, "installed_version", None) or getattr(
        data, "installed_commit", None
    )
    latest = (
        getattr(data, "last_version", None)
        or getattr(data, "available_version", None)
        or getattr(repo, "display_version", None)
    )
    return (
        str(installed) if installed else None,
        str(latest) if latest else None,
    )


def _update_available(installed: str | None, latest: str | None) -> bool:
    if not latest:
        return False
    if not installed:
        return True
    return installed != latest


def get_update_status(hass: HomeAssistant) -> dict[str, Any]:
    """Return HACS update metadata for this integration."""
    hacs = _get_hacs(hass)
    repo = _get_repository(hass)
    entity_id = _find_update_entity(hass)
    state = hass.states.get(entity_id) if entity_id else None

    installed_version: str | None = None
    latest_version: str | None = None
    update_available = False
    in_progress = False

    if state is not None:
        installed_version = state.attributes.get("installed_version")
        latest_version = state.attributes.get("latest_version")
        update_available = state.state == "on"
        in_progress = bool(state.attributes.get("in_progress"))

    if repo is not None:
        repo_installed, repo_latest = _version_from_repo(repo)
        installed_version = installed_version or repo_installed
        latest_version = latest_version or repo_latest
        if not update_available:
            update_available = _update_available(installed_version, latest_version)

    return {
        "hacs_available": hacs is not None,
        "repository_found": repo is not None,
        "entity_id": entity_id,
        "installed_version": installed_version,
        "latest_version": latest_version,
        "update_available": update_available,
        "in_progress": in_progress,
        "repository": HACS_REPOSITORY,
    }


async def refresh_repository(hass: HomeAssistant) -> dict[str, Any]:
    """Force HACS to refresh repository metadata from GitHub."""
    if _get_hacs(hass) is None:
        raise HomeAssistantError("HACS is not installed.")
    repo = _get_repository(hass)
    if repo is None:
        raise HomeAssistantError(
            f"{HACS_REPOSITORY} is not registered in HACS. "
            "Add it under HACS → Integrations."
        )
    await repo.update_repository(ignore_issues=True, force=True)
    hacs = _get_hacs(hass)
    if hacs is not None and hasattr(hacs, "data"):
        await hacs.data.async_write()
    entity_id = _find_update_entity(hass)
    if entity_id:
        await hass.services.async_call(
            "homeassistant",
            "update_entity",
            {"entity_id": entity_id},
            blocking=True,
        )
    return get_update_status(hass)


async def install_update(
    hass: HomeAssistant,
    *,
    force_refresh: bool = False,
    force_reinstall: bool = False,
) -> dict[str, Any]:
    """Check for updates and install when available (or force redownload)."""
    if _get_hacs(hass) is None:
        raise HomeAssistantError("HACS is not installed.")
    repo = _get_repository(hass)
    if repo is None:
        raise HomeAssistantError(
            f"{HACS_REPOSITORY} is not registered in HACS. "
            "Add it under HACS → Integrations."
        )

    status = (
        await refresh_repository(hass) if force_refresh else get_update_status(hass)
    )
    if not status["update_available"] and not force_reinstall:
        return {**status, "installed": False}

    entity_id = status.get("entity_id")
    if entity_id:
        await hass.services.async_call(
            "update",
            "install",
            {"entity_id": entity_id},
            blocking=True,
        )
    else:
        downloader = getattr(repo, "async_download_repository", None)
        if callable(downloader):
            await downloader()
        else:
            await repo.async_install()

    refreshed = await refresh_repository(hass)
    return {**refreshed, "installed": True}

"""Conversation thread metadata for the HA Agent console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, callback

from .const import DATA_KEY, DOMAIN, LOGGER

THREADS_KEY = "conversation_threads"


def _threads_path(hass: HomeAssistant, entry_id: str) -> Path:
    return Path(hass.config.path(".storage")) / f"{DOMAIN}_threads_{entry_id}.json"


@callback
def _threads_store(hass: HomeAssistant) -> dict[str, dict[str, dict[str, Any]]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(THREADS_KEY, {})


@callback
def get_threads(hass: HomeAssistant, entry_id: str) -> dict[str, dict[str, Any]]:
    """Return thread metadata keyed by conversation_id."""
    store = _threads_store(hass)
    if entry_id not in store:
        store[entry_id] = {}
    return store[entry_id]


@callback
def upsert_thread(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    title: str | None = None,
    pinned: bool | None = None,
) -> dict[str, Any]:
    """Create or update thread metadata."""
    threads = get_threads(hass, entry_id)
    current = dict(threads.get(conversation_id, {}))
    if title is not None:
        current["title"] = title
    if pinned is not None:
        current["pinned"] = pinned
    current.setdefault("title", conversation_id[:8])
    current.setdefault("pinned", False)
    threads[conversation_id] = current
    return current


@callback
def list_threads(hass: HomeAssistant, entry_id: str) -> list[dict[str, Any]]:
    """Return thread list with conversation_id included."""
    threads = get_threads(hass, entry_id)
    result = [
        {"conversation_id": cid, **meta}
        for cid, meta in threads.items()
    ]
    result.sort(key=lambda item: (not item.get("pinned", False), item.get("title", "")))
    return result


async def async_load_threads(hass: HomeAssistant, entry_id: str) -> None:
    """Load thread metadata from disk."""
    path = _threads_path(hass, entry_id)
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        LOGGER.warning("Failed to load HA Agent threads for %s: %s", entry_id, err)
        return
    if isinstance(data, dict):
        _threads_store(hass)[entry_id] = data


async def async_save_threads(hass: HomeAssistant, entry_id: str) -> None:
    """Persist thread metadata to disk."""
    threads = _threads_store(hass).get(entry_id, {})
    path = _threads_path(hass, entry_id)
    try:
        path.write_text(json.dumps(threads, indent=2), encoding="utf-8")
    except OSError as err:
        LOGGER.warning("Failed to save HA Agent threads for %s: %s", entry_id, err)

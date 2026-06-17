"""Conversation thread metadata for the HA Agent console."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, callback

from .const import DATA_KEY, DOMAIN, LOGGER
from .memory import (
    _entry_wants_persist,
    _memory_store,
    async_save_memory,
    clear_conversation,
)
from .skills.runtime import pop_pending_draft

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
    current["updated_at"] = time.time()
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
    result.sort(
        key=lambda item: (
            not item.get("pinned", False),
            -(item.get("updated_at") or 0),
            item.get("title", ""),
        )
    )
    return result


def _match_snippet(text: str, query: str, *, radius: int = 48) -> str:
    """Return a short excerpt around the first query match."""
    lowered = text.lower()
    needle = query.lower()
    index = lowered.find(needle)
    if index == -1:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(needle) + radius)
    snippet = text[start:end].replace("\n", " ")
    if start > 0:
        snippet = f"…{snippet}"
    if end < len(text):
        snippet = f"{snippet}…"
    return snippet


def search_threads(
    hass: HomeAssistant,
    entry_id: str,
    query: str,
) -> list[dict[str, Any]]:
    """Return threads whose title or message history matches the query."""
    needle = query.strip().lower()
    if not needle:
        return list_threads(hass, entry_id)

    threads_map = get_threads(hass, entry_id)
    memory = _memory_store(hass)
    conversation_ids = set(threads_map) | set(memory)

    results: list[dict[str, Any]] = []
    for conversation_id in conversation_ids:
        meta = threads_map.get(conversation_id, {})
        title = str(meta.get("title") or conversation_id[:8])
        item: dict[str, Any] = {
            "conversation_id": conversation_id,
            "title": title,
            "pinned": bool(meta.get("pinned", False)),
            "updated_at": meta.get("updated_at") or 0,
        }
        if needle in title.lower():
            item["match_in"] = "title"
            results.append(item)
            continue

        for message in memory.get(conversation_id, []):
            content = str(message.get("content") or "")
            if needle in content.lower():
                item["match_in"] = "message"
                item["snippet"] = _match_snippet(content, needle)
                results.append(item)
                break

    results.sort(
        key=lambda row: (
            not row.get("pinned", False),
            -(row.get("updated_at") or 0),
            row.get("title", ""),
        )
    )
    return results


@callback
def delete_thread_metadata(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> bool:
    """Remove thread metadata for a conversation."""
    threads = get_threads(hass, entry_id)
    return threads.pop(conversation_id, None) is not None


async def async_delete_thread(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> bool:
    """Delete thread metadata, conversation history, and pending drafts."""
    had_thread = delete_thread_metadata(hass, entry_id, conversation_id)
    had_memory = conversation_id in _memory_store(hass)
    clear_conversation(hass, conversation_id)
    pop_pending_draft(hass, conversation_id)

    if not had_thread and not had_memory:
        return False

    await async_save_threads(hass, entry_id)
    if _entry_wants_persist(hass, entry_id):
        await async_save_memory(hass, entry_id)
    return True


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

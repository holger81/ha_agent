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
CONSOLE_CONVERSATION_PREFIX = "console-"


def conversation_source(conversation_id: str) -> str:
    """Return ``console`` or ``assist`` for a conversation id."""
    if conversation_id.startswith(CONSOLE_CONVERSATION_PREFIX):
        return "console"
    return "assist"


def _title_from_memory(
    memory: dict[str, list[dict[str, Any]]],
    conversation_id: str,
) -> str | None:
    """Infer a thread title from the first user message."""
    for message in memory.get(conversation_id, []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content[:48]
    return None


def _thread_item(
    conversation_id: str,
    meta: dict[str, Any],
    *,
    memory: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build a thread list row with inferred title and source."""
    item = {"conversation_id": conversation_id, **meta}
    title = str(item.get("title") or "").strip()
    if not title or title == conversation_id[:8]:
        inferred = _title_from_memory(memory, conversation_id)
        if inferred:
            item["title"] = inferred
    item.setdefault("title", conversation_id[:8])
    item.setdefault("pinned", False)
    item.setdefault("updated_at", 0)
    item.setdefault("source", conversation_source(conversation_id))
    return item


def _filter_threads_by_source(
    threads: list[dict[str, Any]],
    source: str | None,
) -> list[dict[str, Any]]:
    if not source:
        return threads
    return [item for item in threads if item.get("source") == source]


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
    source: str | None = None,
) -> dict[str, Any]:
    """Create or update thread metadata."""
    threads = get_threads(hass, entry_id)
    current = dict(threads.get(conversation_id, {}))
    if title is not None:
        current["title"] = title
    if pinned is not None:
        current["pinned"] = pinned
    if source is not None:
        current["source"] = source
    current["updated_at"] = time.time()
    current.setdefault("title", conversation_id[:8])
    current.setdefault("pinned", False)
    current.setdefault("source", conversation_source(conversation_id))
    threads[conversation_id] = current
    return current


@callback
def ensure_thread_from_turn(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str | None,
    *,
    user_text: str,
) -> bool:
    """Ensure thread metadata exists after a completed turn."""
    if not conversation_id or not entry_id:
        return False
    threads = get_threads(hass, entry_id)
    created = conversation_id not in threads
    upsert_thread(
        hass,
        entry_id,
        conversation_id,
        title=user_text[:48] if created and user_text.strip() else None,
        source=conversation_source(conversation_id),
    )
    return created


@callback
def list_threads(
    hass: HomeAssistant,
    entry_id: str,
    *,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return thread list with conversation_id included."""
    threads_map = get_threads(hass, entry_id)
    memory = _memory_store(hass)
    conversation_ids = set(threads_map) | set(memory)
    result = [
        _thread_item(
            conversation_id,
            dict(threads_map.get(conversation_id, {})),
            memory=memory,
        )
        for conversation_id in conversation_ids
    ]
    result = _filter_threads_by_source(result, source)
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
    *,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return threads whose title or message history matches the query."""
    needle = query.strip().lower()
    if not needle:
        return list_threads(hass, entry_id, source=source)

    threads_map = get_threads(hass, entry_id)
    memory = _memory_store(hass)
    conversation_ids = set(threads_map) | set(memory)

    results: list[dict[str, Any]] = []
    for conversation_id in conversation_ids:
        meta = threads_map.get(conversation_id, {})
        item = _thread_item(conversation_id, dict(meta), memory=memory)
        if source and item.get("source") != source:
            continue
        title = str(item.get("title") or conversation_id[:8])
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

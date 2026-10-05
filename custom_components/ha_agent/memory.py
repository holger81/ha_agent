"""Per-conversation history for multi-turn Assist and console."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from homeassistant.core import HomeAssistant, callback

from .const import CONF_CONVERSATION_MEMORY_PERSIST, DATA_KEY, DOMAIN, LOGGER

_T = TypeVar("_T")

MEMORY_KEY = "conversation_memory"
MEMORY_INDEX_KEY = "conversation_memory_index"
MEMORY_PENDING_SAVES_KEY = "conversation_memory_pending_saves"
MAX_CONVERSATIONS_PER_ENTRY = 200
_SAVE_DEBOUNCE_SECONDS = 0.05


def _memory_file(hass: HomeAssistant, entry_id: str) -> Path:
    return Path(hass.config.path(".storage")) / f"{DOMAIN}_memory_{entry_id}.json"


@callback
def _memory_store(hass: HomeAssistant) -> dict[str, list[dict[str, Any]]]:
    """Return flat conversation_id → messages (O(1) history lookup)."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(MEMORY_KEY, {})


@callback
def _memory_index(hass: HomeAssistant) -> dict[str, dict[str, float]]:
    """Return entry_id → {conversation_id: updated_at} ownership index."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(MEMORY_INDEX_KEY, {})


@callback
def _pending_memory_saves(hass: HomeAssistant) -> dict[str, asyncio.Task[Any]]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(MEMORY_PENDING_SAVES_KEY, {})


def _normalize_messages(raw: Any) -> list[dict[str, Any]]:
    """Keep only dict messages that include both role and content."""
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "role" not in item or "content" not in item:
            continue
        messages.append(item)
    return messages


def _parse_memory_payload(
    data: Any,
) -> dict[str, tuple[list[dict[str, Any]], float]]:
    """Parse disk JSON into conversation_id → (messages, updated_at).

    Supports legacy ``{cid: [messages]}`` and versioned
    ``{conversations: {cid: {messages, updated_at}}}`` shapes.
    """
    if not isinstance(data, dict):
        return {}

    conversations = data.get("conversations") if "conversations" in data else data
    if not isinstance(conversations, dict):
        return {}

    parsed: dict[str, tuple[list[dict[str, Any]], float]] = {}
    now = time.time()
    for conversation_id, raw in conversations.items():
        if not isinstance(conversation_id, str) or not conversation_id:
            continue
        if isinstance(raw, list):
            messages = _normalize_messages(raw)
            updated_at = now
        elif isinstance(raw, dict):
            messages = _normalize_messages(raw.get("messages"))
            try:
                updated_at = float(raw.get("updated_at") or now)
            except (TypeError, ValueError):
                updated_at = now
        else:
            continue
        if messages:
            parsed[conversation_id] = (messages, updated_at)
    return parsed


def _build_memory_payload(
    store: dict[str, list[dict[str, Any]]],
    entry_index: dict[str, float],
) -> dict[str, Any]:
    """Build entry-scoped disk payload from store + ownership index."""
    conversations: dict[str, Any] = {}
    for conversation_id, updated_at in entry_index.items():
        messages = store.get(conversation_id)
        if not messages:
            continue
        conversations[conversation_id] = {
            "messages": messages,
            "updated_at": updated_at,
        }
    return {"version": 1, "conversations": conversations}


def _apply_lru(
    store: dict[str, list[dict[str, Any]]],
    entry_index: dict[str, float],
    *,
    max_conversations: int = MAX_CONVERSATIONS_PER_ENTRY,
) -> list[str]:
    """Evict oldest conversations over the cap. Returns evicted ids."""
    if max_conversations <= 0 or len(entry_index) <= max_conversations:
        return []
    ordered = sorted(entry_index.items(), key=lambda item: item[1])
    overflow = len(entry_index) - max_conversations
    evicted = [conversation_id for conversation_id, _ in ordered[:overflow]]
    for conversation_id in evicted:
        entry_index.pop(conversation_id, None)
        store.pop(conversation_id, None)
    return evicted


def _touch_entry_conversation(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
) -> None:
    """Record ownership + updated_at and enforce per-entry LRU."""
    entry_index = _memory_index(hass).setdefault(entry_id, {})
    entry_index[conversation_id] = time.time()
    _apply_lru(_memory_store(hass), entry_index)


def _trim_history(
    store: dict[str, list[dict[str, Any]]],
    conversation_id: str,
    history: list[dict[str, Any]],
    *,
    max_turns: int,
) -> None:
    max_messages = max_turns * 2
    if len(history) > max_messages:
        store[conversation_id] = history[-max_messages:]


@callback
def get_history(
    hass: HomeAssistant,
    conversation_id: str | None,
    *,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Return stored history for a conversation."""
    if not conversation_id or max_turns <= 0:
        return []
    store = _memory_store(hass)
    history = store.get(conversation_id, [])
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return list(history)


@callback
def conversation_history_for_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    *,
    max_turns: int,
) -> list[dict[str, Any]]:
    """Return prior turns only, excluding the in-flight user message for this turn."""
    history = get_history(
        hass,
        conversation_id,
        max_turns=max_turns,
    )
    if not history:
        return []
    last = history[-1]
    if (
        last.get("role") == "user"
        and str(last.get("content", "")).strip() == user_text.strip()
    ):
        return list(history[:-1])
    return history


@callback
def append_user_message(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    *,
    max_turns: int,
    entry_id: str | None = None,
) -> None:
    """Append only the user side of a turn (for immediate history visibility)."""
    if not conversation_id or max_turns <= 0 or not user_text.strip():
        return

    store = _memory_store(hass)
    history = store.setdefault(conversation_id, [])
    last = history[-1] if history else None
    if last and last.get("role") == "user" and last.get("content") == user_text.strip():
        return
    history.append({"role": "user", "content": user_text.strip()})
    _trim_history(store, conversation_id, history, max_turns=max_turns)

    if entry_id:
        _touch_entry_conversation(hass, entry_id, conversation_id)
        _maybe_persist(hass, entry_id)


@callback
def append_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    user_text: str,
    assistant_text: str,
    *,
    max_turns: int,
    entry_id: str | None = None,
    turn_meta: dict[str, Any] | None = None,
) -> None:
    """Append a user/assistant turn to memory."""
    if not conversation_id or max_turns <= 0:
        return
    if not user_text.strip() and not assistant_text.strip():
        return

    store = _memory_store(hass)
    history = store.setdefault(conversation_id, [])
    if user_text.strip():
        last = history[-1] if history else None
        if not (
            last
            and last.get("role") == "user"
            and last.get("content") == user_text.strip()
        ):
            history.append({"role": "user", "content": user_text.strip()})
    if assistant_text.strip():
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text.strip(),
        }
        if turn_meta:
            assistant_entry["turn_meta"] = turn_meta
        history.append(assistant_entry)

    _trim_history(store, conversation_id, history, max_turns=max_turns)

    if entry_id:
        _touch_entry_conversation(hass, entry_id, conversation_id)
        _maybe_persist(hass, entry_id)
        _record_thread_metadata(hass, entry_id, conversation_id, user_text=user_text)


@callback
def _record_thread_metadata(
    hass: HomeAssistant,
    entry_id: str,
    conversation_id: str,
    *,
    user_text: str,
) -> None:
    """Track Assist and console turns in the thread sidebar."""
    from .threads import (
        CONSOLE_CONVERSATION_PREFIX,
        ensure_thread_from_turn,
        schedule_save_threads,
    )

    ensure_thread_from_turn(hass, entry_id, conversation_id, user_text=user_text)
    if not conversation_id.startswith(CONSOLE_CONVERSATION_PREFIX):
        schedule_save_threads(hass, entry_id)


@callback
def patch_last_assistant_turn(
    hass: HomeAssistant,
    conversation_id: str | None,
    *,
    meta_patch: dict[str, Any] | None = None,
    content_suffix: str | None = None,
    entry_id: str | None = None,
) -> bool:
    """Merge metadata and/or append text on the latest assistant message."""
    if not conversation_id:
        return False
    store = _memory_store(hass)
    history = store.get(conversation_id)
    if not history:
        return False
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if entry.get("role") != "assistant":
            continue
        if content_suffix:
            suffix = content_suffix.strip()
            if suffix:
                current = str(entry.get("content") or "").rstrip()
                entry["content"] = f"{current} {suffix}".strip() if current else suffix
        if meta_patch:
            existing = entry.get("turn_meta")
            merged = dict(existing) if isinstance(existing, dict) else {}
            for key, value in meta_patch.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                elif value is not None:
                    merged[key] = value
            entry["turn_meta"] = merged
        history[index] = entry
        if entry_id:
            _touch_entry_conversation(hass, entry_id, conversation_id)
            _maybe_persist(hass, entry_id)
        return True
    return False


@callback
def clear_conversation(
    hass: HomeAssistant,
    conversation_id: str | None,
    entry_id: str | None = None,
) -> None:
    """Clear stored history for a conversation."""
    if not conversation_id:
        return
    store = _memory_store(hass)
    store.pop(conversation_id, None)
    index = _memory_index(hass)
    if entry_id:
        entry_index = index.get(entry_id)
        if entry_index is not None:
            entry_index.pop(conversation_id, None)
        _maybe_persist(hass, entry_id)
        return
    for entry_index in index.values():
        entry_index.pop(conversation_id, None)


def memory_ids_for_entry(hass: HomeAssistant, entry_id: str) -> set[str]:
    """Return conversation ids owned by an entry (for thread listing)."""
    return set(_memory_index(hass).get(entry_id, {}))


def _entry_wants_persist(hass: HomeAssistant, entry_id: str) -> bool:
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None:
        return False
    return bool(entry.data.get(CONF_CONVERSATION_MEMORY_PERSIST, False))


def _maybe_persist(hass: HomeAssistant, entry_id: str) -> None:
    if _entry_wants_persist(hass, entry_id):
        schedule_save_memory(hass, entry_id)


@callback
def schedule_save_memory(hass: HomeAssistant, entry_id: str) -> None:
    """Debounce disk persistence: one pending save task per entry."""
    pending = _pending_memory_saves(hass)
    existing = pending.get(entry_id)
    if existing is not None and not existing.done():
        existing.cancel()

    async def _debounced() -> None:
        try:
            await asyncio.sleep(_SAVE_DEBOUNCE_SECONDS)
            await async_save_memory(hass, entry_id, _from_debounce=True)
        except asyncio.CancelledError:
            raise
        finally:
            if pending.get(entry_id) is asyncio.current_task():
                pending.pop(entry_id, None)

    pending[entry_id] = hass.async_create_task(_debounced())


def _cancel_pending_memory_save(hass: HomeAssistant, entry_id: str) -> None:
    pending = _pending_memory_saves(hass)
    existing = pending.pop(entry_id, None)
    if existing is not None and not existing.done():
        existing.cancel()


def _read_memory_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _write_memory_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def _async_executor_job(
    hass: HomeAssistant,
    func: Callable[..., _T],
    /,
    *args: Any,
) -> _T:
    """Run ``func`` via ``hass.async_add_executor_job``.

    Falls back to an inline call when the hass stub returns a non-awaitable
    (common in unit tests that use ``MagicMock`` without configuring the
    executor).
    """
    result = hass.async_add_executor_job(func, *args)
    if inspect.isawaitable(result):
        return await result
    return func(*args)


async def async_load_memory(hass: HomeAssistant, entry_id: str) -> None:
    """Load persisted conversation memory from disk (executor I/O)."""
    path = _memory_file(hass, entry_id)
    try:
        text = await _async_executor_job(hass, _read_memory_text, path)
    except OSError as err:
        LOGGER.warning("Failed to load HA Agent memory for %s: %s", entry_id, err)
        return
    if text is None:
        return
    try:
        data = json.loads(text)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as err:
        LOGGER.warning("Failed to load HA Agent memory for %s: %s", entry_id, err)
        return

    parsed = _parse_memory_payload(data)
    store = _memory_store(hass)
    entry_index = _memory_index(hass).setdefault(entry_id, {})

    # Drop prior ownership for this entry before applying the file snapshot.
    for stale_id in list(entry_index):
        if stale_id not in parsed:
            entry_index.pop(stale_id, None)
            # Only remove from flat store if no other entry claims it.
            claimed_elsewhere = any(
                stale_id in other and other_entry != entry_id
                for other_entry, other in _memory_index(hass).items()
            )
            if not claimed_elsewhere:
                store.pop(stale_id, None)

    for conversation_id, (messages, updated_at) in parsed.items():
        store[conversation_id] = messages
        entry_index[conversation_id] = updated_at
    _apply_lru(store, entry_index)


async def async_save_memory(
    hass: HomeAssistant,
    entry_id: str,
    *,
    _from_debounce: bool = False,
) -> None:
    """Persist this entry's conversations to disk (executor I/O)."""
    if not _from_debounce:
        _cancel_pending_memory_save(hass, entry_id)

    store = _memory_store(hass)
    entry_index = _memory_index(hass).get(entry_id, {})
    _apply_lru(store, entry_index)
    payload = _build_memory_payload(store, entry_index)
    path = _memory_file(hass, entry_id)
    try:
        text = json.dumps(payload, indent=2, default=str)
    except (TypeError, ValueError) as err:
        LOGGER.warning("Failed to serialize HA Agent memory for %s: %s", entry_id, err)
        return
    try:
        await _async_executor_job(hass, _write_memory_text, path, text)
    except OSError as err:
        LOGGER.warning("Failed to save HA Agent memory for %s: %s", entry_id, err)

"""Unit tests for conversation thread helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _ensure_ha_core() -> None:
    if "homeassistant.core" in sys.modules:
        return
    ha_core = types.ModuleType("homeassistant.core")
    ha_core.HomeAssistant = object

    def callback(func):
        return func

    ha_core.callback = callback
    sys.modules["homeassistant.core"] = ha_core


def _load_module(name: str, *, reload: bool = False):
    mod_name = f"ha_agent.{name}"
    if not reload and mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if reload:
        sys.modules.pop(mod_name, None)

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_threads_modules():
    _ensure_ha_core()
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    _load_module("const", reload=True)
    memory = _load_module("memory", reload=True)

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for skills_mod, rel in (
        ("ha_agent.skills.models", "skills/models.py"),
        ("ha_agent.skills.runtime", "skills/runtime.py"),
    ):
        sys.modules.pop(skills_mod, None)
        models_path = COMPONENT / rel
        spec = importlib.util.spec_from_file_location(skills_mod, models_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[skills_mod] = mod
        spec.loader.exec_module(mod)

    threads = _load_module("threads", reload=True)
    return memory, threads


def _bind_memory(memory, hass, entry_id: str, conversation_id: str, messages) -> None:
    memory._memory_store(hass)[conversation_id] = messages
    memory._touch_entry_conversation(hass, entry_id, conversation_id)


def _async_hass(tmp_path: Path) -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))

    async def _executor(func, *args):
        return func(*args)

    hass.async_add_executor_job = _executor

    def _create_task(coro):
        return asyncio.get_event_loop().create_task(coro)

    hass.async_create_task = _create_task
    return hass


def test_search_threads_matches_title_and_message() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    threads.upsert_thread(hass, entry_id, "conv-a", title="World Cup news")
    threads.upsert_thread(hass, entry_id, "conv-b", title="Shopping list")
    _bind_memory(
        memory,
        hass,
        entry_id,
        "conv-b",
        [
            {"role": "user", "content": "buy milk"},
            {"role": "assistant", "content": "Added milk to your list."},
        ],
    )

    title_hits = threads.search_threads(hass, entry_id, "world")
    assert len(title_hits) == 1
    assert title_hits[0]["conversation_id"] == "conv-a"
    assert title_hits[0]["match_in"] == "title"

    message_hits = threads.search_threads(hass, entry_id, "milk")
    assert len(message_hits) == 1
    assert message_hits[0]["conversation_id"] == "conv-b"
    assert message_hits[0]["match_in"] == "message"
    assert "milk" in message_hits[0]["snippet"].lower()


def test_list_threads_includes_memory_only_assist_chats() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    _bind_memory(
        memory,
        hass,
        entry_id,
        "assist-conv-1",
        [
            {"role": "user", "content": "turn on the patio lights"},
            {"role": "assistant", "content": "Done."},
        ],
    )

    listed = threads.list_threads(hass, entry_id)
    assert len(listed) == 1
    assert listed[0]["conversation_id"] == "assist-conv-1"
    assert listed[0]["source"] == "assist"
    assert listed[0]["title"] == "turn on the patio lights"


def test_list_threads_filters_by_source() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    threads.upsert_thread(
        hass,
        entry_id,
        "console-abc",
        title="Panel chat",
        source="console",
    )
    _bind_memory(
        memory,
        hass,
        entry_id,
        "assist-xyz",
        [
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "content": "Sunny."},
        ],
    )

    assist_only = threads.list_threads(hass, entry_id, source="assist")
    assert len(assist_only) == 1
    assert assist_only[0]["conversation_id"] == "assist-xyz"

    console_only = threads.list_threads(hass, entry_id, source="console")
    assert len(console_only) == 1
    assert console_only[0]["conversation_id"] == "console-abc"


def test_list_threads_is_entry_scoped() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"

    _bind_memory(
        memory,
        hass,
        "entry-a",
        "conv-a",
        [{"role": "user", "content": "alpha"}],
    )
    _bind_memory(
        memory,
        hass,
        "entry-b",
        "conv-b",
        [{"role": "user", "content": "beta"}],
    )

    listed = threads.list_threads(hass, "entry-a")
    assert [item["conversation_id"] for item in listed] == ["conv-a"]


def test_conversation_source() -> None:
    _, threads = _load_threads_modules()
    assert threads.conversation_source("console-123") == "console"
    assert threads.conversation_source("ha-assist-uuid") == "assist"


@pytest.mark.asyncio
async def test_async_delete_thread_removes_metadata_and_memory() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    threads.upsert_thread(hass, entry_id, "conv-a", title="Old chat")
    _bind_memory(
        memory,
        hass,
        entry_id,
        "conv-a",
        [{"role": "user", "content": "hello"}],
    )

    with (
        patch.object(threads, "async_save_threads", new=AsyncMock()),
        patch.object(threads, "_entry_wants_persist", return_value=False),
        patch.object(memory, "_entry_wants_persist", return_value=False),
    ):
        deleted = await threads.async_delete_thread(hass, entry_id, "conv-a")

    assert deleted is True
    assert "conv-a" not in threads.get_threads(hass, entry_id)
    assert "conv-a" not in memory._memory_store(hass)


@pytest.mark.asyncio
async def test_async_delete_thread_returns_false_when_missing() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    with (
        patch.object(threads, "_entry_wants_persist", return_value=False),
        patch.object(memory, "_entry_wants_persist", return_value=False),
    ):
        deleted = await threads.async_delete_thread(hass, "entry-1", "missing")
    assert deleted is False


@pytest.mark.asyncio
async def test_async_save_threads_uses_executor(tmp_path: Path) -> None:
    _, threads = _load_threads_modules()
    hass = _async_hass(tmp_path)
    entry_id = "entry-threads"
    threads.upsert_thread(hass, entry_id, "conv-1", title="Hello")

    executor_calls = 0
    original = hass.async_add_executor_job

    async def tracking_executor(func, *args):
        nonlocal executor_calls
        executor_calls += 1
        return await original(func, *args)

    hass.async_add_executor_job = tracking_executor
    await threads.async_save_threads(hass, entry_id)
    assert executor_calls >= 1
    saved = json.loads(
        (tmp_path / ".storage" / f"ha_agent_threads_{entry_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "conv-1" in saved
    assert saved["conv-1"]["title"] == "Hello"


@pytest.mark.asyncio
async def test_schedule_save_threads_debounces(tmp_path: Path) -> None:
    _, threads = _load_threads_modules()
    hass = _async_hass(tmp_path)
    entry_id = "entry-debounce"
    threads.upsert_thread(hass, entry_id, "conv-1", title="One")

    save_calls = 0
    original = threads.async_save_threads

    async def counting_save(hass_arg, entry, *, _from_debounce: bool = False):
        nonlocal save_calls
        save_calls += 1
        await original(hass_arg, entry, _from_debounce=_from_debounce)

    threads.async_save_threads = counting_save  # type: ignore[method-assign]
    threads.schedule_save_threads(hass, entry_id)
    threads.schedule_save_threads(hass, entry_id)
    threads.schedule_save_threads(hass, entry_id)
    await asyncio.sleep(0.15)
    assert save_calls == 1

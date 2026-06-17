"""Unit tests for conversation thread helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_core() -> None:
    if "homeassistant.core" in sys.modules:
        return
    ha_core = types.ModuleType("homeassistant.core")
    ha_core.HomeAssistant = object

    def callback(func):
        return func

    ha_core.callback = callback
    sys.modules["homeassistant.core"] = ha_core


def _load_module(name: str):
    mod_name = f"ha_agent.{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

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

    _load_module("const")
    memory = _load_module("memory")

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    models_path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location(
        "ha_agent.skills.models", models_path
    )
    assert spec is not None and spec.loader is not None
    models = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    runtime_path = COMPONENT / "skills" / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        "ha_agent.skills.runtime", runtime_path
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.skills.runtime"] = runtime
    spec.loader.exec_module(runtime)

    threads = _load_module("threads")
    return memory, threads


def test_search_threads_matches_title_and_message() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    threads.upsert_thread(hass, entry_id, "conv-a", title="World Cup news")
    threads.upsert_thread(hass, entry_id, "conv-b", title="Shopping list")
    memory._memory_store(hass)["conv-b"] = [
        {"role": "user", "content": "buy milk"},
        {"role": "assistant", "content": "Added milk to your list."},
    ]

    title_hits = threads.search_threads(hass, entry_id, "world")
    assert len(title_hits) == 1
    assert title_hits[0]["conversation_id"] == "conv-a"
    assert title_hits[0]["match_in"] == "title"

    message_hits = threads.search_threads(hass, entry_id, "milk")
    assert len(message_hits) == 1
    assert message_hits[0]["conversation_id"] == "conv-b"
    assert message_hits[0]["match_in"] == "message"
    assert "milk" in message_hits[0]["snippet"].lower()


@pytest.mark.asyncio
async def test_async_delete_thread_removes_metadata_and_memory() -> None:
    memory, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    hass.config.path.return_value = "/config"
    entry_id = "entry-1"

    threads.upsert_thread(hass, entry_id, "conv-a", title="Old chat")
    memory._memory_store(hass)["conv-a"] = [{"role": "user", "content": "hello"}]

    with (
        patch.object(threads, "async_save_threads", new=AsyncMock()),
        patch.object(memory, "_entry_wants_persist", return_value=False),
    ):
        deleted = await threads.async_delete_thread(hass, entry_id, "conv-a")

    assert deleted is True
    assert "conv-a" not in threads.get_threads(hass, entry_id)
    assert "conv-a" not in memory._memory_store(hass)


@pytest.mark.asyncio
async def test_async_delete_thread_returns_false_when_missing() -> None:
    _, threads = _load_threads_modules()
    hass = MagicMock()
    hass.data = {}
    deleted = await threads.async_delete_thread(hass, "entry-1", "missing")
    assert deleted is False

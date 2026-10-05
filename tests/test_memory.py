"""Unit tests for conversation memory helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_memory():
    if "homeassistant.core" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_core = types.ModuleType("homeassistant.core")

        def callback(func):
            return func

        ha_core.callback = callback
        ha_core.HomeAssistant = object
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.core"] = ha_core

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    # Always reload from disk so edits are picked up under a shared process.
    for mod_name in ("ha_agent.const", "ha_agent.memory"):
        sys.modules.pop(mod_name, None)

    path = COMPONENT / "const.py"
    spec = importlib.util.spec_from_file_location("ha_agent.const", path)
    const = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.const"] = const
    spec.loader.exec_module(const)

    path = COMPONENT / "memory.py"
    spec = importlib.util.spec_from_file_location("ha_agent.memory", path)
    memory = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.memory"] = memory
    spec.loader.exec_module(memory)
    return memory


memory = _load_memory()


def _hass() -> SimpleNamespace:
    return SimpleNamespace(data={})


def _async_hass(tmp_path: Path) -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
    # Persist off so append_* with entry_id does not schedule background saves.
    hass.config_entries.async_get_entry.return_value = SimpleNamespace(data={})

    async def _executor(func, *args):
        return func(*args)

    hass.async_add_executor_job = _executor

    def _create_task(coro):
        return asyncio.get_event_loop().create_task(coro)

    hass.async_create_task = _create_task
    return hass


def test_conversation_history_for_turn_excludes_inflight_user() -> None:
    """Agent prompts should not treat the current user line as prior context."""
    hass = _hass()
    conversation_id = "console-test"
    memory.append_user_message(
        hass,
        conversation_id,
        "what are todays news",
        max_turns=10,
    )
    prior = memory.conversation_history_for_turn(
        hass,
        conversation_id,
        "what are todays news",
        max_turns=10,
    )
    assert prior == []


def test_conversation_history_for_turn_keeps_completed_turns() -> None:
    """Earlier completed turns remain available to the agent."""
    hass = _hass()
    conversation_id = "console-test-2"
    memory.append_turn(
        hass,
        conversation_id,
        "turn on dining lights",
        "Done.",
        max_turns=10,
    )
    memory.append_user_message(
        hass,
        conversation_id,
        "turn them off",
        max_turns=10,
    )
    prior = memory.conversation_history_for_turn(
        hass,
        conversation_id,
        "turn them off",
        max_turns=10,
    )
    assert len(prior) == 2
    assert prior[0]["role"] == "user"
    assert prior[1]["role"] == "assistant"


def test_patch_last_assistant_turn_merges_meta_and_suffix() -> None:
    """Post-turn hooks can patch the latest assistant message in memory."""
    hass = _hass()
    conversation_id = "patch-test"
    memory.append_turn(
        hass,
        conversation_id,
        "check email",
        "You have mail.",
        max_turns=10,
        turn_meta={"route": "email"},
    )
    patched = memory.patch_last_assistant_turn(
        hass,
        conversation_id,
        meta_patch={
            "skill_update": {
                "title": "Email Management",
                "from_version": 1,
                "to_version": 2,
                "reason": "added mailbox",
            },
            "skill_repair": {"issue_kind": "missing_param", "fields": ["mailbox"]},
        },
        content_suffix=" Updated skill: Email Management (v1→v2).",
    )
    assert patched is True
    history = memory.get_history(hass, conversation_id, max_turns=10)
    assistant = history[-1]
    assert "Updated skill" in assistant["content"]
    assert assistant["turn_meta"]["skill_update"]["to_version"] == 2
    assert assistant["turn_meta"]["skill_repair"]["issue_kind"] == "missing_param"
    assert assistant["turn_meta"]["route"] == "email"


def test_normalize_messages_requires_role_and_content() -> None:
    assert memory._normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            "nope",
            {"content": "missing role"},
            {"role": "assistant", "content": "ok", "turn_meta": {"x": 1}},
        ]
    ) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "turn_meta": {"x": 1}},
    ]


def test_build_memory_payload_filters_by_entry_index() -> None:
    store = {
        "conv-a": [{"role": "user", "content": "a"}],
        "conv-b": [{"role": "user", "content": "b"}],
    }
    entry_index = {"conv-a": 100.0}
    payload = memory._build_memory_payload(store, entry_index)
    assert set(payload["conversations"]) == {"conv-a"}
    assert payload["conversations"]["conv-a"]["messages"][0]["content"] == "a"
    assert payload["conversations"]["conv-a"]["updated_at"] == 100.0


def test_apply_lru_evicts_oldest() -> None:
    store = {f"c{i}": [{"role": "user", "content": str(i)}] for i in range(5)}
    entry_index = {f"c{i}": float(i) for i in range(5)}
    evicted = memory._apply_lru(store, entry_index, max_conversations=3)
    assert evicted == ["c0", "c1"]
    assert set(entry_index) == {"c2", "c3", "c4"}
    assert "c0" not in store
    assert "c1" not in store


def test_parse_memory_payload_legacy_and_versioned() -> None:
    legacy = memory._parse_memory_payload(
        {"old": [{"role": "user", "content": "x"}, "bad"]}
    )
    assert "old" in legacy
    assert legacy["old"][0] == [{"role": "user", "content": "x"}]

    versioned = memory._parse_memory_payload(
        {
            "version": 1,
            "conversations": {
                "new": {
                    "messages": [{"role": "assistant", "content": "y"}],
                    "updated_at": 42.5,
                }
            },
        }
    )
    assert versioned["new"][0] == [{"role": "assistant", "content": "y"}]
    assert versioned["new"][1] == 42.5


def test_clear_conversation_accepts_entry_id() -> None:
    hass = SimpleNamespace(
        data={},
        config_entries=SimpleNamespace(async_get_entry=lambda _id: None),
    )
    memory._memory_store(hass)["conv-1"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    memory._touch_entry_conversation(hass, "entry-a", "conv-1")
    assert "conv-1" in memory._memory_index(hass)["entry-a"]
    memory.clear_conversation(hass, "conv-1", entry_id="entry-a")
    assert "conv-1" not in memory._memory_store(hass)
    assert "conv-1" not in memory._memory_index(hass)["entry-a"]


@pytest.mark.asyncio
async def test_async_save_memory_uses_executor_not_event_loop_path(
    tmp_path: Path,
) -> None:
    """Path.write_text must run inside async_add_executor_job, not on the loop."""
    hass = _async_hass(tmp_path)
    entry_id = "entry-save"
    # console- prefix skips assist thread autosave side effects.
    memory.append_turn(
        hass,
        "console-save",
        "hello",
        "world",
        max_turns=5,
        entry_id=entry_id,
    )

    write_calls: list[str] = []
    original_write = memory._write_memory_text

    def tracking_write(path: Path, text: str) -> None:
        write_calls.append("executor")
        original_write(path, text)

    executor_calls = 0
    original_executor = hass.async_add_executor_job

    async def tracking_executor(func, *args):
        nonlocal executor_calls
        executor_calls += 1
        name = getattr(func, "__name__", "")
        if func is tracking_write or name == "_write_memory_text":
            write_calls.append("via_executor")
        return await original_executor(func, *args)

    hass.async_add_executor_job = tracking_executor
    memory._write_memory_text = tracking_write  # type: ignore[method-assign]

    try:
        await memory.async_save_memory(hass, entry_id)
    finally:
        memory._write_memory_text = original_write  # type: ignore[method-assign]

    assert executor_calls >= 1
    assert "via_executor" in write_calls or "executor" in write_calls
    saved = json.loads(
        (tmp_path / ".storage" / f"ha_agent_memory_{entry_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "console-save" in saved["conversations"]
    assert "conv-other" not in saved["conversations"]


@pytest.mark.asyncio
async def test_async_save_memory_is_entry_scoped(tmp_path: Path) -> None:
    hass = _async_hass(tmp_path)
    memory.append_turn(hass, "console-a", "a", "A", max_turns=5, entry_id="entry-1")
    memory.append_turn(hass, "console-b", "b", "B", max_turns=5, entry_id="entry-2")
    await memory.async_save_memory(hass, "entry-1")
    payload = json.loads(
        (tmp_path / ".storage" / "ha_agent_memory_entry-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload["conversations"]) == {"console-a"}


@pytest.mark.asyncio
async def test_async_load_memory_validates_shape(tmp_path: Path) -> None:
    hass = _async_hass(tmp_path)
    entry_id = "entry-load"
    path = tmp_path / ".storage" / f"ha_agent_memory_{entry_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "good": [{"role": "user", "content": "hi"}, {"role": "nope"}],
                "empty": [],
                "bad": "string",
            }
        ),
        encoding="utf-8",
    )
    await memory.async_load_memory(hass, entry_id)
    store = memory._memory_store(hass)
    assert store["good"] == [{"role": "user", "content": "hi"}]
    assert "empty" not in store
    assert "bad" not in store

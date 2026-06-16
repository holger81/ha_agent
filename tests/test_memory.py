"""Unit tests for per-conversation memory."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_memory_module():
    module_name = "ha_agent.memory"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    ha_core = types.ModuleType("homeassistant.core")

    def callback(func):
        return func

    ha_core.HomeAssistant = object
    ha_core.callback = callback
    sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
    sys.modules["homeassistant.core"] = ha_core

    const_path = COMPONENT / "const.py"
    const_spec = importlib.util.spec_from_file_location("ha_agent.const", const_path)
    assert const_spec is not None and const_spec.loader is not None
    const_mod = importlib.util.module_from_spec(const_spec)
    sys.modules["ha_agent.const"] = const_mod
    const_spec.loader.exec_module(const_mod)

    path = COMPONENT / "memory.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


memory = _load_memory_module()


def test_append_turn_and_get_history() -> None:
    """History stores user and assistant messages for a conversation."""
    hass = types.SimpleNamespace(data={})
    memory.append_turn(
        hass,
        "conv-1",
        "turn off the lights",
        "Done.",
        max_turns=5,
    )

    history = memory.get_history(hass, "conv-1", max_turns=5)

    assert history == [
        {"role": "user", "content": "turn off the lights"},
        {"role": "assistant", "content": "Done."},
    ]


def test_history_truncates_to_max_turns() -> None:
    """Only the most recent turns are kept."""
    hass = types.SimpleNamespace(data={})
    for index in range(4):
        memory.append_turn(
            hass,
            "conv-2",
            f"user-{index}",
            f"assistant-{index}",
            max_turns=2,
        )

    history = memory.get_history(hass, "conv-2", max_turns=2)

    assert history == [
        {"role": "user", "content": "user-2"},
        {"role": "assistant", "content": "assistant-2"},
        {"role": "user", "content": "user-3"},
        {"role": "assistant", "content": "assistant-3"},
    ]


def test_clear_conversation_removes_history() -> None:
    """Clearing a conversation drops stored turns."""
    hass = types.SimpleNamespace(data={})
    memory.append_turn(hass, "conv-3", "hello", "hi", max_turns=3)
    memory.clear_conversation(hass, "conv-3")

    assert memory.get_history(hass, "conv-3", max_turns=3) == []


def test_append_user_message_dedupes_same_user_text() -> None:
    """User message is stored immediately and not duplicated."""
    hass = types.SimpleNamespace(data={})
    memory.append_user_message(hass, "conv-4", "hello", max_turns=3)
    memory.append_user_message(hass, "conv-4", "hello", max_turns=3)
    memory.append_turn(hass, "conv-4", "hello", "hi there", max_turns=3)

    assert memory.get_history(hass, "conv-4", max_turns=3) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

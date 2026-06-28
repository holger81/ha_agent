"""Unit tests for conversation memory helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


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

"""Unit tests for reasoning / thinking level helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_thinking():
    if "ha_agent.thinking" in sys.modules:
        return sys.modules["ha_agent.thinking"]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    path = COMPONENT / "thinking.py"
    spec = importlib.util.spec_from_file_location("ha_agent.thinking", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.thinking"] = module
    spec.loader.exec_module(module)
    return module


thinking = _load_thinking()


def test_normalize_thinking_level_from_legacy_bool() -> None:
    assert thinking.normalize_thinking_level(True) == "medium"
    assert thinking.normalize_thinking_level(False) == "off"


def test_apply_thinking_off() -> None:
    payload: dict = {}
    thinking.apply_thinking_to_payload(payload, "off")
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_apply_thinking_medium() -> None:
    payload: dict = {}
    thinking.apply_thinking_to_payload(payload, "medium")
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert payload["reasoning_format"] == "deepseek"


def test_apply_thinking_infinite() -> None:
    payload: dict = {}
    thinking.apply_thinking_to_payload(payload, "infinite")
    assert payload["reasoning_effort"] == "high"
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "reasoning_budget": -1,
    }

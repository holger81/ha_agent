"""Tests for in-turn message compaction."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compaction = _load("compaction")


def test_compact_summarizes_older_tool_results() -> None:
    messages = [
        {"role": "user", "content": "check mail and news"},
        {"role": "tool", "content": "x" * 400},
        {"role": "tool", "content": "y" * 400},
        {"role": "tool", "content": "latest result"},
    ]
    changed = compaction.compact_messages_if_needed(
        messages,
        token_budget=80,
        keep_recent_tool_results=1,
    )
    assert changed is True
    assert messages[1]["content"].startswith("[Earlier tool result summarized]")
    assert messages[3]["content"] == "latest result"

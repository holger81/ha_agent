"""Unit tests for skill discovery helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_discovery():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", path)
    models = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "skills" / "format.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.format", path)
    fmt = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.format"] = fmt
    spec.loader.exec_module(fmt)

    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistant:
        pass

    def callback(func):
        return func

    ha_core.HomeAssistant = HomeAssistant
    ha_core.callback = callback
    sys.modules["homeassistant"] = types.ModuleType("homeassistant")
    sys.modules["homeassistant.core"] = ha_core

    path = COMPONENT / "skills" / "discovery.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.discovery", path)
    discovery = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.discovery"] = discovery
    spec.loader.exec_module(discovery)
    return discovery


discovery_mod = _load_discovery()
build_discovery_query = discovery_mod.build_discovery_query


def test_build_discovery_query_includes_recent_history() -> None:
    query = build_discovery_query(
        "let me read the target email",
        [
            {"role": "user", "content": "check my inbox"},
            {
                "role": "assistant",
                "content": "Target Circle Mastercard reminder about payment.",
            },
        ],
    )

    assert "let me read the target email" in query
    assert "Target Circle" in query
    assert "check my inbox" in query


def test_build_discovery_query_uses_user_text_when_no_history() -> None:
    assert build_discovery_query("read unread mail", None) == "read unread mail"

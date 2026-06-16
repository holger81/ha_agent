"""Unit tests for activity log."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_activity_module():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.api" not in sys.modules:
        api_pkg = types.ModuleType("ha_agent.api")
        api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.api"] = api_pkg

    for name in ("const",):
        mod_name = f"ha_agent.{name}"
        if mod_name not in sys.modules:
            path = COMPONENT / f"{name}.py"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

    if "ha_agent.skills.models" not in sys.modules:
        models_path = COMPONENT / "skills" / "models.py"
        spec = importlib.util.spec_from_file_location(
            "ha_agent.skills.models", models_path
        )
        assert spec is not None and spec.loader is not None
        models = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.skills.models"] = models
        spec.loader.exec_module(models)

    if "ha_agent.api.serialize" not in sys.modules:
        serialize_path = COMPONENT / "api" / "serialize.py"
        spec = importlib.util.spec_from_file_location(
            "ha_agent.api.serialize", serialize_path
        )
        assert spec is not None and spec.loader is not None
        ser = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.api.serialize"] = ser
        spec.loader.exec_module(ser)

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object

        def callback(func):
            return func

        ha_core.callback = callback
        sys.modules["homeassistant.core"] = ha_core

    mod_name = "ha_agent.activity"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = COMPONENT / "activity.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_record_and_list_turns() -> None:
    activity = _load_activity_module()
    TurnTrace = sys.modules["ha_agent.skills.models"].TurnTrace

    hass = MagicMock()
    hass.data = {}
    trace = TurnTrace(user_text="hi", history_len=0, assistant_text="hello")
    activity.record_turn(hass, "entry-1", trace)
    turns, total = activity.list_turns(hass, "entry-1")
    assert total == 1
    assert turns[0]["user_text"] == "hi"
    assert turns[0]["assistant_text"] == "hello"

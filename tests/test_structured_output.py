"""Tests for structured JSON schema helpers."""

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


structured_output = _load("structured_output")


def test_json_schema_format_wraps_schema() -> None:
    payload = structured_output.json_schema_format(
        "route",
        structured_output.ROUTE_SCHEMA,
    )
    assert payload["type"] == "json_schema"
    assert payload["json_schema"]["name"] == "route"
    assert payload["json_schema"]["strict"] is True
    assert payload["json_schema"]["schema"]["required"] == ["route"]

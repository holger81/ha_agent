"""Unit tests for verifier parsing and serialization."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    deps = {
        "verifier": [
            "config_helpers",
            "const",
            "llm_client",
            "llm_telemetry",
            "skills.models",
            "structured_output",
        ],
        "skills.models": [],
        "config_helpers": ["const"],
        "llm_client": ["const", "config_helpers"],
        "llm_telemetry": [],
        "structured_output": [],
        "const": [],
    }
    if name in {"llm_client", "verifier"}:
        if "homeassistant" not in sys.modules:
            sys.modules["homeassistant"] = types.ModuleType("homeassistant")
        if "homeassistant.core" not in sys.modules:
            core = types.ModuleType("homeassistant.core")

            class HomeAssistant:
                pass

            core.HomeAssistant = HomeAssistant
            sys.modules["homeassistant.core"] = core

    if name.startswith("skills.") and "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    path = COMPONENT / f"{name.replace('.', '/')}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_verifier_response_coerces_string_false() -> None:
    """String \"false\" must not be treated as a passing verdict."""
    verifier = _load("verifier")
    parsed = verifier.parse_verifier_response(
        '{"pass": "false", "reason": "tools failed", "skill_followed": "false"}'
    )
    assert parsed is not None
    assert parsed.passed is False
    assert parsed.skill_followed is False
    assert parsed.reason == "tools failed"


def test_parse_verifier_response_accepts_bool_true() -> None:
    verifier = _load("verifier")
    parsed = verifier.parse_verifier_response(
        '{"pass": true, "reason": "ok", "skill_followed": true}'
    )
    assert parsed is not None
    assert parsed.passed is True


def test_serialize_tool_call_caps_error_and_arguments() -> None:
    verifier = _load("verifier")
    serialized = verifier._serialize_tool_call(
        {
            "toolName": "callTool",
            "arguments": {"body": "x" * 5_000},
            "error": "e" * 5_000,
            "succeeded": False,
        }
    )
    assert len(serialized["arguments"]["body"]) < 3_000
    assert "…[truncated]" in serialized["arguments"]["body"]
    assert len(serialized["error"]) < 3_000

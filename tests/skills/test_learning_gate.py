"""Unit tests for skill learning gate."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)

_MODULE_DEPS: dict[str, list[str]] = {
    "const": [],
    "config_helpers": ["const"],
    "llm_client": ["const", "config_helpers"],
    "skills.models": [],
    "skills.learning_gate": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
    ],
}


def _ensure_ha_stubs() -> None:
    if "homeassistant.core" in sys.modules:
        return

    ha_pkg = types.ModuleType("homeassistant")
    ha_exc = types.ModuleType("homeassistant.exceptions")
    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistantError(Exception):
        pass

    class HomeAssistant:
        pass

    ha_exc.HomeAssistantError = HomeAssistantError
    ha_core.HomeAssistant = HomeAssistant
    sys.modules["homeassistant"] = ha_pkg
    sys.modules["homeassistant.exceptions"] = ha_exc
    sys.modules["homeassistant.core"] = ha_core


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if name.startswith("skills."):
        skill_name = name.split(".", 1)[1]
        if "ha_agent.skills" not in sys.modules:
            skills_pkg = types.ModuleType("ha_agent.skills")
            skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
            sys.modules["ha_agent.skills"] = skills_pkg
        path = COMPONENT / "skills" / f"{skill_name}.py"
    else:
        path = COMPONENT / f"{name}.py"

    _ensure_ha_stubs()
    for dep in _MODULE_DEPS.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load_module(dep)

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


models_mod = _load_module("skills.models")
gate_mod = _load_module("skills.learning_gate")
TurnTrace = models_mod.TurnTrace
_parse_learn_gate_response = gate_mod._parse_learn_gate_response
assess_skill_worth_learning = gate_mod.assess_skill_worth_learning


def test_parse_learn_gate_response_true() -> None:
    assert _parse_learn_gate_response('{"learn": true, "reason": "workflow"}') is True


def test_parse_learn_gate_response_false() -> None:
    assert _parse_learn_gate_response('{"learn": false, "reason": "one-off"}') is False


def test_parse_learn_gate_response_invalid() -> None:
    assert _parse_learn_gate_response("not json") is None
    assert _parse_learn_gate_response('{"learn": "yes"}') is None


@pytest.mark.asyncio
async def test_assess_skill_worth_learning_approves() -> None:
    trace = TurnTrace(
        user_text="turn off dining lights every evening",
        history_len=0,
        tool_calls=[
            {"toolName": "home_assistant__ha_call_service"},
            {"toolName": "home_assistant__ha_get_state"},
        ],
        assistant_text="Dining lights are off.",
        iterations=2,
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=MagicMock(
            content=json.dumps({"learn": True, "reason": "repeatable workflow"}),
        ),
    )
    backend = MagicMock()

    assert await assess_skill_worth_learning(
        llm,
        backend,
        trace=trace,
        history=[],
    )


@pytest.mark.asyncio
async def test_assess_skill_worth_learning_rejects() -> None:
    trace = TurnTrace(
        user_text="what are todays news",
        history_len=2,
        tool_calls=[{"toolName": "mcp_news__news_curate"}],
        assistant_text="Here are today's headlines.",
        iterations=1,
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=MagicMock(
            content=json.dumps({"learn": False, "reason": "one-off news summary"}),
        ),
    )
    backend = MagicMock()

    assert not await assess_skill_worth_learning(
        llm,
        backend,
        trace=trace,
        history=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.asyncio
async def test_assess_skill_worth_learning_fails_closed() -> None:
    trace = TurnTrace(
        user_text="do thing",
        history_len=0,
        tool_calls=[{"toolName": "a"}, {"toolName": "b"}],
        assistant_text="Done.",
        iterations=2,
    )
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("offline"))
    backend = MagicMock()

    assert not await assess_skill_worth_learning(
        llm,
        backend,
        trace=trace,
        history=[],
    )

"""Tests for skill creation from observer drafts."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)

_MODULE_DEPS: dict[str, list[str]] = {
    "const": [],
    "config_helpers": ["const"],
    "embedded_tools": [],
    "llm_client": ["const", "config_helpers", "embedded_tools"],
    "skills.models": [],
    "skills.observer": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
    ],
    "skills.store": ["const", "skills.models"],
    "skills.creator": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
        "skills.store",
        "skills.observer",
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

    def callback(func):
        return func

    ha_core.HomeAssistant = HomeAssistant
    ha_core.callback = callback
    ha_exc.HomeAssistantError = HomeAssistantError
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


def _turn_trace(**kwargs):
    models = _load_module("skills.models")
    return models.TurnTrace(**kwargs)


@pytest.mark.asyncio
async def test_create_skill_from_trace_rejects_when_observer_declines() -> None:
    creator = _load_module("skills.creator")
    observer = _load_module("skills.observer")

    trace = _turn_trace(
        user_text="today's news",
        history_len=0,
        tool_calls=[{"toolName": "mcp_news__news_curate"}],
        assistant_text="Headlines.",
    )

    with patch.object(
        creator,
        "observe_skill_candidate",
        AsyncMock(
            return_value=observer.SkillObserverResult(
                learn=False,
                reason="one-off",
            )
        ),
    ):
        result = await creator.create_skill_from_trace(
            MagicMock(),
            "entry",
            MagicMock(),
            MagicMock(),
            trace=trace,
            history=[],
        )

    assert result is None


@pytest.mark.asyncio
async def test_create_skill_from_trace_uses_provided_draft() -> None:
    creator = _load_module("skills.creator")
    models = _load_module("skills.models")
    SkillDraft = models.SkillDraft

    draft = SkillDraft(
        title="Evening lights",
        description="Turns off dining lights.",
        triggers=["turn off dining lights"],
        body="Call ha_call_service.",
        tool_steps=[],
    )
    fake_skill = MagicMock(title="Evening lights")

    class FakeStore:
        def find_duplicate(self, _triggers):
            return None

        def insert_skill(self, **_kwargs):
            return fake_skill

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )

    with patch.object(creator, "get_skill_store", lambda _h, _e: FakeStore()):
        result = await creator.create_skill_from_trace(
            hass,
            "entry",
            MagicMock(),
            MagicMock(),
            trace=_turn_trace(user_text="x", history_len=0),
            history=[],
            draft=draft,
        )

    assert result is fake_skill

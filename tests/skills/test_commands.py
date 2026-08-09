"""Unit tests for skill chat commands."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_commands():
    if "ha_agent.skills.commands" in sys.modules:
        return sys.modules["ha_agent.skills.commands"]

    package = types.ModuleType("ha_agent")
    package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
    sys.modules["ha_agent"] = package

    skills_pkg = types.ModuleType("ha_agent.skills")
    skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.skills"] = skills_pkg

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object

        class ServiceCall:
            def __init__(self, data: dict | None = None) -> None:
                self.data = data or {}

        class SupportsResponse:
            ONLY = "only"

        def callback(func):
            return func

        ha_core.ServiceCall = ServiceCall
        ha_core.SupportsResponse = SupportsResponse
        ha_core.callback = callback
        sys.modules["homeassistant.core"] = ha_core

    for name in ("const", "context"):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    for rel in ("models.py", "store.py", "creator.py", "runtime.py"):
        mod_name = f"ha_agent.skills.{rel[:-3]}"
        path = COMPONENT / "skills" / rel
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    path = COMPONENT / "skills" / "commands.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.commands", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.commands"] = module
    spec.loader.exec_module(module)
    return module


commands = _load_commands()


@pytest.mark.asyncio
async def test_list_skills_empty() -> None:
    """List command reports when no skills exist."""
    hass = MagicMock()
    store = MagicMock()
    store.list_recent.return_value = []
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))

    with patch.object(commands, "get_skill_store", return_value=store):
        reply = await commands.try_handle_skill_command(
            hass, "entry", "list my skills"
        )
    assert reply is not None
    assert "no saved skills" in reply.lower()


def test_is_skill_admin_query() -> None:
    """Admin phrases are detected."""
    assert commands.is_skill_admin_query("disable the dining room lights skill")
    assert commands.is_skill_admin_query("list my skills")
    assert commands.is_skill_admin_query("save this as a skill")
    assert not commands.is_skill_admin_query("turn on the lights")


@pytest.mark.asyncio
async def test_try_handle_manual_skill_save_uses_prior_turn() -> None:
    """Manual save uses the prior workflow turn, not the empty save request."""
    prior_trace = MagicMock()
    prior_trace.user_text = "what is the temperature in Jonathans room?"
    skill = MagicMock()
    skill.title = "Jonathan room temperature"

    with (
        patch.object(
            commands,
            "find_prior_workflow_turn",
            return_value={"user_text": prior_trace.user_text},
        ),
        patch.object(
            commands,
            "activity_turn_to_trace",
            return_value=prior_trace,
        ),
        patch.object(
            commands,
            "should_offer_skill_creation",
            return_value=True,
        ),
        patch.object(
            commands,
            "create_skill_from_trace",
            new=AsyncMock(return_value=skill),
        ) as create,
    ):
        reply = await commands.try_handle_manual_skill_save(
            MagicMock(),
            "entry",
            "c1",
            "save this as a skill",
            llm=MagicMock(),
            backend=MagicMock(),
            history=[],
        )

    assert reply == "Saved skill: Jonathan room temperature."
    create.assert_awaited_once()
    assert create.await_args.kwargs["trace"] is prior_trace
    assert create.await_args.kwargs["manual_save"] is True

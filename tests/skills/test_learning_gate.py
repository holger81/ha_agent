"""Unit tests for skill learning observer."""

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
    "skills.observer": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
    ],
    "skills.learning_gate": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
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
observer_mod = _load_module("skills.observer")
gate_mod = _load_module("skills.learning_gate")
TurnTrace = models_mod.TurnTrace
parse_observer_response = observer_mod.parse_observer_response
observe_skill_candidate = observer_mod.observe_skill_candidate
observe_skill_override = observer_mod.observe_skill_override
build_observer_payload = observer_mod.build_observer_payload
assess_skill_worth_learning = gate_mod.assess_skill_worth_learning


def test_parse_observer_response_rejects() -> None:
    parsed = parse_observer_response(
        json.dumps({"learn": False, "reason": "one-off news summary"})
    )
    assert parsed is not None
    assert parsed.learn is False
    assert parsed.draft is None


def test_parse_observer_response_accepts_with_draft() -> None:
    parsed = parse_observer_response(
        json.dumps(
            {
                "learn": True,
                "reason": "repeatable workflow",
                "title": "Evening lights",
                "description": "Turns off dining lights in the evening.",
                "triggers": ["turn off dining lights"],
                "body": "1. Call ha_call_service for dining lights.",
                "tool_steps": [
                    {
                        "toolName": "home_assistant__ha_call_service",
                        "arguments": {"domain": "light"},
                    }
                ],
            }
        )
    )
    assert parsed is not None
    assert parsed.learn is True
    assert parsed.draft is not None
    assert parsed.draft.title == "Evening lights"


def test_parse_observer_response_reads_update_parent() -> None:
    parsed = parse_observer_response(
        json.dumps(
            {
                "learn": True,
                "reason": "extend parent",
                "title": "Email workflows",
                "description": "Check and mark email.",
                "triggers": ["mark emails read"],
                "body": "Use mark-read tool.",
                "tool_steps": [],
                "update_parent": True,
            }
        )
    )
    assert parsed is not None
    assert parsed.update_parent is True


def test_build_observer_payload_includes_override_fields() -> None:
    trace = TurnTrace(
        user_text="mark all emails read",
        history_len=0,
        route="email",
        skill_plan_override=True,
        skill_plan_override_reason="Skill only covers unread checks.",
        tool_calls=[],
    )
    payload = build_observer_payload(trace, [])
    assert payload["skill_plan_override"] is True
    assert "unread checks" in payload["skill_plan_override_reason"]


@pytest.mark.asyncio
async def test_observe_skill_override_updates_parent() -> None:
    Skill = models_mod.Skill
    parent = Skill(
        id="parent-1",
        slug="check-unread",
        title="Check unread email",
        description="Unread count",
        triggers=["unread email"],
        body="Search unread.",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages", "arguments": {}}],
        route_scope="email",
    )
    trace = TurnTrace(
        user_text="mark all above emails read",
        history_len=0,
        route="email",
        skill_plan_override=True,
        skill_plan_override_reason="No mark-read step in parent skill.",
        tool_calls=[
            {"toolName": "mail_mcp__imap_mark_read", "succeeded": True},
        ],
        assistant_text="Marked 3 messages as read.",
        iterations=4,
        outcome="success",
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "learn": True,
                    "reason": "extend email skill",
                    "title": "Email read workflows",
                    "description": "Check unread and mark messages read.",
                    "triggers": ["mark emails read", "mark all read"],
                    "body": "Call `mail_mcp__imap_mark_read` for each message.",
                    "tool_steps": [
                        {
                            "toolName": "mail_mcp__imap_mark_read",
                            "arguments": {"message_id": "{{message_id}}"},
                        }
                    ],
                    "update_parent": True,
                }
            ),
        ),
    )

    result = await observe_skill_override(
        llm,
        MagicMock(),
        parent_skill=parent,
        trace=trace,
        history=[],
    )
    assert result is not None
    assert result.learn is True
    assert result.update_parent is True
    assert result.draft is not None


def test_build_observer_payload_marks_discovery_tools() -> None:
    trace = TurnTrace(
        user_text="snapshot the door cam",
        history_len=0,
        route="action",
        tool_calls=[
            {
                "toolName": "mcp_proxy__searchToolsForDomain",
                "arguments": {},
                "succeeded": True,
                "discovery": True,
            },
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {"domain": "camera"},
                "succeeded": True,
                "discovery": False,
            },
        ],
    )
    payload = build_observer_payload(trace, [], manual_save=False)
    assert payload["tools"][0]["discovery"] is True
    assert payload["tools"][1]["discovery"] is False


@pytest.mark.asyncio
async def test_observe_skill_candidate_approves() -> None:
    trace = TurnTrace(
        user_text="turn off dining lights every evening",
        history_len=0,
        tool_calls=[
            {"toolName": "home_assistant__ha_call_service", "succeeded": True},
            {"toolName": "home_assistant__ha_get_state", "succeeded": True},
        ],
        assistant_text="Dining lights are off.",
        iterations=2,
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "learn": True,
                    "reason": "repeatable workflow",
                    "title": "Evening lights",
                    "description": "Turns off dining lights.",
                    "triggers": ["turn off dining lights"],
                    "body": "Call ha_call_service.",
                    "tool_steps": [],
                }
            ),
        ),
    )

    result = await observe_skill_candidate(
        llm,
        MagicMock(),
        trace=trace,
        history=[],
    )
    assert result.learn is True
    assert result.draft is not None


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

    assert not await assess_skill_worth_learning(
        llm,
        MagicMock(),
        trace=trace,
        history=[],
    )

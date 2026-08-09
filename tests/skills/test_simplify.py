"""Tests for LLM skill simplification proposals."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


def _load_simplify():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    # Minimal stubs for HA imports used by simplify.py
    if "homeassistant" not in sys.modules:
        ha = types.ModuleType("homeassistant")
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object
        sys.modules["homeassistant"] = ha
        sys.modules["homeassistant.core"] = ha_core

    if "ha_agent.const" not in sys.modules:
        const = types.ModuleType("ha_agent.const")
        const.DATA_KEY = "ha_agent"
        const.LOGGER = MagicMock()
        const.DOMAIN = "ha_agent"
        sys.modules["ha_agent.const"] = const

    if "ha_agent.config_helpers" not in sys.modules:
        helpers = types.ModuleType("ha_agent.config_helpers")

        class LlmBackend:
            def __init__(self, **kwargs):
                self.base_url = kwargs.get("base_url", "")
                self.model = kwargs.get("model", "")
                self.api_key = kwargs.get("api_key")
                self.max_tokens = kwargs.get("max_tokens", 1024)
                self.temperature = kwargs.get("temperature", 0.2)
                self.timeout = kwargs.get("timeout", 30)
                self.thinking_level = kwargs.get("thinking_level", "off")

            def __replace__(self, **changes):
                data = {
                    "base_url": self.base_url,
                    "model": self.model,
                    "api_key": self.api_key,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                    "thinking_level": self.thinking_level,
                }
                data.update(changes)
                return LlmBackend(**data)

        helpers.LlmBackend = LlmBackend  # type: ignore[attr-defined]
        helpers.get_llm_backend = MagicMock()  # type: ignore[attr-defined]
        sys.modules["ha_agent.config_helpers"] = helpers

    if "ha_agent.llm_client" not in sys.modules:
        llm_client = types.ModuleType("ha_agent.llm_client")
        llm_client.LlmClient = object
        sys.modules["ha_agent.llm_client"] = llm_client

    for name in ("models", "body", "tool_names"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    mod_name = "ha_agent.skills.simplify"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    path = COMPONENT / "skills" / "simplify.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


simplify = _load_simplify()
Skill = sys.modules["ha_agent.skills.models"].Skill


def _skill(skill_id: str, *, title: str, tools: list[str] | None = None) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        title=title,
        description=title,
        triggers=[title.lower()],
        body=f"# {title}\n\nDo the thing.",
        tool_steps=[
            {"toolName": name, "arguments": {}}
            for name in (tools or ["home_assistant__ha_get_state"])
        ],
        route_scope="action",
        use_count=3,
    )


def test_parse_simplify_combine_and_simplify_proposals() -> None:
    alpha = _skill("a", title="Dining lights off", tools=["light.turn_off"])
    beta = _skill("b", title="Kitchen lights off", tools=["light.turn_off"])
    gamma = _skill(
        "c",
        title="Verbose temp skill",
        tools=["home_assistant__ha_get_state"],
    )
    payload = {
        "summary": "Merge light skills; shorten temperature skill.",
        "proposals": [
            {
                "action": "combine",
                "skill_ids": ["a", "b"],
                "survivor_id": "a",
                "reason": "Same tool, different rooms.",
                "draft": {
                    "title": "Turn off room lights",
                    "description": "Turn off lights in a room",
                    "triggers": ["turn off lights", "lights off"],
                    "body": "# Lights\nUse {{entity_id}}",
                    "tool_steps": [
                        {
                            "toolName": "light.turn_off",
                            "arguments": {"entity_id": "{{entity_id}}"},
                        }
                    ],
                    "slots": [
                        {
                            "name": "entity_id",
                            "description": "Light entity",
                            "source": "user",
                            "default": None,
                        }
                    ],
                    "route_scope": "action",
                },
            },
            {
                "action": "simplify",
                "skill_ids": ["c"],
                "survivor_id": "c",
                "reason": "Body is too long.",
                "draft": {
                    "title": "Room temperature",
                    "description": "Read room temperature",
                    "triggers": ["temperature"],
                    "body": "# Temperature\nRead sensor.",
                    "tool_steps": [
                        {"toolName": "home_assistant__ha_get_state", "arguments": {}}
                    ],
                    "route_scope": "action",
                },
            },
            {
                "action": "combine",
                "skill_ids": ["missing"],
                "survivor_id": "missing",
                "reason": "drop",
                "draft": {"title": "x", "body": "y"},
            },
        ],
    }
    summary, proposals = simplify.parse_simplify_response(
        json.dumps(payload),
        skills_by_id={"a": alpha, "b": beta, "c": gamma},
        model_used="strong-model",
    )
    assert summary.startswith("Merge light")
    assert len(proposals) == 2
    assert proposals[0].action == "combine"
    assert proposals[0].skill_ids == ["a", "b"]
    assert proposals[0].draft.title == "Turn off room lights"
    assert proposals[0].draft.slots[0].name == "entity_id"
    assert proposals[1].action == "simplify"
    assert proposals[1].model_used == "strong-model"


def test_parse_rejects_combine_with_one_skill() -> None:
    skill = _skill("only", title="Solo")
    summary, proposals = simplify.parse_simplify_response(
        json.dumps(
            {
                "summary": "noop",
                "proposals": [
                    {
                        "action": "combine",
                        "skill_ids": ["only"],
                        "survivor_id": "only",
                        "reason": "bad",
                        "draft": {
                            "title": "Solo",
                            "description": "Solo",
                            "triggers": ["solo"],
                            "body": "# Solo",
                            "tool_steps": [],
                        },
                    }
                ],
            }
        ),
        skills_by_id={"only": skill},
    )
    assert summary == "noop"
    assert proposals == []


def test_proposal_to_dict_roundtrip_keys() -> None:
    skill = _skill("a", title="A")
    _, proposals = simplify.parse_simplify_response(
        json.dumps(
            {
                "summary": "ok",
                "proposals": [
                    {
                        "action": "simplify",
                        "skill_ids": ["a"],
                        "survivor_id": "a",
                        "reason": "trim",
                        "draft": {
                            "title": "A",
                            "description": "A",
                            "triggers": ["a"],
                            "body": "# A",
                            "tool_steps": [{"toolName": "x"}],
                        },
                    }
                ],
            }
        ),
        skills_by_id={"a": skill},
    )
    data = simplify.proposal_to_dict(proposals[0])
    assert data["action"] == "simplify"
    assert data["draft"]["title"] == "A"
    assert data["proposal_id"]

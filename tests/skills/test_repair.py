"""Tests for skill auto-repair."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_repair():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in ("const", "models", "observer", "defaults", "params", "store"):
        mod_name = "ha_agent.const" if name == "const" else f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / ("const.py" if name == "const" else f"skills/{name}.py")
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    if "homeassistant.core" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.core"] = ha_core

    path = COMPONENT / "skills" / "repair.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.repair", path)
    repair = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["ha_agent.skills.repair"] = repair
    spec.loader.exec_module(repair)
    return repair, sys.modules["ha_agent.skills.models"]


repair_mod, models_mod = _load_repair()
Skill = models_mod.Skill
TurnTrace = models_mod.TurnTrace
detect_repairable_issues = repair_mod.detect_repairable_issues
repair_skill_from_trace = repair_mod.repair_skill_from_trace


def test_detect_missing_mailbox_param() -> None:
    skill = Skill(
        id="s1",
        slug="email-mgmt",
        title="Email Management",
        description="Check email",
        triggers=["email"],
        body="Check inbox",
        tool_steps=[{"toolName": "mail_mcp__imap_mailbox_status"}],
        route_scope="email",
    )
    trace = TurnTrace(
        user_text="any new emails?",
        history_len=0,
        tool_calls=[
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "arguments": {},
                "succeeded": False,
                "error_kind": "param",
                "missing_fields": ["mailbox"],
            }
        ],
    )
    issues = detect_repairable_issues(trace, skill)
    assert any(issue.kind == "missing_param" for issue in issues)


def test_repair_adds_mailbox_to_imap_steps() -> None:
    skill = Skill(
        id="s1",
        slug="email-mgmt",
        title="Email Management",
        description="Check email",
        triggers=["email"],
        body="Check inbox",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status"},
            {"toolName": "mail_mcp__imap_search_messages"},
        ],
        route_scope="email",
    )
    trace = TurnTrace(
        user_text="any new emails?",
        history_len=0,
        route="email",
        tool_calls=[
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "arguments": {},
                "succeeded": False,
                "error_kind": "param",
                "missing_fields": ["mailbox"],
            }
        ],
    )
    result = repair_skill_from_trace(skill, trace)
    assert result is not None
    updated, reason = result
    assert "mailbox" in reason.lower()
    assert updated.slots
    assert any(s.name == "mailbox" for s in updated.slots)
    for step in updated.tool_steps:
        args = step.get("arguments") or {}
        assert args.get("mailbox") == "{{mailbox}}"

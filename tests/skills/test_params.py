"""Unit tests for skill slot binding helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_modules():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    loaded = {}
    for name in ("models", "defaults", "params"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            loaded[name] = sys.modules[mod_name]
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


mods = _load_modules()
Skill = mods["models"].Skill
default_slots_for_route = mods["defaults"].default_slots_for_route
apply_route_defaults = mods["defaults"].apply_route_defaults
missing_required_bindings = mods["params"].missing_required_bindings


def test_default_slots_for_email_route() -> None:
    slots = default_slots_for_route("email")
    assert len(slots) == 1
    assert slots[0].name == "mailbox"
    assert slots[0].default == "INBOX"


def test_apply_route_defaults_merges_email_mailbox() -> None:
    skill = Skill(
        id="1",
        slug="email",
        title="Email",
        description="Check email.",
        triggers=["email"],
        body="Check inbox.",
        tool_steps=[],
        route_scope="email",
    )
    updated = apply_route_defaults(skill)
    assert any(slot.name == "mailbox" for slot in updated.slots)


def test_missing_required_bindings_detects_unbound_slot() -> None:
    skill = Skill(
        id="1",
        slug="email",
        title="Email",
        description="Check email.",
        triggers=["email"],
        body="Use mailbox {{mailbox}}.",
        tool_steps=[
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "arguments": {"mailbox": "{{mailbox}}"},
            }
        ],
        route_scope="email",
    )
    missing = missing_required_bindings(skill, {})
    assert "mailbox" in missing

    bound = missing_required_bindings(skill, {"mailbox": "INBOX"})
    assert bound == []


def test_bind_tool_steps_fills_entity_and_service_slots() -> None:
    """Universal on/off plans render entity_id/service from bindings."""
    bind_tool_steps = mods["params"].bind_tool_steps
    steps = bind_tool_steps(
        [
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "service": "{{service}}",
                    "entity_id": "{{entity_id}}",
                },
            }
        ],
        {
            "service": "turn_on",
            "entity_id": "light.dining_room_lights_ceiling",
        },
    )
    assert steps[0]["arguments"] == {
        "service": "turn_on",
        "entity_id": "light.dining_room_lights_ceiling",
    }
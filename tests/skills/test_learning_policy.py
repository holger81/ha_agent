"""Tests for generic skill learning policy."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"

_MODULE_DEPS: dict[str, list[str]] = {
    "const": [],
    "skills.models": [],
    "skills.tool_names": ["const", "skills.models"],
    "skills.defaults": ["skills.models"],
    "skills.body": [
        "skills.models",
        "skills.tool_names",
        "skills.defaults",
    ],
    "skills.observer": [
        "const",
        "skills.models",
        "skills.body",
        "skills.learning_policy",
    ],
    "skills.learning_policy": [
        "skills.models",
        "skills.body",
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


policy = _load_module("skills.learning_policy")
models = _load_module("skills.models")
observer = _load_module("skills.observer")

Skill = models.Skill
SkillDraft = models.SkillDraft
TurnTrace = models.TurnTrace
SkillObserverResult = observer.SkillObserverResult


def _email_check_skill() -> Skill:
    return Skill(
        id="parent-1",
        slug="check-unread",
        title="Check unread email",
        description="Unread count",
        triggers=["unread email", "check inbox"],
        body="Search unread mail.",
        tool_steps=[
            {"toolName": "mail_mcp__imap_mailbox_status", "arguments": {}},
            {"toolName": "mail_mcp__imap_search_messages", "arguments": {}},
        ],
        route_scope="email",
    )


def test_analyze_workflow_delta_forks_when_new_mutating_tools() -> None:
    """Read-only parent + mutating trace tools recommend fork."""
    parent = _email_check_skill()
    trace = TurnTrace(
        user_text="mark all unread as read",
        history_len=0,
        tool_calls=[
            {"toolName": "mail_mcp__imap_search_messages", "succeeded": True},
            {"toolName": "mail_mcp__imap_bulk_update_flags", "succeeded": True},
        ],
        outcome="success",
        skill_plan_override=True,
    )

    delta = policy.analyze_workflow_delta(parent, trace)

    assert delta.recommendation == "fork"
    assert "mail_mcp__imap_bulk_update_flags" in delta.new_tools
    assert "mutate" in delta.added_effects


def test_analyze_workflow_delta_updates_when_parent_empty() -> None:
    """Bootstrap skills with no concrete steps can update in place."""
    parent = Skill(
        id="parent-2",
        slug="draft",
        title="Draft skill",
        description="",
        triggers=["do thing"],
        body="Placeholder.",
        tool_steps=[],
    )
    trace = TurnTrace(
        user_text="do thing",
        history_len=0,
        tool_calls=[
            {"toolName": "domain__do_thing", "succeeded": True, "arguments": {}},
        ],
        outcome="success",
        skill_plan_override=True,
    )

    delta = policy.analyze_workflow_delta(parent, trace)

    assert delta.recommendation == "update"


def test_resolve_override_observer_result_forks_despite_llm_merge() -> None:
    """Policy overrides observer update_parent=true when workflows diverge."""
    parent = _email_check_skill()
    trace = TurnTrace(
        user_text="mark all unread emails in INBOX as read",
        history_len=0,
        tool_calls=[
            {
                "toolName": "mail_mcp__imap_search_messages",
                "succeeded": True,
                "arguments": {"mailbox": "INBOX", "unread_only": True},
            },
            {
                "toolName": "mail_mcp__imap_bulk_update_flags",
                "succeeded": True,
                "arguments": {"add_flags": ["\\Seen"]},
            },
        ],
        outcome="success",
        skill_plan_override=True,
    )
    observed = SkillObserverResult(
        learn=True,
        reason="extend email skill",
        draft=SkillDraft(
            title="Mark all unread emails in INBOX as read",
            description="Mark read",
            triggers=["mark read"],
            body="Mark messages read.",
            tool_steps=[],
        ),
        update_parent=True,
    )

    resolved = policy.resolve_override_observer_result(parent, trace, observed)

    assert resolved.update_parent is False
    assert resolved.draft.parent_id == parent.id
    assert resolved.draft.title == "Mark all unread emails in INBOX as read"
    assert any(
        step["toolName"] == "mail_mcp__imap_bulk_update_flags"
        for step in resolved.draft.tool_steps
    )


def test_merge_parent_skill_draft_preserves_parent_workflow() -> None:
    """In-place updates union triggers and append tool steps."""
    parent = _email_check_skill()
    draft = SkillDraft(
        title="Different title",
        description="Extra",
        triggers=["mark read"],
        body="Also mark messages read.",
        tool_steps=[
            {
                "toolName": "mail_mcp__imap_bulk_update_flags",
                "arguments": {"add_flags": ["\\Seen"]},
            }
        ],
    )
    trace = TurnTrace(user_text="mark read", history_len=0)

    merged = policy.merge_parent_skill_draft(parent, draft, trace)

    assert merged.title == parent.title
    assert "check inbox" in merged.triggers
    assert "mark read" in merged.triggers
    assert merged.tool_steps[0]["toolName"] == "mail_mcp__imap_mailbox_status"
    assert merged.tool_steps[-1]["toolName"] == "mail_mcp__imap_bulk_update_flags"


def test_draft_would_regress_parent_when_steps_wiped() -> None:
    """Empty draft tool_steps must not replace a concrete parent skill."""
    parent = _email_check_skill()
    draft = SkillDraft(
        title="Broken",
        description="",
        triggers=["x"],
        body="Open your email client and click mark read.",
        tool_steps=[],
    )

    assert policy.draft_would_regress_parent(parent, draft) is True


def test_build_deterministic_override_result_forks_child() -> None:
    """Deterministic fallback creates a child skill with trace tool_steps."""
    parent = _email_check_skill()
    trace = TurnTrace(
        user_text="mark all unread emails in INBOX as read",
        history_len=0,
        tool_calls=[
            {
                "toolName": "mail_mcp__imap_search_messages",
                "succeeded": True,
                "arguments": {"mailbox": "INBOX", "unread_only": True},
            },
            {
                "toolName": "mail_mcp__imap_bulk_update_flags",
                "succeeded": True,
                "arguments": {"add_flags": ["\\Seen"]},
            },
        ],
        outcome="success",
        skill_plan_override=True,
    )

    result = policy.build_deterministic_override_result(parent, trace)

    assert result is not None
    assert result.update_parent is False
    assert result.draft.parent_id == parent.id
    assert any(
        step["toolName"] == "mail_mcp__imap_bulk_update_flags"
        for step in result.draft.tool_steps
    )


def test_prepare_learned_draft_rejects_discovery_only_turn() -> None:
    """Prose or discovery-only turns cannot become skills."""
    draft = SkillDraft(
        title="Turn off dining room light",
        description="I have turned off the dining room lights.",
        triggers=["turn off dining room light"],
        body="1. Open the smart home app.",
        tool_steps=[],
    )
    trace = TurnTrace(
        user_text="turn off dining room light",
        history_len=0,
        tool_calls=[
            {
                "toolName": "searchToolsForDomain",
                "succeeded": True,
                "arguments": {"domain": "smart-home"},
            }
        ],
    )
    assert policy.prepare_learned_draft(draft, trace) is None


def test_prepare_learned_draft_grounds_and_slotifies_action() -> None:
    """Successful control tools become slotted tool_steps."""
    draft = SkillDraft(
        title="Turn off dining room light",
        description="Turn off dining room lights with MCP.",
        triggers=["turn off dining room light"],
        body="Use `home_assistant__ha_call_service`.",
        tool_steps=[],
    )
    trace = TurnTrace(
        user_text="turn off dining room light",
        history_len=0,
        route="action",
        tool_calls=[
            {
                "toolName": "home_assistant__ha_call_service",
                "succeeded": True,
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.dining_room",
                },
            }
        ],
        controlled_entity_ids=["light.dining_room"],
    )
    prepared = policy.prepare_learned_draft(draft, trace)
    assert prepared is not None
    assert prepared.tool_steps
    assert prepared.tool_steps[0]["arguments"]["entity_id"] == "{{entity_id}}"
    assert any(slot.name == "entity_id" for slot in prepared.slots)

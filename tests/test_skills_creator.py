"""Tests for skill distillation helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)

_MODULE_DEPS: dict[str, list[str]] = {
    "const": [],
    "config_helpers": ["const"],
    "embedded_tools": [],
    "llm_client": ["const", "config_helpers", "embedded_tools"],
    "skills.models": [],
    "skills.store": ["const", "skills.models"],
    "skills.creator": [
        "const",
        "config_helpers",
        "llm_client",
        "skills.models",
        "skills.store",
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


def _fallback_skill_draft(trace):
    creator = _load_module("skills.creator")
    return creator._fallback_skill_draft(trace)


def _turn_trace(**kwargs):
    models = _load_module("skills.models")
    return models.TurnTrace(**kwargs)


def test_fallback_skill_draft_uses_upstream_tool_names() -> None:
    trace = _turn_trace(
        user_text="check my new emails",
        history_len=2,
        assistant_text="You have 3 unread messages.",
        tool_calls=[
            {
                "toolName": "mail_mcp_imap_mailbox_status",
                "name": "mail_mcp_imap_mailbox_status",
                "arguments": {"account_id": "default", "mailbox": "INBOX"},
            },
            {
                "toolName": "mail_mcp_imap_search_messages",
                "name": "mail_mcp_imap_search_messages",
                "arguments": {"unread_only": True},
            },
        ],
    )

    draft = _fallback_skill_draft(trace)

    assert draft is not None
    assert draft.title == "check my new emails"
    assert len(draft.tool_steps) == 2
    assert draft.tool_steps[0]["toolName"] == "mail_mcp_imap_mailbox_status"
    assert "mail_mcp_imap_search_messages" in draft.body


def test_fallback_skill_draft_requires_tool_calls() -> None:
    trace = _turn_trace(user_text="hello", history_len=0)

    assert _fallback_skill_draft(trace) is None

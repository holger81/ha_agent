"""Tests for turn prepass route/skill consistency."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _ensure_ha_stubs() -> None:
    if "homeassistant" not in sys.modules:
        ha = types.ModuleType("homeassistant")
        sys.modules["homeassistant"] = ha
    if "homeassistant.core" not in sys.modules:
        core = types.ModuleType("homeassistant.core")

        class HomeAssistant:
            pass

        core.HomeAssistant = HomeAssistant
        sys.modules["homeassistant.core"] = core


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if name.startswith("skills.") and "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    deps = {
        "prepass": [
            "config_helpers",
            "const",
            "context",
            "llm_client",
            "llm_telemetry",
            "orchestrator",
            "router",
            "skills.models",
            "skills.params",
            "skills.selection",
            "skills.store",
            "structured_output",
        ],
        "skills.selection": [
            "const",
            "config_helpers",
            "context",
            "llm_client",
            "structured_output",
            "skills.discovery",
            "skills.models",
            "skills.store",
        ],
        "skills.discovery": ["skills.models", "skills.store", "skills.format"],
        "skills.store": ["skills.models", "const"],
        "skills.params": ["skills.models"],
        "skills.format": ["skills.models"],
        "skills.models": [],
        "config_helpers": ["const"],
        "router": ["config_helpers", "context"],
        "orchestrator": ["config_helpers", "const", "llm_client", "structured_output"],
        "llm_client": ["const", "config_helpers"],
        "llm_telemetry": [],
        "structured_output": [],
        "const": [],
        "context": [],
    }

    if name in {"context", "router"}:
        _ensure_ha_stubs()
        if "homeassistant.components" not in sys.modules:
            sys.modules["homeassistant.components"] = types.ModuleType(
                "homeassistant.components"
            )
        if "homeassistant.components.conversation" not in sys.modules:
            sys.modules["homeassistant.components.conversation"] = types.ModuleType(
                "homeassistant.components.conversation"
            )

    if name in {"skills.store", "prepass", "llm_client"}:
        _ensure_ha_stubs()

    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    if name.startswith("skills."):
        path = COMPONENT / "skills" / f"{name.split('.', 1)[1]}.py"
    else:
        path = COMPONENT / f"{name}.py"

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_align_route_uses_skill_scope_not_hardcoded_domains() -> None:
    router = _load("router")
    route, hint, reason = router.align_route_to_skill(
        router.TaskRoute.HA_ACTION,
        skill_scope="calendar",
        domain_hint=None,
    )
    assert route == router.TaskRoute.CHAT
    assert hint == "calendar"
    assert reason and "calendar" in reason

    route, hint, reason = router.align_route_to_skill(
        router.TaskRoute.CHAT,
        skill_scope="action",
        domain_hint=None,
    )
    assert route == router.TaskRoute.HA_ACTION
    assert hint is None
    assert reason and "action" in reason

    # Distilled status skills often mis-tag route_scope=action.
    route, hint, reason = router.align_route_to_skill(
        router.TaskRoute.HA_ACTION,
        skill_scope="action",
        domain_hint=None,
        skill_tool_steps=[
            {"toolName": "home_assistant__ha_search", "arguments": {}},
            {"toolName": "home_assistant__ha_get_state", "arguments": {}},
        ],
    )
    assert route == router.TaskRoute.CHAT
    assert reason and "lookup-only" in reason


def test_align_route_domain_hint_forces_chat_without_skill() -> None:
    router = _load("router")
    route, hint, reason = router.align_route_to_skill(
        router.TaskRoute.HA_ACTION,
        skill_scope=None,
        domain_hint="weather",
    )
    assert route == router.TaskRoute.CHAT
    assert hint == "weather"
    assert reason and "weather" in reason


def test_prepass_aligns_action_to_skill_scope() -> None:
    prepass = _load("prepass")
    models = _load("skills.models")
    skill = models.Skill(
        id="1",
        slug="news-briefing",
        title="News briefing",
        description="Curate headlines",
        triggers=["news"],
        body="# News",
        tool_steps=[{"toolName": "mcp_news__news_curate", "arguments": {}}],
        route_scope="news",
    )
    keyword = SimpleNamespace(
        summary="domain hint → news (news: news)",
        domain_hint="news",
        route=prepass.TaskRoute.CHAT,
    )
    result = prepass._parse_prepass_payload(
        {
            "route": "action",
            "domain_hint": "news",
            "complexity": "single",
            "skill_slug": "news-briefing",
            "slot_bindings": {"digest_scope": "news"},
            "reason": "confused",
        },
        catalog_by_slug={"news-briefing": skill},
        keyword_decision=keyword,
        heuristic=prepass.Complexity.SINGLE,
        user_text="what are today's news",
    )
    assert result is not None
    assert result.route_resolution.route == prepass.TaskRoute.CHAT
    assert result.route_resolution.domain_hint == "news"
    assert result.skill_selection is not None
    assert result.skill_selection.skills[0].slug == "news-briefing"


def test_prepass_aligns_from_custom_skill_scope_alone() -> None:
    prepass = _load("prepass")
    models = _load("skills.models")
    skill = models.Skill(
        id="1",
        slug="weekly-digest",
        title="Weekly digest",
        description="Custom digest workflow",
        triggers=["digest"],
        body="# Digest",
        tool_steps=[{"toolName": "custom__digest", "arguments": {}}],
        route_scope="digest",
    )
    keyword = SimpleNamespace(
        summary="default chat",
        domain_hint=None,
        route=prepass.TaskRoute.CHAT,
    )
    result = prepass._parse_prepass_payload(
        {
            "route": "action",
            "domain_hint": "",
            "complexity": "single",
            "skill_slug": "weekly-digest",
            "slot_bindings": {},
        },
        catalog_by_slug={"weekly-digest": skill},
        keyword_decision=keyword,
        heuristic=prepass.Complexity.SINGLE,
        user_text="make a weekly digest",
    )
    assert result is not None
    assert result.route_resolution.route == prepass.TaskRoute.CHAT
    assert result.route_resolution.domain_hint == "digest"


def test_prepass_trusts_intent_skill_despite_weak_lexical_overlap() -> None:
    prepass = _load("prepass")
    models = _load("skills.models")
    skill = models.Skill(
        id="1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Parameterized status lookup",
        triggers=["status of {{query}}", "look up {{query}} sensor"],
        body="# Status",
        tool_steps=[{"toolName": "home_assistant__ha_search", "arguments": {}}],
        route_scope="action",
    )
    keyword = SimpleNamespace(
        summary="default chat",
        domain_hint=None,
        route=prepass.TaskRoute.HA_ACTION,
    )
    result = prepass._parse_prepass_payload(
        {
            "route": "action",
            "complexity": "single",
            "skill_slug": "look-up-sensor-or-entity-status",
            "slot_bindings": {"query": "great room"},
        },
        catalog_by_slug={"look-up-sensor-or-entity-status": skill},
        keyword_decision=keyword,
        heuristic=prepass.Complexity.SINGLE,
        user_text="what is the temperature in the great room",
    )
    assert result is not None
    assert result.skill_selection is not None
    assert result.skill_selection.skills[0].slug == ("look-up-sensor-or-entity-status")


def test_prepass_drops_ha_status_skill_on_email_ask() -> None:
    """Email asks must not keep an HA entity-status skill from prepass."""
    prepass = _load("prepass")
    models = _load("skills.models")
    skill = models.Skill(
        id="1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Parameterized status lookup",
        triggers=["status of {{query}}", "look up {{query}} sensor"],
        body="# Status",
        tool_steps=[{"toolName": "home_assistant__ha_search", "arguments": {}}],
        route_scope="action",
    )
    keyword = SimpleNamespace(
        summary="default chat",
        domain_hint=None,
        route=prepass.TaskRoute.CHAT,
    )
    result = prepass._parse_prepass_payload(
        {
            "route": "chat",
            "domain_hint": "",
            "complexity": "single",
            "skill_slug": "look-up-sensor-or-entity-status",
            "slot_bindings": {"query": "do I have new emails"},
        },
        catalog_by_slug={"look-up-sensor-or-entity-status": skill},
        keyword_decision=keyword,
        heuristic=prepass.Complexity.SINGLE,
        user_text="do I have new emails",
    )
    assert result is not None
    assert result.route_resolution.domain_hint == "email"
    assert result.skill_selection is None
    assert "dropped conflicting skill" in result.orch_plan.reason

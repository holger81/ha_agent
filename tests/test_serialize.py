"""Unit tests for API serialization helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_exc() -> None:
    if "homeassistant.exceptions" in sys.modules:
        return
    ha_pkg = types.ModuleType("homeassistant")
    ha_exc = types.ModuleType("homeassistant.exceptions")

    class HomeAssistantError(Exception):
        pass

    ha_exc.HomeAssistantError = HomeAssistantError
    sys.modules.setdefault("homeassistant", ha_pkg)
    sys.modules["homeassistant.exceptions"] = ha_exc


def _load_serialize():
    if "ha_agent.api.serialize" in sys.modules:
        return sys.modules["ha_agent.api.serialize"]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    api_pkg = types.ModuleType("ha_agent.api")
    api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.api"] = api_pkg

    _ensure_ha_exc()

    for name in ("models", "body", "markdown"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    # Stub heavy identity / memory imports used only for type-level imports.
    for stub_name, attrs in (
        ("ha_agent.identity.models", ("AgentUser", "ResolvedIdentity", "VoiceProfile")),
        ("ha_agent.persistent_memory.models", ("MemoryEntry",)),
        ("ha_agent.recovery_hints", ("RecoveryHint",)),
        ("ha_agent.route_keywords", ("RouteKeywords",)),
    ):
        if stub_name in sys.modules:
            continue
        stub = types.ModuleType(stub_name)
        for attr in attrs:
            setattr(stub, attr, type(attr, (), {}))
        parent = stub_name.rsplit(".", 1)[0]
        if parent not in sys.modules:
            parent_mod = types.ModuleType(parent)
            parent_mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[parent] = parent_mod
        sys.modules[stub_name] = stub

    path = COMPONENT / "api" / "serialize.py"
    spec = importlib.util.spec_from_file_location("ha_agent.api.serialize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.api.serialize"] = module
    spec.loader.exec_module(module)
    return module


serialize = _load_serialize()
models = sys.modules["ha_agent.skills.models"]
Skill = models.Skill
SkillDraft = models.SkillDraft
PendingSkillDraft = models.PendingSkillDraft
TurnTrace = models.TurnTrace


def test_skill_to_dict_roundtrip_fields() -> None:
    skill = Skill(
        id="id-1",
        slug="test-skill",
        title="Test",
        description="Desc",
        triggers=["hello"],
        body="Do the thing",
        tool_steps=[{"toolName": "callTool"}],
        enabled=True,
        created_at=1.0,
        use_count=2,
    )
    data = serialize.skill_to_dict(skill)
    assert data["id"] == "id-1"
    assert data["title"] == "Test"
    assert data["triggers"] == ["hello"]
    assert data["tool_steps"][0]["toolName"] == "callTool"


def test_pending_draft_to_dict_includes_markdown_and_slots() -> None:
    draft = SkillDraft(
        title="News briefing",
        description="Curate headlines",
        triggers=["news"],
        body="# News\n\nCall tools.",
        tool_steps=[{"toolName": "mcp_news__news_curate", "arguments": {}}],
        slots=[models.SkillSlot(name="digest_scope", default="")],
        route_scope="news",
    )
    pending = PendingSkillDraft(
        entry_id="entry",
        conversation_id="console-1",
        trace=TurnTrace(user_text="news", history_len=0),
        history=[],
        skill_draft=draft,
        observer_reason="repeatable briefing",
        update_skill_id="skill-9",
    )
    data = serialize.pending_draft_to_dict(pending)
    assert data["update_skill_id"] == "skill-9"
    assert data["observer_reason"] == "repeatable briefing"
    sd = data["skill_draft"]
    assert sd["title"] == "News briefing"
    assert sd["route_scope"] == "news"
    assert sd["slots"][0]["name"] == "digest_scope"
    assert "markdown" in sd
    assert "News briefing" in sd["markdown"]
    assert "mcp_news__news_curate" in sd["markdown"]

"""Unit tests for skill SQLite store and FTS."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


def _load_store_module():
    mod_name = "ha_agent.skills.store"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    skills_pkg = types.ModuleType("ha_agent.skills")
    skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.skills"] = skills_pkg

    for name in ("const",):
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[f"ha_agent.{name}"] = module
        spec.loader.exec_module(module)

    for name in ("models",):
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"ha_agent.skills.{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[f"ha_agent.skills.{name}"] = module
        spec.loader.exec_module(module)

    path = COMPONENT / "skills" / "store.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


store_mod = _load_store_module()
SkillStore = store_mod.SkillStore
_build_fts_query = store_mod._build_fts_query


@pytest.fixture
def store(tmp_path: Path) -> SkillStore:
    """Return a connected skill store in a temp directory."""
    db = tmp_path / "skills.db"
    skill_store = SkillStore(db)
    skill_store.connect()
    yield skill_store
    skill_store.close()


def test_insert_and_search_skill(store: SkillStore) -> None:
    """FTS finds a skill by trigger phrase."""
    store.insert_skill(
        title="Dining room lights",
        description="Turn dining room ceiling lights on or off.",
        triggers=["turn on dining room lights", "dining room lights"],
        body="1. Use light.dining_room_ceiling 2. call turn_on or turn_off",
        tool_steps=[{"toolName": "home_assistant__ha_call_service"}],
    )
    matches = store.search("turn on dining room lights", limit=5)
    assert len(matches) == 1
    assert matches[0].title == "Dining room lights"


def test_enabled_filter_excludes_disabled(store: SkillStore) -> None:
    """Disabled skills are not returned when enabled_only is true."""
    skill = store.insert_skill(
        title="Bedroom lights",
        description="Control bedroom lights.",
        triggers=["bedroom lights"],
        body="workflow",
        tool_steps=[],
    )
    store.set_enabled(skill.id, False)
    assert store.search("bedroom lights", enabled_only=True) == []
    assert store.search("bedroom lights", enabled_only=False)


def test_find_duplicate(store: SkillStore) -> None:
    """Near-duplicate triggers match an existing skill."""
    store.insert_skill(
        title="Patio cover",
        description="Open or close patio cover.",
        triggers=["open patio cover", "open the patio cover"],
        body="workflow",
        tool_steps=[],
    )
    duplicate = store.find_duplicate(["open the patio cover"])
    assert duplicate is not None
    assert duplicate.title == "Patio cover"


def test_record_use_and_improvement_cooldown(store: SkillStore) -> None:
    """Improvement cooldown blocks until one hour elapsed."""
    import time

    skill = store.insert_skill(
        title="Test",
        description="Test skill.",
        triggers=["test"],
        body="body",
        tool_steps=[],
    )
    assert store.can_improve(skill.id) is True
    skill.last_improved_at = time.time()
    store.update_skill(skill)
    assert store.can_improve(skill.id) is False
    updated = store.record_use(skill.id, succeeded=True)
    assert updated is not None
    assert updated.use_count == 1
    assert updated.success_count == 1


def test_build_fts_query_strips_stop_words() -> None:
    """FTS query omits common filler words."""
    query = _build_fts_query("please turn on the dining room lights")
    assert "please" not in query
    assert "dining" in query


def test_search_prefers_higher_scored_skill(store: SkillStore) -> None:
    """Composite FTS ranking favors skills with better scores."""
    low = store.insert_skill(
        title="Email check low",
        description="Check email inbox unread messages.",
        triggers=["check email inbox"],
        body="workflow",
        tool_steps=[],
    )
    high = store.insert_skill(
        title="Email check high",
        description="Check email inbox unread messages.",
        triggers=["check email inbox"],
        body="workflow",
        tool_steps=[],
    )
    low.score = 0.2
    high.score = 0.95
    store.update_skill(low)
    store.update_skill(high)

    matches = store.search("check email inbox", limit=5)
    assert len(matches) >= 2
    assert matches[0].id == high.id


def test_revision_save_list_restore(store: SkillStore) -> None:
    """Revision snapshots can be saved, listed, and restored."""
    skill = store.insert_skill(
        title="Revision test",
        description="Original description.",
        triggers=["revision test"],
        body="original body",
        tool_steps=[{"toolName": "tool_a"}],
    )
    revision_id = store.save_revision(skill, reason="before edit")
    skill.description = "Updated description."
    skill.body = "updated body"
    skill.version += 1
    store.update_skill(skill)

    revisions = store.list_revisions(skill.id)
    assert len(revisions) == 1
    assert revisions[0].id == revision_id
    assert revisions[0].reason == "before edit"

    restored = store.restore_revision(revision_id)
    assert restored is not None
    assert restored.description == "Original description."
    assert restored.body == "original body"
    assert restored.version > skill.version

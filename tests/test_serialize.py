"""Unit tests for API serialization helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_serialize():
    if "ha_agent.api.serialize" in sys.modules:
        return sys.modules["ha_agent.api.serialize"]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    api_pkg = types.ModuleType("ha_agent.api")
    api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.api"] = api_pkg

    models_path = COMPONENT / "skills" / "models.py"
    spec = importlib.util.spec_from_file_location("ha_agent.skills.models", models_path)
    assert spec is not None and spec.loader is not None
    models = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.skills.models"] = models
    spec.loader.exec_module(models)

    path = COMPONENT / "api" / "serialize.py"
    spec = importlib.util.spec_from_file_location("ha_agent.api.serialize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ha_agent.api.serialize"] = module
    spec.loader.exec_module(module)
    return module


serialize = _load_serialize()
Skill = sys.modules["ha_agent.skills.models"].Skill


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

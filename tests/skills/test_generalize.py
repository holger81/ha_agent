"""Tests for skill generalization / merge clustering."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


def _load_generalize():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in ("models", "tool_names"):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    if "ha_agent.skills.observer" not in sys.modules:
        stub = types.ModuleType("ha_agent.skills.observer")

        def is_discovery_tool(tool_name: str) -> bool:
            lowered = (tool_name or "").lower()
            return any(
                token in lowered
                for token in (
                    "searchtoolsfordomain",
                    "searchtool",
                    "tools/list",
                    "tools_list",
                )
            )

        stub.is_discovery_tool = is_discovery_tool  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills.observer"] = stub

    mod_name = "ha_agent.skills.generalize"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    path = COMPONENT / "skills" / "generalize.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


generalize = _load_generalize()
Skill = sys.modules["ha_agent.skills.models"].Skill


def _skill(
    skill_id: str,
    *,
    title: str,
    tools: list[str],
    entity_ids: list[str] | None = None,
    route_scope: str | None = "action",
    use_count: int = 0,
    enabled: bool = True,
    is_builtin: bool = False,
    triggers: list[str] | None = None,
) -> Skill:
    entity_ids = entity_ids or []
    steps = []
    for index, tool in enumerate(tools):
        args: dict = {}
        if index < len(entity_ids):
            args["entity_id"] = entity_ids[index]
        steps.append({"toolName": tool, "arguments": args})
    return Skill(
        id=skill_id,
        slug=skill_id,
        title=title,
        description=f"Control {title}",
        triggers=triggers or [f"turn on {title}"],
        body=f"# {title}\n\nCall the control tool.",
        tool_steps=steps,
        route_scope=route_scope,
        use_count=use_count,
        enabled=enabled,
        is_builtin=is_builtin,
        score=1.0,
        created_at=float(len(skill_id)),
    )


def test_cluster_exact_fingerprint_merges_same_tools() -> None:
    a = _skill(
        "a",
        title="Dining lights on",
        tools=["HassTurnOn"],
        entity_ids=["light.dining"],
        use_count=2,
    )
    b = _skill(
        "b",
        title="Kitchen lights on",
        tools=["HassTurnOn"],
        entity_ids=["light.kitchen"],
        use_count=5,
    )
    c = _skill(
        "c",
        title="Fan on",
        tools=["HassTurnOn", "HassFanSetSpeed"],
        entity_ids=["fan.office"],
    )
    clusters = generalize.cluster_skills_for_generalize([a, b, c])
    assert len(clusters) == 1
    cluster = clusters[0]
    assert set(cluster.skill_ids) == {"a", "b"}
    assert cluster.survivor_id == "b"
    assert "entity_id" in {slot.name for slot in cluster.draft.slots}
    assert cluster.draft.tool_steps[0]["arguments"]["entity_id"] == "{{entity_id}}"


def test_cluster_skips_builtin_and_disabled() -> None:
    a = _skill("a", title="A", tools=["HassTurnOn"], entity_ids=["light.a"])
    builtin = _skill(
        "builtin",
        title="Built-in",
        tools=["HassTurnOn"],
        entity_ids=["light.b"],
        is_builtin=True,
    )
    disabled = _skill(
        "off",
        title="Off",
        tools=["HassTurnOn"],
        entity_ids=["light.c"],
        enabled=False,
    )
    assert generalize.cluster_skills_for_generalize([a, builtin, disabled]) == []


def test_near_miss_jaccard_cluster() -> None:
    # Exact fingerprints differ, but Jaccard(A,B)=4/5=0.8 meets threshold.
    a = _skill(
        "a",
        title="A",
        tools=["t1", "t2", "t3", "t4"],
        use_count=3,
    )
    b = _skill(
        "b",
        title="B",
        tools=["t1", "t2", "t3", "t4", "t5"],
        use_count=1,
    )
    clusters = generalize.cluster_skills_for_generalize([a, b])
    assert len(clusters) == 1
    assert "near" in clusters[0].key
    assert set(clusters[0].skill_ids) == {"a", "b"}


def test_build_generalized_draft_unions_triggers() -> None:
    a = _skill(
        "a",
        title="Dining",
        tools=["HassTurnOn"],
        entity_ids=["light.dining"],
        triggers=["dining lights", "turn on dining"],
        use_count=1,
    )
    b = _skill(
        "b",
        title="Kitchen",
        tools=["HassTurnOn"],
        entity_ids=["light.kitchen"],
        triggers=["kitchen lights", "dining lights"],
        use_count=4,
    )
    draft = generalize.build_generalized_draft([a, b])
    assert draft.title.startswith("Generalized")
    assert "dining lights" in draft.triggers
    assert "kitchen lights" in draft.triggers
    assert any(slot.name == "entity_id" for slot in draft.slots)


def test_cluster_to_dict_shape() -> None:
    a = _skill("a", title="A", tools=["HassTurnOn"], entity_ids=["light.a"])
    b = _skill(
        "b",
        title="B",
        tools=["HassTurnOn"],
        entity_ids=["light.b"],
        use_count=2,
    )
    cluster = generalize.cluster_skills_for_generalize([a, b])[0]
    payload = generalize.cluster_to_dict(cluster)
    assert payload["survivor_id"] == "b"
    assert "draft" in payload
    assert payload["draft"]["title"]
    assert isinstance(payload["draft"]["slots"], list)

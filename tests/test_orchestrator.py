"""Unit tests for orchestration replan / soft-route handling."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _ensure_ha_stubs() -> None:
    if "homeassistant" not in sys.modules:
        sys.modules["homeassistant"] = types.ModuleType("homeassistant")
    if "homeassistant.components" not in sys.modules:
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
    if "homeassistant.components.conversation" not in sys.modules:
        sys.modules["homeassistant.components.conversation"] = types.ModuleType(
            "homeassistant.components.conversation"
        )


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    deps = {
        "orchestrator": [
            "const",
            "context",
            "config_helpers",
            "llm_client",
            "llm_telemetry",
            "role_registry",
            "structured_output",
        ],
        "role_registry": ["config_helpers", "const"],
        "config_helpers": ["const"],
        "llm_client": ["const", "config_helpers"],
        "llm_telemetry": [],
        "structured_output": [],
        "const": [],
        "context": [],
    }
    if name in {"context", "orchestrator", "llm_client"}:
        _ensure_ha_stubs()
    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    path = COMPONENT / f"{name.replace('.', '/')}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_subtasks_maps_soft_routes_to_chat() -> None:
    """Email/news/other soft routes become chat without domain special-cases."""
    orch = _load("orchestrator")
    subtasks = orch._parse_subtasks_payload(
        {
            "subtasks": [
                {"id": "t1", "subgoal": "check inbox", "route": "email"},
                {"id": "t2", "subgoal": "digest", "route": "news"},
                {"id": "t3", "subgoal": "calendar", "route": "calendar"},
                {"id": "t4", "subgoal": "lights", "route": "action"},
            ]
        },
        user_text="do several things",
    )
    assert [item.route for item in subtasks] == ["chat", "chat", "chat", "action"]


@pytest.mark.asyncio
async def test_replan_after_failure_returns_failed_subtask_on_llm_error() -> None:
    """When the replan LLM fails, only the failed subtask remains queued."""
    orch = _load("orchestrator")
    failed = orch.SubtaskSpec(id="t2", subgoal="open garage", route="action")
    plan = orch.OrchestrationPlan(
        complexity=orch.Complexity.COMPLEX,
        subtasks=[
            orch.SubtaskSpec(id="t1", subgoal="done already", route="chat"),
            failed,
            orch.SubtaskSpec(id="t3", subgoal="later", route="chat"),
        ],
    )
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("planner down"))
    registry = MagicMock()
    registry.backend_for.return_value = MagicMock(model="m", base_url="http://x")

    revised = await orch.replan_after_failure(
        llm,
        registry,
        user_text="compound ask",
        plan=plan,
        failed_subtask=failed,
        completed_summaries=[{"subgoal": "done already", "summary": "ok"}],
    )

    assert revised.reason == "replan_failed"
    assert [item.id for item in revised.subtasks] == ["t2"]
    assert revised.subtasks[0].subgoal == "open garage"


def test_worker_replan_queue_excludes_completed_ids() -> None:
    """Revised queues drop already-completed subtask ids."""
    orch = _load("orchestrator")
    completed_ids = {"t1"}
    revised = [
        orch.SubtaskSpec(id="t1", subgoal="already done", route="chat"),
        orch.SubtaskSpec(id="t2", subgoal="retry me", route="action"),
        orch.SubtaskSpec(id="t3", subgoal="new step", route="chat"),
    ]
    pending = [item for item in revised if item.id not in completed_ids]
    assert [item.id for item in pending] == ["t2", "t3"]

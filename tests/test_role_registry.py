"""Tests for role registry collapse on single-machine setups."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    deps = {"role_registry": ["config_helpers"], "config_helpers": ["const"]}
    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)
    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


role_registry = _load("role_registry")
config_helpers = _load("config_helpers")


def test_collapse_identical_roles() -> None:
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="same-model",
        api_key=None,
        max_tokens=256,
        temperature=0.0,
        timeout=30,
        thinking_level="off",
    )
    registry = role_registry.RoleRegistry(
        chat_backend=backend,
        roles={
            role_registry.ModelRole.ROUTER: backend,
            role_registry.ModelRole.PLANNER: backend,
            role_registry.ModelRole.VERIFIER: backend,
            role_registry.ModelRole.OBSERVER: backend,
            role_registry.ModelRole.WORKER_CHAT: backend,
            role_registry.ModelRole.WORKER_ACTION: backend,
            role_registry.ModelRole.WORKER_EMAIL: backend,
            role_registry.ModelRole.WORKER_NEWS: backend,
        },
    )
    collapsed = role_registry.collapse_identical_roles(registry)
    assert collapsed.backend_for(role_registry.ModelRole.PLANNER) is backend
    assert collapsed.backend_for(role_registry.ModelRole.VERIFIER) is backend

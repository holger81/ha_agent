"""Unit tests for editable recovery hints."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_recovery_hints():
    mod_name = "ha_agent.recovery_hints"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistant:
            pass

        def callback(func):
            return func

        ha_core.HomeAssistant = HomeAssistant
        ha_core.callback = callback
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.core"] = ha_core

    if "ha_agent.const" not in sys.modules:
        const_spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert const_spec and const_spec.loader
        const_mod = importlib.util.module_from_spec(const_spec)
        sys.modules["ha_agent.const"] = const_mod
        const_spec.loader.exec_module(const_mod)

    path = COMPONENT / "recovery_hints.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _store(tmp_path):
    rh = _load_recovery_hints()
    store = rh.RecoveryHintStore(tmp_path / "recovery_hints.db")
    store.connect()
    return rh, store


def test_seeds_default_hints(tmp_path) -> None:
    """Connecting seeds the four built-in recovery hints."""
    rh, store = _store(tmp_path)
    try:
        hints = store.list_hints()
        rule_ids = {h.rule_id for h in hints}
        expected = {rule["rule_id"] for rule in rh.DEFAULT_RECOVERY_HINTS}
        assert expected.issubset(rule_ids)
        assert all(h.enabled and h.is_default for h in hints if h.is_builtin)
        assert store.custom_count() == 0
    finally:
        store.close()


def test_update_marks_customized(tmp_path) -> None:
    """Editing a built-in body flips is_default off."""
    _rh, store = _store(tmp_path)
    try:
        updated = store.update_hint("news_curate", body="Try news_curate first.")
        assert updated is not None
        assert updated.body == "Try news_curate first."
        assert updated.is_default is False
    finally:
        store.close()


def test_disabled_hint_excluded_from_enabled(tmp_path) -> None:
    """A disabled hint drops out of the runtime list."""
    _rh, store = _store(tmp_path)
    try:
        store.update_hint("mcp_down", enabled=False)
        enabled_ids = {h.rule_id for h in store.list_enabled()}
        assert "mcp_down" not in enabled_ids
    finally:
        store.close()


def test_reset_restores_default(tmp_path) -> None:
    """Reset returns a built-in hint to its shipped default and enables it."""
    rh, store = _store(tmp_path)
    try:
        store.update_hint("news_curate", body="changed", enabled=False)
        reset = store.reset_hint("news_curate")
        assert reset is not None
        assert reset.enabled is True
        assert reset.is_default is True
        default = rh._DEFAULT_BY_ID["news_curate"]
        assert reset.body == default["body"]
    finally:
        store.close()


def test_create_and_delete_custom_hint(tmp_path) -> None:
    """Custom rules can be added and removed; built-ins cannot be deleted."""
    _rh, store = _store(tmp_path)
    try:
        created = store.create_hint(
            title="Calendar",
            body="Retry with an ISO date range.",
            tool_substring="calendar",
            error_pattern="date",
        )
        assert created.is_builtin is False
        assert created.rule_id.startswith("custom-")
        assert store.custom_count() == 1

        ids = [h.rule_id for h in store.list_hints()]
        assert created.rule_id in ids

        assert store.delete_hint("news_curate") is False  # built-in protected
        assert store.delete_hint(created.rule_id) is True
        assert store.custom_count() == 0
    finally:
        store.close()


def test_active_hints_thread_into_enrich(tmp_path) -> None:
    """Enabled store rules drive loop_policy.enrich_tool_output."""
    _rh, store = _store(tmp_path)
    try:
        store.create_hint(
            title="Calendar",
            body="Retry with an ISO date range.",
            tool_substring="calendar",
            error_pattern="invalid date",
        )
        rules = store.list_enabled()
        policy = _load_loop_policy()
        output = policy.enrich_tool_output(
            "calendar_mcp__create_event",
            {},
            "Tool error: invalid date format",
            rules=rules,
        )
        assert "RECOVERY HINTS" in output
        assert "ISO date range" in output
    finally:
        store.close()


def _load_loop_policy():
    mod_name = "ha_agent.loop_policy"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    path = COMPONENT / "loop_policy.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

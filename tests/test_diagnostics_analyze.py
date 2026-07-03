"""Unit tests for turn diagnostics analysis."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_analyze():
    mod_name = "ha_agent.diagnostics.analyze"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    diag_pkg = types.ModuleType("ha_agent.diagnostics")
    diag_pkg.__path__ = [str(COMPONENT / "diagnostics")]  # type: ignore[attr-defined]
    sys.modules["ha_agent.diagnostics"] = diag_pkg

    path = COMPONENT / "diagnostics" / "analyze.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


analyze_mod = _load_analyze()
analyze_turn_dict = analyze_mod.analyze_turn_dict


def test_analyze_turn_detects_tool_param_error() -> None:
    turn = {
        "user_text": "any new emails?",
        "assistant_text": "Sorry",
        "tool_errors": 1,
        "tool_calls": [
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "succeeded": False,
                "error_kind": "param",
                "missing_fields": ["mailbox"],
                "error": "Tool error: missing field 'mailbox'",
            }
        ],
        "conversation_id": "assist-1",
    }
    result = analyze_turn_dict(turn)
    assert result["severity"] == "error"
    assert any(issue["kind"] == "tool_call_failed" for issue in result["issues"])
    assert any(
        action["action"] == "inspect_skill_repair"
        for action in result["suggested_actions"]
    )


def test_analyze_turn_ok_for_clean_success() -> None:
    turn = {
        "user_text": "turn on the light",
        "assistant_text": "Done.",
        "tool_errors": 0,
        "outcome": "success",
        "verifier_verdict": "pass",
        "tool_calls": [
            {"toolName": "home_assistant__ha_call_service", "succeeded": True}
        ],
        "timestamp": 1710000000.0,
        "conversation_id": "console-1",
    }
    result = analyze_turn_dict(turn)
    assert result["severity"] == "ok"
    assert any(
        action["action"] == "promote_eval_case"
        for action in result["suggested_actions"]
    )

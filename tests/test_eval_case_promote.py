"""Tests for promoting activity turns to eval cases."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_stubs() -> None:
    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistantError(Exception):
            pass

        ha_core.HomeAssistant = object
        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc
        sys.modules["homeassistant.core"] = ha_core


def _load(name: str, path: Path):
    module_name = f"ha_agent.{name.replace('/', '.')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    _ensure_ha_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eval_models = _load("eval.models", COMPONENT / "eval" / "models.py")
skills_models = _load("skills.models", COMPONENT / "skills" / "models.py")
case_promote = _load("eval.case_promote", COMPONENT / "eval" / "case_promote.py")
case_serde = _load("eval.case_serde", COMPONENT / "eval" / "case_serde.py")
eval_cases = _load("eval.cases", COMPONENT / "eval" / "cases.py")
eval_store = _load("eval.store", COMPONENT / "eval" / "store.py")
eval_scorer = _load("eval.scorer", COMPONENT / "eval" / "scorer.py")


def test_build_case_from_action_turn() -> None:
    trace = skills_models.TurnTrace(
        user_text="turn off the dining room lights",
        history_len=2,
        route="action",
        exposed_entities=[
            {
                "entity_id": "light.dining",
                "name": "Dining",
                "state": "on",
            }
        ],
        tool_calls=[
            {
                "toolName": "mcp-tools-admin__searchToolsForDomain",
                "arguments": {"domain": "smart-home"},
            },
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.dining",
                },
            },
        ],
        assistant_text="The dining room lights are off.",
        outcome="success",
        iterations=2,
    )
    case = case_promote.build_case_from_turn(trace, source_timestamp=1_700_000_000.0)
    assert case.source == "promoted"
    assert case.task == "action"
    assert case.expected_tool == "home_assistant__ha_call_service"
    assert case.expected_tool_args == {
        "domain": "light",
        "service": "turn_off",
        "entity_id": "light.dining",
    }
    assert "dining" in case.expected_text_contains
    assert len(case.mock_mcp_responses) == 2


def test_build_case_rejects_fallback() -> None:
    trace = skills_models.TurnTrace(
        user_text="hello",
        history_len=0,
        route="chat",
        fallback=True,
        assistant_text="Sorry",
        outcome="success",
    )
    with pytest.raises(Exception, match="fallback"):
        case_promote.build_case_from_turn(trace)


def test_custom_case_round_trip_store() -> None:
    trace = skills_models.TurnTrace(
        user_text="what is the weather",
        history_len=0,
        route="chat",
        assistant_text="It is sunny today.",
        outcome="success",
    )
    case = case_promote.build_case_from_turn(trace, case_id="promoted-test")
    with tempfile.TemporaryDirectory() as tmp:
        store = eval_store.EvalStore(Path(tmp) / "eval.db")
        store.connect()
        saved = store.save_custom_case(case)
        loaded = store.get_custom_case("promoted-test")
        assert loaded is not None
        assert loaded.user_text == saved.user_text
        assert store.delete_custom_case("promoted-test")
        assert store.get_custom_case("promoted-test") is None
        store.close()


def test_promoted_case_scores_against_matching_trace() -> None:
    trace = skills_models.TurnTrace(
        user_text="What's the news?",
        history_len=0,
        route="news",
        tool_calls=[
            {
                "toolName": "mcp_news__news_curate",
                "arguments": {},
            }
        ],
        assistant_text="Here are today's headlines.",
        outcome="success",
    )
    case = case_promote.build_case_from_turn(trace, case_id="promoted-news")
    score = eval_scorer.score_case(
        case,
        model="test-model",
        trace=trace,
        latency_ms=500.0,
    )
    assert score.passed is True
    assert score.score >= 0.9


def test_list_eval_cases_includes_custom() -> None:
    custom = eval_models.EvalCase(
        id="promoted-custom",
        task="chat",
        user_text="custom prompt",
        source="promoted",
    )
    cases = eval_cases.list_eval_cases(custom_cases=[custom])
    assert any(case.id == "promoted-custom" for case in cases)
    assert any(case.id == "chat_weather" for case in cases)


def test_case_serde_round_trip() -> None:
    case = eval_models.EvalCase(
        id="promoted-1",
        task="email",
        user_text="check mail",
        expected_tool="mail_mcp__imap_search_messages",
        source="promoted",
        promoted_at=123.0,
    )
    payload = case_serde.eval_case_to_dict(case)
    restored = case_serde.eval_case_from_dict(payload)
    assert restored.id == case.id
    assert restored.task == case.task
    assert restored.source == "promoted"

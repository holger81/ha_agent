"""Unit tests for the eval system."""

from __future__ import annotations

import importlib.util
import json
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

        def callback(func):
            return func

        ha_core.HomeAssistant = object
        ha_core.callback = callback
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


config_helpers = _load("config_helpers", COMPONENT / "config_helpers.py")
thinking = _load("thinking", COMPONENT / "thinking.py")
embedded_tools = _load("embedded_tools", COMPONENT / "embedded_tools.py")
llm_client = _load("llm_client", COMPONENT / "llm_client.py")
llm_server = _load("llm_server", COMPONENT / "llm_server.py")
eval_models = _load("eval.models", COMPONENT / "eval" / "models.py")
eval_cases = _load("eval.cases", COMPONENT / "eval" / "cases.py")
eval_scorer = _load("eval.scorer", COMPONENT / "eval" / "scorer.py")
eval_store = _load("eval.store", COMPONENT / "eval" / "store.py")
eval_recommender = _load("eval.recommender", COMPONENT / "eval" / "recommender.py")
skills_models = _load("skills.models", COMPONENT / "skills" / "models.py")


def test_server_root_from_base_url() -> None:
    assert (
        llm_server.server_root_from_base_url("http://192.168.10.31:9292/v1")
        == "http://192.168.10.31:9292"
    )
    assert (
        llm_server.server_root_from_base_url("http://example:8080/v1/")
        == "http://example:8080"
    )


def test_parse_prometheus_metrics() -> None:
    text = (
        "# comment\n"
        'llamacpp:prompt_tokens_total{host="x"} 42\n'
        "llamacpp:tokens_predicted_total 100\n"
    )
    parsed = llm_server._parse_prometheus(text)
    assert parsed["llamacpp:prompt_tokens_total"] == 42.0
    assert parsed["llamacpp:tokens_predicted_total"] == 100.0


def test_list_eval_cases_filters_tasks() -> None:
    cases = eval_cases.list_eval_cases(tasks=["news", "email"])
    assert {case.task for case in cases} == {"news", "email"}


def test_builtin_case_counts() -> None:
    action_cases = eval_cases.cases_for_task("action")
    chat_cases = eval_cases.cases_for_task("chat")
    assert len(action_cases) == 6
    assert len(chat_cases) == 5
    assert len(eval_cases.list_eval_cases()) == 16


def test_score_case_passes_when_tool_and_text_match() -> None:
    case = eval_cases.list_eval_cases(tasks=["news"])[0]
    trace = skills_models.TurnTrace(
        user_text=case.user_text,
        history_len=0,
        tool_calls=[
            {
                "toolName": "mcp_news__news_curate",
                "name": "mcp_news__news_curate",
                "arguments": {"limit": 5},
            }
        ],
        assistant_text="Here are today's headlines.",
        outcome="success",
    )
    score = eval_scorer.score_case(
        case,
        model="test-model",
        trace=trace,
        latency_ms=1200.0,
    )
    assert score.passed is True
    assert score.score >= 0.9


def test_aggregate_task_scores() -> None:
    scores = [
        eval_models.EvalCaseScore(
            case_id="a",
            task="chat",
            model="m1",
            score=0.8,
            passed=True,
            latency_ms=100.0,
        ),
        eval_models.EvalCaseScore(
            case_id="b",
            task="chat",
            model="m1",
            score=0.6,
            passed=False,
            latency_ms=200.0,
        ),
    ]
    task_scores = eval_scorer.aggregate_task_scores(scores)
    assert len(task_scores) == 1
    assert task_scores[0].score == pytest.approx(0.7)
    assert task_scores[0].passed_count == 1


def test_build_settings_recommendation_merges_benchmarks() -> None:
    caps = llm_server.ServerCapabilities(server_root="http://example:9292")
    task_scores = [
        eval_models.EvalTaskScore(
            task="chat",
            model="gemma",
            score=0.9,
            case_count=1,
            passed_count=1,
        )
    ]
    recommendation = eval_recommender.build_settings_recommendation(
        capabilities=caps,
        task_scores=task_scores,
        llm_content=json.dumps(
            {
                "summary": "Use one slot for voice latency.",
                "recommendations": [
                    {
                        "setting": "parallel",
                        "value": "1",
                        "reason": "Voice agent is single-user.",
                    }
                ],
                "warnings": [],
                "model_assignments": {
                    "chat": {"model": "gemma", "reason": "Best score."},
                },
            }
        ),
    )
    assert recommendation.model_assignments["chat"]["model"] == "gemma"
    assert recommendation.recommendations[0]["setting"] == "parallel"


def test_eval_candidate_models_prefers_loaded() -> None:
    caps = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        models=["a", "b", "c"],
        loaded_models=["a"],
    )
    selected = llm_server.eval_candidate_models(
        caps,
        configured_models=["b"],
    )
    assert selected == ["b", "a"]


def test_recommendations_to_preset() -> None:
    preset_mod = _load("eval.preset", COMPONENT / "eval" / "preset.py")
    preset = preset_mod.recommendations_to_preset(
        [
            {"setting": "parallel", "value": "2", "reason": "Single user"},
            {"setting": "ctx-size", "value": "65536", "reason": "Long context"},
        ]
    )
    assert "parallel = 2" in preset
    assert "ctx-size = 65536" in preset


def test_build_host_context_from_loaded_model() -> None:
    host_mod = _load("eval.host_context", COMPONENT / "eval" / "host_context.py")
    caps = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        router_role="router",
        max_instances=3,
        model_details=[
            llm_server.ModelInfo(
                model_id="gemma",
                status="loaded",
                raw={
                    "status": {
                        "value": "loaded",
                        "args": ["--parallel", "2", "--n-gpu-layers", "99"],
                    }
                },
            )
        ],
    )
    context = host_mod.build_host_context(caps)
    assert context["loaded_model_args"] == ["--parallel", "2", "--n-gpu-layers", "99"]
    assert context["max_instances"] == 3


def test_server_apply_mode_router_uses_preset() -> None:
    server_apply = _load("eval.server_apply", COMPONENT / "eval" / "server_apply.py")
    caps = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        router_role="router",
        props_writable=False,
    )
    assert server_apply.server_apply_mode(caps) == "preset"


def test_server_apply_mode_props_when_writable() -> None:
    server_apply = _load("eval.server_apply", COMPONENT / "eval" / "server_apply.py")
    caps = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        props_writable=True,
    )
    assert server_apply.server_apply_mode(caps) == "props"


def test_verify_settings_applied_detects_ctx_change() -> None:
    server_apply = _load("eval.server_apply", COMPONENT / "eval" / "server_apply.py")
    before = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        props=llm_server.ServerProps(n_ctx=8192),
    )
    after = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        props=llm_server.ServerProps(n_ctx=65536),
    )
    result = server_apply.verify_settings_applied(
        before,
        after,
        {"ctx-size": "65536"},
    )
    assert result["verified_count"] == 1
    assert result["all_verified"] is True


def test_probe_setting_value_reads_parallel() -> None:
    server_apply = _load("eval.server_apply", COMPONENT / "eval" / "server_apply.py")
    caps = llm_server.ServerCapabilities(
        server_root="http://example:9292",
        max_instances=3,
        props=llm_server.ServerProps(total_slots=2),
    )
    assert server_apply.probe_setting_value(caps, "parallel") == 2
    assert server_apply.probe_setting_value(caps, "n_parallel") == 2


def test_eval_store_persists_runs_and_download_history() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "eval.db"
        store = eval_store.EvalStore(db_path)
        store.connect()
        run = store.create_run("entry-1")
        run.status = "completed"
        run.server_capabilities = {"models": ["gemma"]}
        run.settings_recommendation = {"summary": "ok"}
        run.task_scores = [
            eval_models.EvalTaskScore(
                task="chat",
                model="gemma",
                score=1.0,
                case_count=1,
                passed_count=1,
            )
        ]
        run.case_scores = [
            eval_models.EvalCaseScore(
                case_id="news_headlines",
                task="news",
                model="gemma",
                score=1.0,
                passed=True,
            )
        ]
        store.finish_run(run)
        assert store.has_benchmarked_model("gemma") is True
        store.record_model_download("new-model", source_url="https://example/model")
        assert store.should_skip_download("new-model") is False
        store.mark_model_deleted("new-model")
        assert store.should_skip_download("new-model") is True
        latest = store.latest_run()
        assert latest is not None
        assert latest["status"] == "completed"
        store.close()

"""LLM reasoning about llama.cpp settings and per-task model selection."""

from __future__ import annotations

import json
import re
from typing import Any

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from ..llm_server import ServerCapabilities
from .models import EvalTaskScore, SettingsRecommendation
from .scorer import best_model_per_task

_SETTINGS_PROMPT = (
    "You analyze llama.cpp server capabilities and recommend optimal server "
    "settings for a Home Assistant voice agent.\n\n"
    "The agent runs these task routes, each may use a different model:\n"
    "- chat: general assistant Q&A\n"
    "- action: device control (lights, covers, locks)\n"
    "- email: unread mail queries via MCP\n"
    "- news: headline briefing via MCP\n"
    "- classifier: route/playbook selection (small, fast model)\n\n"
    "Server capabilities:\n"
    "{capabilities_json}\n\n"
    "Benchmark scores (model -> task -> score 0..1, higher is better):\n"
    "{benchmark_json}\n\n"
    "Recommend llama.cpp server settings (parallel slots, ctx-size, batch-size, "
    "threads, n-gpu-layers, cache-reuse, etc.) for this hardware and workload.\n"
    "Also recommend which loaded model to assign to each task.\n\n"
    "Return ONLY JSON:\n"
    "{{\n"
    '  "summary": "short overview",\n'
    '  "recommendations": [\n'
    '    {{"setting": "parallel", "value": "2", "reason": "..."}}\n'
    "  ],\n"
    '  "warnings": ["..."],\n'
    '  "model_assignments": {{\n'
    '    "chat": {{"model": "...", "reason": "..."}},\n'
    '    "action": {{"model": "...", "reason": "..."}},\n'
    '    "email": {{"model": "...", "reason": "..."}},\n'
    '    "news": {{"model": "...", "reason": "..."}},\n'
    '    "classifier": {{"model": "...", "reason": "..."}}\n'
    "  }}\n"
    "}}"
)


def _extract_json(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_assignments(
    task_scores: list[EvalTaskScore],
) -> dict[str, dict[str, str]]:
    best = best_model_per_task(task_scores)
    return {
        task: {
            "model": model,
            "reason": "Highest benchmark score for this task.",
        }
        for task, model in best.items()
    }


def build_settings_recommendation(
    *,
    capabilities: ServerCapabilities,
    task_scores: list[EvalTaskScore],
    llm_content: str | None,
) -> SettingsRecommendation:
    """Parse LLM output and merge with benchmark winners."""
    parsed = _extract_json(llm_content or "") if llm_content else None
    fallback_assignments = _fallback_assignments(task_scores)
    if not parsed:
        return SettingsRecommendation(
            summary=(
                "Benchmark scores selected models per task; "
                "server tuning recommendation unavailable."
            ),
            recommendations=[],
            warnings=["LLM settings analysis did not return valid JSON."],
            model_assignments=fallback_assignments,
            raw={},
        )

    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        recommendations = []
    warnings = parsed.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    assignments = parsed.get("model_assignments")
    if not isinstance(assignments, dict):
        assignments = {}
    merged_assignments = dict(fallback_assignments)
    for task, payload in assignments.items():
        if not isinstance(payload, dict):
            continue
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            merged_assignments[task] = {
                "model": model.strip(),
                "reason": str(payload.get("reason") or "Recommended by eval agent."),
            }

    return SettingsRecommendation(
        summary=str(
            parsed.get("summary") or "Eval-based server tuning recommendation."
        ),
        recommendations=[
            {
                "setting": str(item.get("setting") or ""),
                "value": str(item.get("value") or ""),
                "reason": str(item.get("reason") or ""),
            }
            for item in recommendations
            if isinstance(item, dict) and item.get("setting")
        ],
        warnings=[str(item) for item in warnings],
        model_assignments=merged_assignments,
        raw=parsed,
    )


async def recommend_settings(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    capabilities: ServerCapabilities,
    task_scores: list[EvalTaskScore],
) -> SettingsRecommendation:
    """Use the agent LLM to reason about server settings and model mapping."""
    benchmark_json = json.dumps(
        [
            {
                "task": item.task,
                "model": item.model,
                "score": round(item.score, 3),
                "passed": item.passed_count,
                "cases": item.case_count,
                "avg_latency_ms": item.avg_latency_ms,
            }
            for item in task_scores
        ],
        ensure_ascii=False,
    )
    prompt = _SETTINGS_PROMPT.format(
        capabilities_json=json.dumps(capabilities.summary(), ensure_ascii=False),
        benchmark_json=benchmark_json,
    )
    try:
        result = await llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a llama.cpp performance engineer for Home Assistant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            LlmBackend(
                base_url=backend.base_url,
                model=backend.model,
                api_key=backend.api_key,
                max_tokens=1024,
                temperature=0.1,
                timeout=backend.timeout,
                thinking_level="off",
            ),
        )
    except Exception as err:
        LOGGER.warning("Eval settings recommendation failed: %s", err)
        return build_settings_recommendation(
            capabilities=capabilities,
            task_scores=task_scores,
            llm_content=None,
        )

    return build_settings_recommendation(
        capabilities=capabilities,
        task_scores=task_scores,
        llm_content=result.content,
    )


def settings_recommendation_to_dict(
    recommendation: SettingsRecommendation,
) -> dict[str, Any]:
    """Serialize a settings recommendation."""
    return {
        "summary": recommendation.summary,
        "recommendations": list(recommendation.recommendations),
        "warnings": list(recommendation.warnings),
        "model_assignments": dict(recommendation.model_assignments),
        "raw": dict(recommendation.raw),
    }

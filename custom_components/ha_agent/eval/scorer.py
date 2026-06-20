"""Score eval benchmark results."""

from __future__ import annotations

from typing import Any

from ..skills.models import TurnTrace
from .models import EvalCase, EvalCaseScore, EvalTaskScore


def _tool_matches(trace: TurnTrace, case: EvalCase) -> tuple[bool, list[str]]:
    if not case.expected_tool:
        return True, []
    details: list[str] = []
    for call in trace.tool_calls:
        tool_name = str(
            call.get("toolName") or call.get("name") or call.get("tool_name") or ""
        )
        if case.expected_tool not in tool_name:
            continue
        if not case.expected_tool_args:
            return True, details
        args = call.get("arguments")
        if not isinstance(args, dict):
            continue
        inner = args.get("arguments")
        payload = inner if isinstance(inner, dict) else args
        mismatches = [
            f"{key} expected {case.expected_tool_args[key]!r}, got {payload.get(key)!r}"
            for key in case.expected_tool_args
            if payload.get(key) != case.expected_tool_args[key]
        ]
        if mismatches:
            details.extend(mismatches)
            continue
        return True, details
    details.append(f"expected tool containing {case.expected_tool!r}")
    return False, details


def _text_matches(trace: TurnTrace, case: EvalCase) -> tuple[bool, list[str]]:
    if not case.expected_text_contains:
        return True, []
    text = (trace.assistant_text or "").lower()
    missing = [
        token for token in case.expected_text_contains if token.lower() not in text
    ]
    if missing:
        return False, [f"missing text tokens: {', '.join(missing)}"]
    return True, []


def score_case(
    case: EvalCase,
    *,
    model: str,
    trace: TurnTrace,
    latency_ms: float | None,
) -> EvalCaseScore:
    """Score one benchmark case result."""
    details: list[str] = []
    tool_match, tool_details = _tool_matches(trace, case)
    details.extend(tool_details)
    text_match, text_details = _text_matches(trace, case)
    details.extend(text_details)

    outcome_ok = trace.outcome in {"success", "partial", ""}
    if trace.tool_errors:
        details.append(f"{trace.tool_errors} tool error(s)")
    if trace.fallback:
        details.append("fallback response used")
    if trace.iterations >= case.max_iterations and not trace.assistant_text:
        details.append("hit iteration limit without final text")

    passed = tool_match and text_match and outcome_ok and not trace.fallback
    score = 0.0
    if tool_match:
        score += 0.5
    if text_match:
        score += 0.3
    if outcome_ok and not trace.fallback:
        score += 0.2
    if trace.tool_errors:
        score -= 0.1
    score = max(0.0, min(1.0, score))

    return EvalCaseScore(
        case_id=case.id,
        task=case.task,
        model=model,
        score=score,
        passed=passed,
        latency_ms=latency_ms,
        iterations=trace.iterations,
        outcome=trace.outcome,
        tool_match=tool_match,
        text_match=text_match,
        details=details,
    )


def aggregate_task_scores(case_scores: list[EvalCaseScore]) -> list[EvalTaskScore]:
    """Aggregate per-model scores by task."""
    grouped: dict[tuple[str, str], list[EvalCaseScore]] = {}
    for item in case_scores:
        grouped.setdefault((item.task, item.model), []).append(item)

    results: list[EvalTaskScore] = []
    for (task, model), items in sorted(grouped.items()):
        latencies = [item.latency_ms for item in items if item.latency_ms is not None]
        passed_count = sum(1 for item in items if item.passed)
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        results.append(
            EvalTaskScore(
                task=task,
                model=model,
                score=sum(item.score for item in items) / len(items),
                case_count=len(items),
                passed_count=passed_count,
                avg_latency_ms=avg_latency,
            )
        )
    return results


def best_model_per_task(task_scores: list[EvalTaskScore]) -> dict[str, str]:
    """Pick the highest-scoring model for each task."""
    best: dict[str, tuple[float, str]] = {}
    for item in task_scores:
        current = best.get(item.task)
        if current is None or item.score > current[0]:
            best[item.task] = (item.score, item.model)
    return {task: model for task, (_, model) in best.items()}


def case_score_to_dict(score: EvalCaseScore) -> dict[str, Any]:
    """Serialize one case score."""
    return {
        "case_id": score.case_id,
        "task": score.task,
        "model": score.model,
        "score": score.score,
        "passed": score.passed,
        "latency_ms": score.latency_ms,
        "iterations": score.iterations,
        "outcome": score.outcome,
        "tool_match": score.tool_match,
        "text_match": score.text_match,
        "details": list(score.details),
    }


def task_score_to_dict(score: EvalTaskScore) -> dict[str, Any]:
    """Serialize one task aggregate score."""
    return {
        "task": score.task,
        "model": score.model,
        "score": score.score,
        "case_count": score.case_count,
        "passed_count": score.passed_count,
        "avg_latency_ms": score.avg_latency_ms,
    }

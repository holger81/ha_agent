"""Record per-call LLM telemetry on turn traces."""

from __future__ import annotations

from typing import Any

from .config_helpers import LlmBackend
from .llm_client import ChatResult
from .skills.models import TurnTrace


def record_llm_call(
    trace: TurnTrace | None,
    *,
    role: str,
    backend: LlmBackend,
    result: ChatResult | None = None,
    error: str | None = None,
) -> None:
    """Append one LLM call record to the active turn trace."""
    if trace is None:
        return
    entry: dict[str, Any] = {
        "role": role,
        "model": backend.model,
        "host": backend.base_url.split("//", 1)[-1].split("/", 1)[0],
    }
    if result is not None:
        if result.latency_ms is not None:
            entry["latency_ms"] = round(result.latency_ms, 1)
        if result.prompt_tokens is not None:
            entry["prompt_tokens"] = result.prompt_tokens
        if result.completion_tokens is not None:
            entry["completion_tokens"] = result.completion_tokens
    if error:
        entry["error"] = error[:240]
    trace.llm_calls.append(entry)

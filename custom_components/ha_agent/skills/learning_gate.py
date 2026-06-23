"""LLM gate for whether a successful turn should become a learned skill."""

from __future__ import annotations

from ..config_helpers import LlmBackend
from ..llm_client import LlmClient
from .models import TurnTrace
from .observer import observe_skill_candidate


async def assess_skill_worth_learning(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    trace: TurnTrace,
    history: list[dict[str, str]],
    manual_save: bool = False,
) -> bool:
    """Ask the skill observer whether this turn merits skill creation."""
    result = await observe_skill_candidate(
        llm,
        backend,
        trace=trace,
        history=history,
        manual_save=manual_save,
    )
    return result.learn

"""Verify answer stage of turn pipeline."""

from typing import Any, Tuple
from custom_components.ha_agent.skills.models import TurnTrace
from .pipeline import TurnContext, ExecutionTrace


def verify_answer(
    context: TurnContext,
    trace: ExecutionTrace,
    **kwargs: Any,
) -> tuple[TurnContext, ExecutionTrace]:
    """Verify the answer before release.
    
    This stage handles:
    - Deterministic verification first (control tool succeeded?)
    - LLM verifier only on ambiguity
    - Post-answer async learning/verification
    """
    # Merge with existing trace if provided
    if not isinstance(trace, ExecutionTrace):
        exec_trace = ExecutionTrace(context=context)
    else:
        exec_trace = trace
        
    # TODO: Integrate with verify_turn() logic
    # For now, return the context as-is
    
    exec_trace.add_decision("verify_answer: initial verification")
    
    return context, exec_trace

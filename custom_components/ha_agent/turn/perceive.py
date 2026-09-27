"""Perceive intent stage of turn pipeline."""

from typing import Any, Tuple
from custom_components.ha_agent.skills.models import TurnTrace
from .pipeline import TurnContext, ExecutionTrace


def perceive_intent(
    context: TurnContext,
    trace: TurnTrace,
    **kwargs: Any,
) -> tuple[TurnContext, ExecutionTrace]:
    """Perceive intent from user text and history.
    
    This stage handles:
    - Route classification (chat vs action)
    - Intent parsing
    - Prepass execution (route, complexity, skill, slots)
    """
    exec_trace = ExecutionTrace(context=context)
    
    # TODO: Integrate with run_turn_prepass() and resolve_route_with_classifier()
    # For now, return the context as-is
    
    exec_trace.add_decision("perceive_intent: initial route from context")
    
    return context, exec_trace

"""Retrieve tools and skills stage of turn pipeline."""

from typing import Any, Tuple
from custom_components.ha_agent.skills.models import TurnTrace
from .pipeline import TurnContext, ExecutionTrace


def retrieve_tools_skills(
    context: TurnContext,
    trace: ExecutionTrace,
    **kwargs: Any,
) -> tuple[TurnContext, ExecutionTrace]:
    """Retrieve tools and skills for the turn.
    
    This stage handles:
    - Skills selection via FTS5 + route_scope
    - Tool pruning
    - Skill parameter resolution
    """
    # Merge with existing trace if provided
    if not isinstance(trace, ExecutionTrace):
        exec_trace = ExecutionTrace(context=context)
    else:
        exec_trace = trace
        
    # TODO: Integrate with resolve_skills_for_turn() and prune_loop_tools()
    # For now, return the context as-is
    
    exec_trace.add_decision("retrieve_tools_skills: initial retrieval")
    
    return context, exec_trace

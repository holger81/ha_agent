"""Execute ReAct loop stage of turn pipeline."""

from typing import Any, Tuple
from custom_components.ha_agent.skills.models import TurnTrace
from .pipeline import TurnContext, ExecutionTrace, Observation


def execute_react_loop(
    context: TurnContext,
    trace: ExecutionTrace,
    **kwargs: Any,
) -> tuple[TurnContext, ExecutionTrace]:
    """Execute the bounded ReAct loop.
    
    This stage handles:
    - LLM call with tools
    - Tool execution
    - Loop continuation or answer generation
    - Orchestrator for complex turns
    """
    # Merge with existing trace if provided
    if not isinstance(trace, ExecutionTrace):
        exec_trace = ExecutionTrace(context=context)
    else:
        exec_trace = trace
        
    # TODO: Integrate with run_agent() loop logic
    # For now, return the context as-is
    
    exec_trace.add_decision("execute_react_loop: initial execution")
    
    return context, exec_trace

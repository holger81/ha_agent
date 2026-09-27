"""Turn pipeline orchestrator."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from custom_components.ha_agent.skills.models import TurnTrace
from .perceive import perceive_intent
from .retrieve import retrieve_tools_skills
from .execute import execute_react_loop
from .verify import verify_answer


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Frozen context for a turn."""
    user_text: str
    history: list[dict[str, str]]
    exposed_entities: list[dict]
    conversation_id: str | None
    identity: str | None = None
    route: str | None = None  # Route is mutable, stored here as initial route


@dataclass(slots=True)
class Observation:
    """Observation of a tool execution."""
    tool_name: str
    arguments: dict
    succeeded: bool
    output: str
    entity_id: str | None = None


@dataclass(slots=True)
class ExecutionTrace:
    """Append-only trace of turn execution."""
    context: TurnContext
    observations: list[Observation] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    route_history: list[str] = field(default_factory=list)
    
    def add_observation(self, obs: Observation):
        self.observations.append(obs)
        
    def add_decision(self, decision: str):
        self.decisions.append(decision)
        
    def update_route(self, new_route: str):
        if not self.route_history:
            self.route_history.append(self.context.route or "initial")
        self.route_history.append(new_route)
        self.context.route = new_route


def run_turn_pipeline(
    context: TurnContext,
    trace: TurnTrace,
    **kwargs: Any,
) -> tuple[TurnContext, ExecutionTrace, TurnTrace]:
    """Run the turn pipeline through all stages."""
    # Initialize execution trace
    exec_trace = ExecutionTrace(context=context)
    
    # Stage 1: Perceive intent
    perceived_context, perceived_trace = perceive_intent(context, trace, **kwargs)
    
    # Stage 2: Retrieve tools and skills
    retrieved_context, retrieved_trace = retrieve_tools_skills(perceived_context, perceived_trace, **kwargs)
    
    # Stage 3: Execute ReAct loop
    executed_context, executed_trace = execute_react_loop(retrieved_context, retrieved_trace, **kwargs)
    
    # Stage 4: Verify answer
    verified_context, verified_trace = verify_answer(executed_context, executed_trace, **kwargs)
    
    # Merge traces
    final_trace = verified_trace or executed_trace or retrieved_trace or perceived_trace
    
    return verified_context, verified_trace, final_trace

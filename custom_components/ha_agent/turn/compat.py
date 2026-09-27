"""Compatibility shim for LoopState to TurnContext/ExecutionTrace mapping."""

from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LoopStateShim:
    """Shim to map LoopState fields to TurnContext/ExecutionTrace."""
    
    # Map LoopState fields to TurnContext (frozen)
    user_text: str
    history: list[dict[str, str]]
    exposed_entities: list[dict]
    conversation_id: str | None
    identity: str | None = None
    
    # Map LoopState fields to ExecutionTrace (append-only)
    route_history: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    
    @classmethod
    def from_loop_state(cls, loop_state: Any) -> "LoopStateShim":
        """Create shim from a LoopState object."""
        # Extract fields from LoopState
        return cls(
            user_text=getattr(loop_state, 'user_text', ''),
            history=getattr(loop_state, 'history', []),
            exposed_entities=getattr(loop_state, 'exposed_entities', []),
            conversation_id=getattr(loop_state, 'conversation_id', None),
            identity=getattr(loop_state, 'identity', None),
            route_history=getattr(loop_state, 'route_history', []),
            observations=getattr(loop_state, 'observations', []),
            decisions=getattr(loop_state, 'decisions', []),
        )
        
    def to_turn_context(self):
        """Convert to TurnContext."""
        from .pipeline import TurnContext
        return TurnContext(
            user_text=self.user_text,
            history=self.history,
            exposed_entities=self.exposed_entities,
            conversation_id=self.conversation_id,
            identity=self.identity,
            route=self.route_history[0] if self.route_history else None,
        )
        
    def to_execution_trace(self):
        """Convert to ExecutionTrace."""
        from .pipeline import ExecutionTrace, Observation
        from dataclasses import field as dc_field
        
        # Create observations from loop_state observations
        observations = []
        for obs in self.observations:
            observations.append(Observation(
                tool_name=obs.get('tool_name', ''),
                arguments=obs.get('arguments', {}),
                succeeded=obs.get('succeeded', False),
                output=obs.get('output', ''),
                entity_id=obs.get('entity_id'),
            ))
            
        return ExecutionTrace(
            context=self.to_turn_context(),
            observations=observations,
            decisions=self.decisions,
            route_history=self.route_history,
        )

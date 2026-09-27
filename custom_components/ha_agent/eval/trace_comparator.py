"""Trace comparator for eval record-and-replay harness."""

from typing import Dict, Any, List
from custom_components.ha_agent.skills.models import TurnTrace, TurnOutcome


class TraceComparator:
    """Compares TurnTrace objects from live vs replayed runs."""
    
    def __init__(self, ignore_timestamps: bool = True):
        self.ignore_timestamps = ignore_timestamps
        self.differences: List[str] = []
        
    def compare(self, expected: TurnTrace, actual: TurnTrace) -> bool:
        """Compare two TurnTrace objects. Returns True if they match."""
        self.differences = []
        
        # Compare route decisions
        if not self._compare_routes(expected, actual):
            return False
            
        # Compare skill selections
        if not self._compare_skills(expected, actual):
            return False
            
        # Compare LLM call counts
        if not self._compare_llm_calls(expected, actual):
            return False
            
        # Compare tool execution order
        if not self._compare_tool_calls(expected, actual):
            return False
            
        # Compare outcome
        if expected.outcome != actual.outcome:
            self.differences.append(f"Outcome mismatch: expected {expected.outcome}, got {actual.outcome}")
            return False
            
        return len(self.differences) == 0
        
    def _compare_routes(self, expected: TurnTrace, actual: TurnTrace) -> bool:
        """Compare route decisions between traces."""
        # Note: Route history would need to be extracted from ExecutionTrace
        # For now, compare the final route if available
        expected_route = getattr(expected, 'final_route', None)
        actual_route = getattr(actual, 'final_route', None)
        
        if expected_route != actual_route:
            self.differences.append(f"Route mismatch: expected {expected_route}, got {actual_route}")
            return False
            
        return True
        
    def _compare_skills(self, expected: TurnTrace, actual: TurnTrace) -> bool:
        """Compare skill selections between traces."""
        expected_skills = getattr(expected, 'selected_skills', [])
        actual_skills = getattr(actual, 'selected_skills', [])
        
        expected_names = [s.get('name') if isinstance(s, dict) else getattr(s, 'name', None) for s in expected_skills]
        actual_names = [s.get('name') if isinstance(s, dict) else getattr(s, 'name', None) for s in actual_skills]
        
        if expected_names != actual_names:
            self.differences.append(f"Skill selection mismatch: expected {expected_names}, got {actual_names}")
            return False
            
        return True
        
    def _compare_llm_calls(self, expected: TurnTrace, actual: TurnTrace) -> bool:
        """Compare LLM call counts between traces."""
        expected_calls = len(getattr(expected, 'llm_calls', []))
        actual_calls = len(getattr(actual, 'llm_calls', []))
        
        if expected_calls != actual_calls:
            self.differences.append(f"LLM call count mismatch: expected {expected_calls}, got {actual_calls}")
            return False
            
        return True
        
    def _compare_tool_calls(self, expected: TurnTrace, actual: TurnTrace) -> bool:
        """Compare tool execution order between traces."""
        expected_tools = [
            call.get('toolName') or call.get('name')
            for call in getattr(expected, 'tool_calls', [])
        ]
        actual_tools = [
            call.get('toolName') or call.get('name')
            for call in getattr(actual, 'tool_calls', [])
        ]
        
        if expected_tools != actual_tools:
            self.differences.append(f"Tool call order mismatch: expected {expected_tools}, got {actual_tools}")
            return False
            
        return True
        
    def get_differences(self) -> List[str]:
        """Return list of differences found between traces."""
        return self.differences

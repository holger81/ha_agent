"""Response replayer for eval record-and-replay harness."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from custom_components.ha_agent.eval.recorder import TurnRecord, LlmCallRecord, McpToolCallRecord


class ReplayError(Exception):
    """Raised when a replay request has no recorded response."""
    pass


class ResponseReplayer:
    """Replays recorded LLM and MCP responses."""
    
    def __init__(self, fixtures_dir: str = "tests/fixtures/replay"):
        self.fixtures_dir = Path(fixtures_dir)
        self.turn_records: Dict[str, TurnRecord] = {}
        self._load_fixtures()
        
    def _load_fixtures(self):
        """Load all recorded turn fixtures from disk."""
        if not self.fixtures_dir.exists():
            return
            
        for filepath in self.fixtures_dir.glob("*.jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = TurnRecord.from_dict(data)
                        self.turn_records[record.turn_id] = record
                    except json.JSONDecodeError:
                        continue
                        
    def get_llm_response(
        self,
        turn_id: str,
        call_index: int,
        model: str,
        backend_url: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Get recorded LLM response for a call."""
        record = self._get_turn_record(turn_id)
        
        llm_calls = [c for c in record.llm_calls if c.call_index == call_index]
        if not llm_calls:
            raise ReplayError(f"No recorded LLM call {call_index} for turn {turn_id}")
            
        llm_record = llm_calls[0]
        
        # Verify call matches
        if llm_record.model != model or llm_record.backend_url != backend_url:
            raise ReplayError(f"LLM call mismatch: recorded model={llm_record.model}, backend={llm_record.backend_url}")
            
        # Return the recorded response (stored in a mock format)
        return {
            "type": "llm_response",
            "call_index": call_index,
            "model": model,
            "content": llm_record.to_dict().get("mock_content", ""),
            "tool_calls": llm_record.to_dict().get("mock_tool_calls", []),
        }
        
    def get_mcp_tool_response(
        self,
        turn_id: str,
        call_index: int,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get recorded MCP tool response for a call."""
        record = self._get_turn_record(turn_id)
        
        mcp_calls = [c for c in record.mcp_calls if c.call_index == call_index]
        if not mcp_calls:
            raise ReplayError(f"No recorded MCP call {call_index} for turn {turn_id}")
            
        mcp_record = mcp_calls[0]
        
        # Verify call matches
        if mcp_record.tool_name != tool_name:
            raise ReplayError(f"MCP call mismatch: recorded tool={mcp_record.tool_name}")
            
        if mcp_record.error:
            return {
                "type": "mcp_error",
                "call_index": call_index,
                "tool_name": tool_name,
                "error": mcp_record.error,
            }
            
        return {
            "type": "mcp_response",
            "call_index": call_index,
            "tool_name": tool_name,
            "response": mcp_record.response,
        }
        
    def _get_turn_record(self, turn_id: str) -> TurnRecord:
        """Get a turn record by ID."""
        if turn_id not in self.turn_records:
            # Try with sanitized turn_id
            safe_turn_id = turn_id.replace("/", "_").replace("\\", "_")
            if safe_turn_id in self.turn_records:
                turn_id = safe_turn_id
            else:
                raise ReplayError(f"No recorded turn found for {turn_id}")
                
        return self.turn_records[turn_id]
        
    def has_record(self, turn_id: str) -> bool:
        """Check if a turn has recorded responses."""
        safe_turn_id = turn_id.replace("/", "_").replace("\\", "_")
        return turn_id in self.turn_records or safe_turn_id in self.turn_records

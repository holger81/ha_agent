"""Response recorder for eval record-and-replay harness."""

import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


class LlmCallRecord:
    """Record of a single LLM chat completion call."""
    
    def __init__(
        self,
        call_index: int,
        model: str,
        backend_url: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: bool = False,
    ):
        self.call_index = call_index
        self.model = model
        self.backend_url = backend_url
        self.messages = messages
        self.tools = tools
        self.structured_output = structured_output
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "llm_call",
            "call_index": self.call_index,
            "model": self.model,
            "backend_url": self.backend_url,
            "messages": self.messages,
            "tools": self.tools,
            "structured_output": self.structured_output,
            "timestamp": self.timestamp,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlmCallRecord":
        record = cls(
            call_index=data["call_index"],
            model=data["model"],
            backend_url=data["backend_url"],
            messages=data["messages"],
            tools=data.get("tools"),
            structured_output=data.get("structured_output", False),
        )
        record.timestamp = data.get("timestamp", "")
        return record


class McpToolCallRecord:
    """Record of a single MCP callTool response."""
    
    def __init__(
        self,
        call_index: int,
        tool_name: str,
        arguments: Dict[str, Any],
        response: Dict[str, Any],
        error: Optional[str] = None,
    ):
        self.call_index = call_index
        self.tool_name = tool_name
        self.arguments = arguments
        self.response = response
        self.error = error
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "mcp_tool_call",
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "response": self.response,
            "error": self.error,
            "timestamp": self.timestamp,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "McpToolCallRecord":
        record = cls(
            call_index=data["call_index"],
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            response=data["response"],
            error=data.get("error"),
        )
        record.timestamp = data.get("timestamp", "")
        return record


class TurnRecord:
    """Record of a complete turn with all LLM and MCP calls."""
    
    def __init__(self, turn_id: str, user_text: str, conversation_id: str | None = None):
        self.turn_id = turn_id
        self.user_text = user_text
        self.conversation_id = conversation_id
        self.llm_calls: List[LlmCallRecord] = []
        self.mcp_calls: List[McpToolCallRecord] = []
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        
    def add_llm_call(self, record: LlmCallRecord):
        self.llm_calls.append(record)
        
    def add_mcp_call(self, record: McpToolCallRecord):
        self.mcp_calls.append(record)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_text": self.user_text,
            "conversation_id": self.conversation_id,
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "mcp_calls": [call.to_dict() for call in self.mcp_calls],
            "timestamp": self.timestamp,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnRecord":
        record = cls(
            turn_id=data["turn_id"],
            user_text=data["user_text"],
            conversation_id=data.get("conversation_id"),
        )
        for llm_data in data.get("llm_calls", []):
            record.llm_calls.append(LlmCallRecord.from_dict(llm_data))
        for mcp_data in data.get("mcp_calls", []):
            record.mcp_calls.append(McpToolCallRecord.from_dict(mcp_data))
        record.timestamp = data.get("timestamp", "")
        return record


class ResponseRecorder:
    """Records LLM and MCP responses for replay harness."""
    
    def __init__(self, output_dir: str = "tests/fixtures/replay"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_turn: Optional[TurnRecord] = None
        self.llm_call_index = 0
        self.mcp_call_index = 0
        
    def begin_turn(self, turn_id: str, user_text: str, conversation_id: str | None = None):
        """Begin recording a new turn."""
        self.current_turn = TurnRecord(turn_id, user_text, conversation_id)
        self.llm_call_index = 0
        self.mcp_call_index = 0
        
    def record_llm_call(
        self,
        turn_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        response: Dict[str, Any] | None = None,
    ):
        """Record an LLM non-streaming call."""
        if self.current_turn is None:
            self.begin_turn(turn_id, messages[-1].get("content", "") if messages else "")
            
        # Extract backend_url from model or create a placeholder
        backend_url = f"model:{model}"
        
        record = LlmCallRecord(
            call_index=self.llm_call_index,
            model=model,
            backend_url=backend_url,
            messages=messages,
            tools=tools,
            structured_output=response is not None and "response_format" in str(response or {}),
        )
        self.current_turn.add_llm_call(record)
        self.llm_call_index += 1
        
    def record_llm_stream_call(
        self,
        turn_id: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        assistant_message: Dict[str, Any] | None = None,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: Any | None = None,
    ):
        """Record an LLM streaming call."""
        if self.current_turn is None:
            self.begin_turn(turn_id, messages[-1].get("content", "") if messages else "")
            
        backend_url = f"model:{model}"
        
        record = LlmCallRecord(
            call_index=self.llm_call_index,
            model=model,
            backend_url=backend_url,
            messages=messages,
            tools=tools,
            structured_output=False,
        )
        self.current_turn.add_llm_call(record)
        self.llm_call_index += 1
        
    def record_mcp_call(
        self,
        turn_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        extracted_result: str,
    ):
        """Record an MCP tool call response."""
        if self.current_turn is None:
            self.begin_turn(turn_id, "mcp_call")
            
        # Convert result to dict if it's not already
        response_dict = result if isinstance(result, dict) else {"result": result}
        
        record = McpToolCallRecord(
            call_index=self.mcp_call_index,
            tool_name=tool_name,
            arguments=arguments,
            response=response_dict,
            error=None,
        )
        self.current_turn.add_mcp_call(record)
        self.mcp_call_index += 1
        
    def end_turn(self, turn_id: str | None = None):
        """End recording and save the turn record."""
        if self.current_turn is None:
            raise RuntimeError("No turn to end")
            
        if turn_id and self.current_turn.turn_id != turn_id:
            raise RuntimeError(f"Turn ID mismatch: expected {turn_id}, got {self.current_turn.turn_id}")
            
        self._save_turn_record(self.current_turn)
        self.current_turn = None
        
    def _save_turn_record(self, record: TurnRecord):
        """Save a turn record to disk."""
        # Create deterministic filename from turn_id
        safe_turn_id = record.turn_id.replace("/", "_").replace("\\", "_")
        filepath = self.output_dir / f"{safe_turn_id}.jsonl"
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
            
    def get_record_key(self, record: LlmCallRecord | McpToolCallRecord) -> str:
        """Generate a deterministic key for looking up recorded responses."""
        if isinstance(record, LlmCallRecord):
            # Key based on model, backend_url, and normalized messages
            msg_hash = hashlib.sha256(
                json.dumps(record.messages, sort_keys=True).encode()
            ).hexdigest()
            return f"llm:{record.model}:{record.backend_url}:{msg_hash}:{record.call_index}"
        else:
            # Key based on tool_name and normalized arguments
            args_hash = hashlib.sha256(
                json.dumps(record.arguments, sort_keys=True).encode()
            ).hexdigest()
            return f"mcp:{record.tool_name}:{args_hash}:{record.call_index}"

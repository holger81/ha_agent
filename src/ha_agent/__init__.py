"""Home Assistant-compatible agentic loop with MCP and local LLM backends."""

from .agent import run_agent
from .config import AgentConfig, LlmBackend, McpConfig
from .exceptions import HaAgentError
from .llm_client import MCP_CALL_TOOL_SCHEMA, LlmClient
from .mcp_client import McpProxyClient
from .memory import ConversationMemory

__all__ = [
    "MCP_CALL_TOOL_SCHEMA",
    "AgentConfig",
    "ConversationMemory",
    "HaAgentError",
    "LlmBackend",
    "LlmClient",
    "McpConfig",
    "McpProxyClient",
    "run_agent",
]

__version__ = "0.1.0"

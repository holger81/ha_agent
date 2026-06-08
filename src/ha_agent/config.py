"""Configuration dataclasses and defaults for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "local-model"
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_LLM_TIMEOUT = 120

DEFAULT_MCP_URL = "http://127.0.0.1:2222/mcp"
DEFAULT_MCP_TIMEOUT = 120

DEFAULT_MAX_AGENT_ITERATIONS = 8
DEFAULT_CONVERSATION_HISTORY_TURNS = 10

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are a voice assistant for Home Assistant.\n"
    "Answer questions truthfully in plain text. Keep replies concise for speech.\n"
    "When the user asks you to perform an action, ALWAYS use a tool.\n"
    "When asked for news, use the MCP news tool — never invent headlines."
)

DEFAULT_TOOL_INSTRUCTIONS = (
    "MCP tools via mcp_call_tool only. Fields: toolName and arguments (flat). "
    'Never add "value" or nest toolName inside arguments. '
    'Light: {"toolName":"home_assistant__ha_call_service","arguments":{'
    '"domain":"light","service":"turn_off","entity_id":"light.example"}}. '
    'Cover: use open_cover/close_cover. '
    'Search if needed: {"toolName":"home_assistant__ha_search_entities",'
    '"arguments":{"query":"patio door","domain_filter":"cover"}}. '
    'News: {"toolName":"mcp_news__news_curate","arguments":{}} once. '
    "Never use searxng or generic web search for news."
)


@dataclass(frozen=True, slots=True)
class LlmBackend:
    """OpenAI-compatible LLM backend settings."""

    base_url: str
    model: str
    api_key: str | None = None
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    timeout: int = DEFAULT_LLM_TIMEOUT
    enable_thinking: bool = False


@dataclass(frozen=True, slots=True)
class McpConfig:
    """MCP Proxy connection settings."""

    url: str
    bearer_token: str | None = None
    timeout: int = DEFAULT_MCP_TIMEOUT
    health_url: str | None = None

    def resolved_health_url(self) -> str:
        """Return the configured or derived MCP health URL."""
        if self.health_url:
            return self.health_url
        parsed = urlparse(self.url.rstrip("/"))
        return f"{parsed.scheme}://{parsed.netloc}/api/health"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Agent behaviour settings."""

    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT
    tool_instructions: str = DEFAULT_TOOL_INSTRUCTIONS
    max_iterations: int = DEFAULT_MAX_AGENT_ITERATIONS
    history_turns: int = DEFAULT_CONVERSATION_HISTORY_TURNS
    enable_streaming: bool = True

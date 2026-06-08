"""Constants for the HA Agent integration."""

from logging import Logger, getLogger

DOMAIN = "ha_agent"

CONF_AGENT_SYSTEM_PROMPT = "agent_system_prompt"
CONF_TOOL_INSTRUCTIONS = "tool_instructions"
CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_MAX_TOKENS = "llm_max_tokens"
CONF_LLM_TEMPERATURE = "llm_temperature"
CONF_LLM_TIMEOUT = "llm_timeout"
CONF_LLM_ENABLE_THINKING = "llm_enable_thinking"

CONF_MCP_URL = "mcp_url"
CONF_MCP_BEARER_TOKEN = "mcp_bearer_token"
CONF_MCP_TIMEOUT = "mcp_timeout"
CONF_MCP_HEALTH_URL = "mcp_health_url"

CONF_MAX_AGENT_ITERATIONS = "max_agent_iterations"
CONF_CONVERSATION_HISTORY_TURNS = "conversation_history_turns"
CONF_CONVERSATION_ENABLE_STREAMING = "conversation_enable_streaming"

DEFAULT_LLM_BASE_URL = "http://192.168.10.31:9292/v1"
DEFAULT_LLM_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF:IQ4_XS"
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_LLM_TIMEOUT = 120

DEFAULT_MCP_URL = "http://192.168.10.31:2222/mcp"
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

SUPPORTED_LANGUAGES = ["en", "en-US"]
DEFAULT_LANGUAGE = "en-US"

DATA_KEY = DOMAIN

LOGGER: Logger = getLogger(__package__)

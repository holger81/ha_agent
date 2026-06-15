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
# Legacy key; migrated to CONF_LLM_THINKING_LEVEL.
CONF_LLM_ENABLE_THINKING = "llm_enable_thinking"
CONF_LLM_THINKING_LEVEL = "llm_thinking_level"

CONF_MCP_URL = "mcp_url"
CONF_MCP_BEARER_TOKEN = "mcp_bearer_token"
CONF_MCP_TIMEOUT = "mcp_timeout"
CONF_MCP_HEALTH_URL = "mcp_health_url"

CONF_MAX_AGENT_ITERATIONS = "max_agent_iterations"
CONF_CONVERSATION_HISTORY_TURNS = "conversation_history_turns"
CONF_CONVERSATION_ENABLE_STREAMING = "conversation_enable_streaming"
CONF_CONVERSATION_SHOW_REASONING = "conversation_show_reasoning"

CONF_SKILLS_LEARNING_ENABLED = "skills_learning_enabled"
CONF_SKILLS_AUTO_SAVE = "skills_auto_save"
CONF_SKILLS_USE_ENABLED = "skills_use_enabled"
CONF_SKILLS_MAX_INJECT = "skills_max_inject"

CONF_ACTION_MODEL_ENABLED = "action_model_enabled"
CONF_ACTION_LLM_BASE_URL = "action_llm_base_url"
CONF_ACTION_LLM_MODEL = "action_llm_model"
CONF_ACTION_LLM_TEMPERATURE = "action_llm_temperature"
CONF_ACTION_LLM_MAX_TOKENS = "action_llm_max_tokens"

DEFAULT_LLM_BASE_URL = "http://192.168.10.31:9292/v1"
DEFAULT_LLM_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF:IQ4_XS"
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_LLM_TIMEOUT = 120

DEFAULT_ACTION_LLM_TEMPERATURE = 0.1
DEFAULT_ACTION_LLM_MAX_TOKENS = 512

DEFAULT_MCP_URL = "http://192.168.10.31:2222/mcp"
DEFAULT_MCP_TIMEOUT = 120

DEFAULT_MAX_AGENT_ITERATIONS = 8
DEFAULT_CONVERSATION_HISTORY_TURNS = 10

DEFAULT_SKILLS_MAX_INJECT = 3

MCP_SESSION_TOOLS_TTL_SECONDS = 3600

DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are a voice assistant for Home Assistant.\n"
    "Answer questions truthfully in plain text. Keep replies concise for speech.\n"
    "When the user asks you to perform an action, ALWAYS use a tool.\n"
    "Never invent facts; use tools to fetch real data."
)

DEFAULT_TOOL_INSTRUCTIONS = (
    "Follow MCP SERVER INSTRUCTIONS and use the provided session tools. "
    "For homeassistant device actions, call callTool with toolName "
    "home_assistant__ha_call_service and arguments containing domain, service, "
    "and entity_id."
)

# Saved during setup before MCP-compliant defaults; reset on upgrade.
LEGACY_TOOL_INSTRUCTION_MARKERS = ("mcp_call_tool",)

CONFIG_ENTRY_VERSION = 6

SUPPORTED_LANGUAGES = ["en", "en-US"]
DEFAULT_LANGUAGE = "en-US"

# Assist entity exposure uses the global "conversation" assistant key, not DOMAIN.
ASSIST_EXPOSE_ASSISTANT = "conversation"

DATA_KEY = DOMAIN

LOGGER: Logger = getLogger(__package__)

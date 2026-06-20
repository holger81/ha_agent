"""Build config objects from a config entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .const import (
    CONF_ACTION_LLM_BASE_URL,
    CONF_ACTION_LLM_MAX_TOKENS,
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_LLM_TEMPERATURE,
    CONF_ACTION_MODEL_ENABLED,
    CONF_AGENT_SYSTEM_PROMPT,
    CONF_CLASSIFIER_LLM_BASE_URL,
    CONF_CLASSIFIER_LLM_MODEL,
    CONF_CLASSIFIER_MODEL_ENABLED,
    CONF_CONVERSATION_ENABLE_STREAMING,
    CONF_CONVERSATION_HISTORY_TURNS,
    CONF_CONVERSATION_SHOW_REASONING,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_ENABLE_THINKING,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_LLM_THINKING_LEVEL,
    CONF_LLM_TIMEOUT,
    CONF_MAX_AGENT_ITERATIONS,
    CONF_MCP_BEARER_TOKEN,
    CONF_MCP_HEALTH_URL,
    CONF_MCP_TIMEOUT,
    CONF_MCP_URL,
    CONF_SKILLS_AUTO_SAVE,
    CONF_SKILLS_LEARNING_ENABLED,
    CONF_SKILLS_MAX_INJECT,
    CONF_SKILLS_USE_ENABLED,
    CONF_TOOL_INSTRUCTIONS,
    DEFAULT_ACTION_LLM_MAX_TOKENS,
    DEFAULT_ACTION_LLM_TEMPERATURE,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_CLASSIFIER_LLM_MAX_TOKENS,
    DEFAULT_CLASSIFIER_LLM_TEMPERATURE,
    DEFAULT_CONVERSATION_HISTORY_TURNS,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_AGENT_ITERATIONS,
    DEFAULT_MCP_TIMEOUT,
    DEFAULT_MCP_URL,
    DEFAULT_SKILLS_MAX_INJECT,
    DEFAULT_TOOL_INSTRUCTIONS,
)
from .thinking import DEFAULT_THINKING_LEVEL, normalize_thinking_level

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry


@dataclass(frozen=True, slots=True)
class LlmBackend:
    """OpenAI-compatible LLM backend settings."""

    base_url: str
    model: str
    api_key: str | None
    max_tokens: int
    temperature: float
    timeout: int
    thinking_level: str


@dataclass(frozen=True, slots=True)
class McpConfig:
    """MCP Proxy connection settings."""

    url: str
    bearer_token: str | None
    timeout: int
    health_url: str


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Agent behaviour settings."""

    system_prompt: str
    tool_instructions: str
    max_iterations: int
    history_turns: int
    enable_streaming: bool
    show_reasoning_in_chat: bool


@dataclass(frozen=True, slots=True)
class SkillsConfig:
    """Learned skills behaviour settings."""

    learning_enabled: bool
    auto_save: bool
    use_enabled: bool
    max_inject: int


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Optional action-model routing settings."""

    action_enabled: bool
    action_backend: LlmBackend | None
    classifier_backend: LlmBackend | None = None


def default_mcp_health_url(mcp_url: str) -> str:
    """Derive MCP health URL from the MCP endpoint."""
    parsed = urlparse(mcp_url.rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}/api/health"


def get_llm_backend(entry: ConfigEntry) -> LlmBackend:
    """Return LLM settings for the config entry."""
    data = entry.data
    return LlmBackend(
        base_url=data.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL).rstrip("/"),
        model=data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
        api_key=data.get(CONF_LLM_API_KEY) or None,
        max_tokens=int(data.get(CONF_LLM_MAX_TOKENS, DEFAULT_LLM_MAX_TOKENS)),
        temperature=float(data.get(CONF_LLM_TEMPERATURE, DEFAULT_LLM_TEMPERATURE)),
        timeout=int(data.get(CONF_LLM_TIMEOUT, DEFAULT_LLM_TIMEOUT)),
        thinking_level=normalize_thinking_level(
            data.get(CONF_LLM_THINKING_LEVEL, data.get(CONF_LLM_ENABLE_THINKING))
        ),
    )


def get_mcp_config(entry: ConfigEntry) -> McpConfig:
    """Return MCP Proxy settings for the config entry."""
    data = entry.data
    mcp_url = data.get(CONF_MCP_URL, DEFAULT_MCP_URL).rstrip("/")
    return McpConfig(
        url=mcp_url,
        bearer_token=data.get(CONF_MCP_BEARER_TOKEN) or None,
        timeout=int(data.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT)),
        health_url=data.get(CONF_MCP_HEALTH_URL) or default_mcp_health_url(mcp_url),
    )


def get_action_backend(entry: ConfigEntry) -> LlmBackend | None:
    """Return action LLM settings when action routing is enabled."""
    data = entry.data
    if not data.get(CONF_ACTION_MODEL_ENABLED):
        return None

    action_model = data.get(CONF_ACTION_LLM_MODEL)
    if not action_model:
        return None

    chat = get_llm_backend(entry)
    action_url = (data.get(CONF_ACTION_LLM_BASE_URL) or chat.base_url).rstrip("/")
    return LlmBackend(
        base_url=action_url,
        model=action_model,
        api_key=chat.api_key,
        max_tokens=int(
            data.get(CONF_ACTION_LLM_MAX_TOKENS, DEFAULT_ACTION_LLM_MAX_TOKENS)
        ),
        temperature=float(
            data.get(CONF_ACTION_LLM_TEMPERATURE, DEFAULT_ACTION_LLM_TEMPERATURE)
        ),
        timeout=chat.timeout,
        thinking_level=DEFAULT_THINKING_LEVEL,
    )


def get_classifier_backend(entry: ConfigEntry) -> LlmBackend | None:
    """Return a dedicated playbook-classifier backend when configured."""
    data = entry.data
    if not data.get(CONF_CLASSIFIER_MODEL_ENABLED):
        return None

    model = data.get(CONF_CLASSIFIER_LLM_MODEL)
    if not model:
        return None

    chat = get_llm_backend(entry)
    base_url = (data.get(CONF_CLASSIFIER_LLM_BASE_URL) or chat.base_url).rstrip("/")
    return LlmBackend(
        base_url=base_url,
        model=model,
        api_key=chat.api_key,
        max_tokens=DEFAULT_CLASSIFIER_LLM_MAX_TOKENS,
        temperature=DEFAULT_CLASSIFIER_LLM_TEMPERATURE,
        timeout=chat.timeout,
        thinking_level="off",
    )


def get_skills_config(entry: ConfigEntry) -> SkillsConfig:
    """Return skills settings for the config entry."""
    data = entry.data
    return SkillsConfig(
        learning_enabled=bool(data.get(CONF_SKILLS_LEARNING_ENABLED, False)),
        auto_save=bool(data.get(CONF_SKILLS_AUTO_SAVE, False)),
        use_enabled=bool(data.get(CONF_SKILLS_USE_ENABLED, True)),
        max_inject=int(data.get(CONF_SKILLS_MAX_INJECT, DEFAULT_SKILLS_MAX_INJECT)),
    )


def get_router_config(entry: ConfigEntry) -> RouterConfig:
    """Return router settings for the config entry."""
    return RouterConfig(
        action_enabled=bool(entry.data.get(CONF_ACTION_MODEL_ENABLED)),
        action_backend=get_action_backend(entry),
        classifier_backend=get_classifier_backend(entry),
    )


def get_agent_config(entry: ConfigEntry) -> AgentConfig:
    """Return agent settings for the config entry."""
    data = entry.data
    return AgentConfig(
        system_prompt=data.get(CONF_AGENT_SYSTEM_PROMPT, DEFAULT_AGENT_SYSTEM_PROMPT),
        tool_instructions=data.get(CONF_TOOL_INSTRUCTIONS, DEFAULT_TOOL_INSTRUCTIONS),
        max_iterations=int(
            data.get(CONF_MAX_AGENT_ITERATIONS, DEFAULT_MAX_AGENT_ITERATIONS)
        ),
        history_turns=int(
            data.get(
                CONF_CONVERSATION_HISTORY_TURNS,
                DEFAULT_CONVERSATION_HISTORY_TURNS,
            )
        ),
        enable_streaming=bool(
            data.get(CONF_CONVERSATION_ENABLE_STREAMING, True),
        ),
        show_reasoning_in_chat=bool(
            data.get(CONF_CONVERSATION_SHOW_REASONING, True),
        ),
    )

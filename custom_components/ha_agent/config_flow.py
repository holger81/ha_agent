"""Config flow for HA Agent."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_create_clientsession

from .config_helpers import LlmBackend, McpConfig, default_mcp_health_url
from .const import (
    CONF_AGENT_SYSTEM_PROMPT,
    CONF_CONVERSATION_ENABLE_STREAMING,
    CONF_CONVERSATION_HISTORY_TURNS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_ENABLE_THINKING,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_LLM_TIMEOUT,
    CONF_MAX_AGENT_ITERATIONS,
    CONF_MCP_BEARER_TOKEN,
    CONF_MCP_HEALTH_URL,
    CONF_MCP_TIMEOUT,
    CONF_MCP_URL,
    CONF_TOOL_INSTRUCTIONS,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_CONVERSATION_HISTORY_TURNS,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_AGENT_ITERATIONS,
    DEFAULT_MCP_TIMEOUT,
    DEFAULT_MCP_URL,
    DEFAULT_TOOL_INSTRUCTIONS,
    DOMAIN,
    LOGGER,
)
from .llm_client import LlmClient
from .mcp_client import McpProxyClient


def _agent_prompt_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_AGENT_SYSTEM_PROMPT,
                default=defaults.get(
                    CONF_AGENT_SYSTEM_PROMPT, DEFAULT_AGENT_SYSTEM_PROMPT
                ),
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.TEXT,
                    multiline=True,
                ),
            ),
            vol.Required(
                CONF_TOOL_INSTRUCTIONS,
                default=defaults.get(CONF_TOOL_INSTRUCTIONS, DEFAULT_TOOL_INSTRUCTIONS),
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.TEXT,
                    multiline=True,
                ),
            ),
        }
    )


def _llm_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_LLM_BASE_URL,
                default=defaults.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL),
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.URL),
            ),
            vol.Required(
                CONF_LLM_MODEL,
                default=defaults.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
            ): str,
            vol.Optional(
                CONF_LLM_API_KEY,
                default=defaults.get(CONF_LLM_API_KEY, ""),
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.PASSWORD,
                ),
            ),
            vol.Optional(
                CONF_LLM_MAX_TOKENS,
                default=defaults.get(CONF_LLM_MAX_TOKENS, DEFAULT_LLM_MAX_TOKENS),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=256,
                    max=32768,
                    step=256,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_LLM_TEMPERATURE,
                default=defaults.get(CONF_LLM_TEMPERATURE, DEFAULT_LLM_TEMPERATURE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_LLM_TIMEOUT,
                default=defaults.get(CONF_LLM_TIMEOUT, DEFAULT_LLM_TIMEOUT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=10,
                    max=600,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_LLM_ENABLE_THINKING,
                default=defaults.get(CONF_LLM_ENABLE_THINKING, False),
            ): bool,
        }
    )


def _mcp_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    mcp_url = defaults.get(CONF_MCP_URL, DEFAULT_MCP_URL)
    return vol.Schema(
        {
            vol.Required(
                CONF_MCP_URL,
                default=mcp_url,
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.URL),
            ),
            vol.Optional(
                CONF_MCP_BEARER_TOKEN,
                default=defaults.get(CONF_MCP_BEARER_TOKEN, ""),
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    type=selector.TextSelectorType.PASSWORD,
                ),
            ),
            vol.Optional(
                CONF_MCP_TIMEOUT,
                default=defaults.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=10,
                    max=600,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_MCP_HEALTH_URL,
                default=defaults.get(CONF_MCP_HEALTH_URL)
                or default_mcp_health_url(mcp_url),
            ): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.URL),
            ),
        }
    )


def _agent_settings_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    defaults = defaults or {}
    return vol.Schema(
        {
            vol.Optional(
                CONF_MAX_AGENT_ITERATIONS,
                default=defaults.get(
                    CONF_MAX_AGENT_ITERATIONS, DEFAULT_MAX_AGENT_ITERATIONS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=20,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_CONVERSATION_HISTORY_TURNS,
                default=defaults.get(
                    CONF_CONVERSATION_HISTORY_TURNS,
                    DEFAULT_CONVERSATION_HISTORY_TURNS,
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=1,
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_CONVERSATION_ENABLE_STREAMING,
                default=defaults.get(CONF_CONVERSATION_ENABLE_STREAMING, True),
            ): bool,
        }
    )


class HaAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA Agent."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the flow."""
        self._data: dict[str, Any] = {}
        self._reconfigure_entry: config_entries.ConfigEntry | None = None

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Reconfigure an existing entry."""
        self._reconfigure_entry = self._get_reconfigure_entry()
        self._data = dict(self._reconfigure_entry.data)
        return await self.async_step_user(user_input)

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.FlowResult:
        """Configure agent prompts."""
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_llm()

        return self.async_show_form(
            step_id="user",
            data_schema=_agent_prompt_schema(self._data),
        )

    async def async_step_llm(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.FlowResult:
        """Configure LLM backend."""
        errors: dict[str, str] = {}
        if user_input is not None:
            base_url = user_input[CONF_LLM_BASE_URL].rstrip("/")
            api_key = user_input.get(CONF_LLM_API_KEY) or None
            backend = LlmBackend(
                base_url=base_url,
                model=user_input[CONF_LLM_MODEL],
                api_key=api_key,
                max_tokens=int(
                    user_input.get(CONF_LLM_MAX_TOKENS, DEFAULT_LLM_MAX_TOKENS)
                ),
                temperature=float(
                    user_input.get(CONF_LLM_TEMPERATURE, DEFAULT_LLM_TEMPERATURE)
                ),
                timeout=int(user_input.get(CONF_LLM_TIMEOUT, DEFAULT_LLM_TIMEOUT)),
                enable_thinking=bool(user_input.get(CONF_LLM_ENABLE_THINKING, False)),
            )
            client = LlmClient(async_create_clientsession(self.hass))
            try:
                await client.check_connection(backend)
            except Exception as err:
                LOGGER.warning("LLM connection check failed: %s", err)
                errors["base"] = "llm_connect_failed"
            else:
                self._data.update(user_input)
                self._data[CONF_LLM_BASE_URL] = base_url
                if not api_key and CONF_LLM_API_KEY in self._data:
                    self._data.pop(CONF_LLM_API_KEY, None)
                return await self.async_step_mcp()

        return self.async_show_form(
            step_id="llm",
            data_schema=_llm_schema(self._data),
            errors=errors,
        )

    async def async_step_mcp(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.FlowResult:
        """Configure MCP Proxy."""
        errors: dict[str, str] = {}
        if user_input is not None:
            mcp_url = user_input[CONF_MCP_URL].rstrip("/")
            llm_url = self._data[CONF_LLM_BASE_URL]
            config = McpConfig(
                url=mcp_url,
                bearer_token=user_input.get(CONF_MCP_BEARER_TOKEN) or None,
                timeout=int(user_input.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT)),
                health_url=user_input.get(CONF_MCP_HEALTH_URL)
                or default_mcp_health_url(mcp_url),
            )
            client = McpProxyClient(async_create_clientsession(self.hass), config)
            try:
                await client.check_health()
                await client.initialize()
            except Exception as err:
                LOGGER.warning("MCP connection check failed: %s", err)
                errors["base"] = "mcp_connect_failed"
            else:
                self._data.update(user_input)
                self._data[CONF_MCP_URL] = mcp_url
                token = user_input.get(CONF_MCP_BEARER_TOKEN)
                if not token:
                    self._data.pop(CONF_MCP_BEARER_TOKEN, None)
                if not self._reconfigure_entry:
                    await self.async_set_unique_id(f"{llm_url}|{mcp_url}")
                    self._abort_if_unique_id_configured()
                return await self.async_step_agent_settings()

        return self.async_show_form(
            step_id="mcp",
            data_schema=_mcp_schema(self._data),
            errors=errors,
        )

    async def async_step_agent_settings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> config_entries.FlowResult:
        """Configure agent behaviour."""
        if user_input is not None:
            self._data.update(user_input)
            if self._reconfigure_entry:
                self.hass.config_entries.async_update_entry(
                    self._reconfigure_entry,
                    data=self._data,
                )
                await self.hass.config_entries.async_reload(
                    self._reconfigure_entry.entry_id
                )
                return self.async_abort(reason="reconfigure_successful")

            return self.async_create_entry(
                title="HA Agent",
                data=self._data,
            )

        return self.async_show_form(
            step_id="agent_settings",
            data_schema=_agent_settings_schema(self._data),
        )

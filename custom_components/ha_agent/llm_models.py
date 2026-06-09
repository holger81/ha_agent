"""Helpers for discovering models from the LLM server."""

from __future__ import annotations

from typing import Any

from homeassistant.exceptions import HomeAssistantError

from .config_helpers import LlmBackend
from .const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_TIMEOUT,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
)
from .llm_client import LlmClient


def llm_backend_from_data(data: dict[str, Any]) -> LlmBackend:
    """Build minimal LLM backend settings for model discovery."""
    return LlmBackend(
        base_url=data.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL).rstrip("/"),
        model=data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
        api_key=data.get(CONF_LLM_API_KEY) or None,
        max_tokens=256,
        temperature=0.2,
        timeout=int(data.get(CONF_LLM_TIMEOUT, DEFAULT_LLM_TIMEOUT)),
        enable_thinking=False,
    )


async def async_fetch_model_ids(
    client: LlmClient,
    data: dict[str, Any],
) -> list[str]:
    """Return sorted model ids from the LLM server."""
    backend = llm_backend_from_data(data)
    models = await client.list_models(backend)
    current = data.get(CONF_LLM_MODEL)
    if current and current not in models:
        return [current, *models]
    return models


async def async_fetch_model_options(
    client: LlmClient,
    data: dict[str, Any],
) -> list[str]:
    """Return model ids for selectors, with a fallback when discovery fails."""
    try:
        return await async_fetch_model_ids(client, data)
    except HomeAssistantError:
        current = data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)
        return [current] if current else [DEFAULT_LLM_MODEL]

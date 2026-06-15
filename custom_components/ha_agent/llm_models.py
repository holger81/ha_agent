"""Helpers for discovering models from the LLM server."""

from __future__ import annotations

from typing import Any

from homeassistant.exceptions import HomeAssistantError

from .config_helpers import LlmBackend
from .const import (
    CONF_ACTION_LLM_BASE_URL,
    CONF_ACTION_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_TIMEOUT,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
)
from .llm_client import LlmClient
from .thinking import DEFAULT_THINKING_LEVEL

ModelTarget = str  # "chat" | "action"


def llm_backend_from_data(
    data: dict[str, Any],
    *,
    target: ModelTarget = "chat",
) -> LlmBackend:
    """Build minimal LLM backend settings for model discovery."""
    chat_url = data.get(CONF_LLM_BASE_URL, DEFAULT_LLM_BASE_URL).rstrip("/")
    if target == "action":
        base_url = (data.get(CONF_ACTION_LLM_BASE_URL) or chat_url).rstrip("/")
        model = data.get(CONF_ACTION_LLM_MODEL) or data.get(
            CONF_LLM_MODEL,
            DEFAULT_LLM_MODEL,
        )
    else:
        base_url = chat_url
        model = data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)

    return LlmBackend(
        base_url=base_url,
        model=model,
        api_key=data.get(CONF_LLM_API_KEY) or None,
        max_tokens=256,
        temperature=0.2,
        timeout=int(data.get(CONF_LLM_TIMEOUT, DEFAULT_LLM_TIMEOUT)),
        thinking_level=DEFAULT_THINKING_LEVEL,
    )


async def async_fetch_model_ids(
    client: LlmClient,
    data: dict[str, Any],
    *,
    target: ModelTarget = "chat",
    current_model: str | None = None,
) -> list[str]:
    """Return sorted model ids from the LLM server."""
    backend = llm_backend_from_data(data, target=target)
    models = await client.list_models(backend)
    current = current_model or (
        data.get(CONF_ACTION_LLM_MODEL)
        if target == "action"
        else data.get(CONF_LLM_MODEL)
    )
    if current and current not in models:
        return [current, *models]
    return models


async def async_fetch_model_options(
    client: LlmClient,
    data: dict[str, Any],
    *,
    target: ModelTarget = "chat",
    current_model: str | None = None,
) -> list[str]:
    """Return model ids for selectors, with a fallback when discovery fails."""
    fallback = current_model or (
        data.get(CONF_ACTION_LLM_MODEL)
        if target == "action"
        else data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)
    )
    try:
        return await async_fetch_model_ids(
            client,
            data,
            target=target,
            current_model=current_model,
        )
    except HomeAssistantError:
        return [fallback] if fallback else [DEFAULT_LLM_MODEL]

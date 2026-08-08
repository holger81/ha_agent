"""Config API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..config_helpers import LlmBackend, get_llm_backend
from ..const import (
    CONF_ACTION_LLM_BASE_URL,
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_MODEL_ENABLED,
    CONF_CLASSIFIER_LLM_BASE_URL,
    CONF_CLASSIFIER_LLM_MODEL,
    CONF_CLASSIFIER_MODEL_ENABLED,
    CONF_CONVERSATION_ENABLE_STREAMING,
    CONF_CONVERSATION_HISTORY_TURNS,
    CONF_CONVERSATION_MEMORY_PERSIST,
    CONF_CONVERSATION_SHOW_REASONING,
    CONF_EMAIL_LLM_BASE_URL,
    CONF_EMAIL_LLM_MODEL,
    CONF_EMAIL_MODEL_ENABLED,
    CONF_EVAL_DISCOVER_MAX_MODELS,
    CONF_EVAL_DISCOVER_REQUIRE_DOWNLOAD_APPROVAL,
    CONF_EVAL_DISCOVER_REQUIRE_TRIAL_APPROVAL,
    CONF_EVAL_DOWNLOAD_WEBHOOK_URL,
    CONF_EVAL_MODELS_DIR,
    CONF_IDENTITY_AUTO_NAME_ENABLED,
    CONF_IDENTITY_GUEST_CREATE_THRESHOLD,
    CONF_IDENTITY_GUEST_MATCH_THRESHOLD,
    CONF_IDENTITY_GUEST_TIE_MARGIN,
    CONF_IDENTITY_MIN_UTTERANCE_MS,
    CONF_IDENTITY_VOICE_ENABLED,
    CONF_LLM_MODEL,
    CONF_LLM_THINKING_LEVEL,
    CONF_MAX_AGENT_ITERATIONS,
    CONF_NEWS_LLM_BASE_URL,
    CONF_NEWS_LLM_MODEL,
    CONF_NEWS_MODEL_ENABLED,
    CONF_OBSERVER_LLM_BASE_URL,
    CONF_OBSERVER_LLM_MODEL,
    CONF_OBSERVER_MODEL_ENABLED,
    CONF_PLANNER_LLM_BASE_URL,
    CONF_PLANNER_LLM_MODEL,
    CONF_PLANNER_MODEL_ENABLED,
    CONF_SKILLS_AUTO_SAVE,
    CONF_SKILLS_LEARNING_ENABLED,
    CONF_SKILLS_MAX_INJECT,
    CONF_SKILLS_USE_ENABLED,
    CONF_VERIFIER_LLM_BASE_URL,
    CONF_VERIFIER_LLM_MODEL,
    CONF_VERIFIER_MODEL_ENABLED,
)
from ..llm_client import LlmClient
from ..memory import async_load_memory, async_save_memory
from ..thinking import DEFAULT_THINKING_LEVEL
from .helpers import config_snapshot, get_entry

_CONFIG_KEYS = {
    "llm_model": CONF_LLM_MODEL,
    "thinking_level": CONF_LLM_THINKING_LEVEL,
    "action_model_enabled": CONF_ACTION_MODEL_ENABLED,
    "action_llm_model": CONF_ACTION_LLM_MODEL,
    "action_llm_base_url": CONF_ACTION_LLM_BASE_URL,
    "classifier_model_enabled": CONF_CLASSIFIER_MODEL_ENABLED,
    "classifier_llm_model": CONF_CLASSIFIER_LLM_MODEL,
    "classifier_llm_base_url": CONF_CLASSIFIER_LLM_BASE_URL,
    "planner_model_enabled": CONF_PLANNER_MODEL_ENABLED,
    "planner_llm_model": CONF_PLANNER_LLM_MODEL,
    "planner_llm_base_url": CONF_PLANNER_LLM_BASE_URL,
    "verifier_model_enabled": CONF_VERIFIER_MODEL_ENABLED,
    "verifier_llm_model": CONF_VERIFIER_LLM_MODEL,
    "verifier_llm_base_url": CONF_VERIFIER_LLM_BASE_URL,
    "observer_model_enabled": CONF_OBSERVER_MODEL_ENABLED,
    "observer_llm_model": CONF_OBSERVER_LLM_MODEL,
    "observer_llm_base_url": CONF_OBSERVER_LLM_BASE_URL,
    "email_model_enabled": CONF_EMAIL_MODEL_ENABLED,
    "email_llm_model": CONF_EMAIL_LLM_MODEL,
    "email_llm_base_url": CONF_EMAIL_LLM_BASE_URL,
    "news_model_enabled": CONF_NEWS_MODEL_ENABLED,
    "news_llm_model": CONF_NEWS_LLM_MODEL,
    "news_llm_base_url": CONF_NEWS_LLM_BASE_URL,
    "max_iterations": CONF_MAX_AGENT_ITERATIONS,
    "history_turns": CONF_CONVERSATION_HISTORY_TURNS,
    "enable_streaming": CONF_CONVERSATION_ENABLE_STREAMING,
    "show_reasoning_in_chat": CONF_CONVERSATION_SHOW_REASONING,
    "skills_learning_enabled": CONF_SKILLS_LEARNING_ENABLED,
    "skills_auto_save": CONF_SKILLS_AUTO_SAVE,
    "skills_use_enabled": CONF_SKILLS_USE_ENABLED,
    "skills_max_inject": CONF_SKILLS_MAX_INJECT,
    "memory_persist": CONF_CONVERSATION_MEMORY_PERSIST,
    "eval_models_dir": CONF_EVAL_MODELS_DIR,
    "eval_download_webhook_url": CONF_EVAL_DOWNLOAD_WEBHOOK_URL,
    "eval_discover_require_download_approval": (
        CONF_EVAL_DISCOVER_REQUIRE_DOWNLOAD_APPROVAL
    ),
    "eval_discover_require_trial_approval": CONF_EVAL_DISCOVER_REQUIRE_TRIAL_APPROVAL,
    "eval_discover_max_models": CONF_EVAL_DISCOVER_MAX_MODELS,
    "identity_voice_enabled": CONF_IDENTITY_VOICE_ENABLED,
    "identity_guest_match_threshold": CONF_IDENTITY_GUEST_MATCH_THRESHOLD,
    "identity_guest_create_threshold": CONF_IDENTITY_GUEST_CREATE_THRESHOLD,
    "identity_guest_tie_margin": CONF_IDENTITY_GUEST_TIE_MARGIN,
    "identity_min_utterance_ms": CONF_IDENTITY_MIN_UTTERANCE_MS,
    "identity_auto_name_enabled": CONF_IDENTITY_AUTO_NAME_ENABLED,
}


def get_config(hass: HomeAssistant, entry_id: str) -> dict[str, Any]:
    """Return config snapshot for an entry."""
    entry = get_entry(hass, entry_id)
    return config_snapshot(hass, entry)


async def set_config(
    hass: HomeAssistant,
    entry_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Update config entry data fields from the console."""
    entry = get_entry(hass, entry_id)
    data = dict(entry.data)
    changed = False
    for key, value in updates.items():
        conf_key = _CONFIG_KEYS.get(key)
        if conf_key is None:
            continue
        data[conf_key] = value
        changed = True
    if not changed:
        raise HomeAssistantError("No valid config keys in update")
    hass.config_entries.async_update_entry(entry, data=data)
    if updates.get("memory_persist") is True:
        await async_load_memory(hass, entry_id)
    elif updates.get("memory_persist") is False:
        await async_save_memory(hass, entry_id)
    await hass.config_entries.async_reload(entry_id)
    reloaded = get_entry(hass, entry_id)
    return config_snapshot(hass, reloaded)


async def reload_integration(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, Any]:
    """Reload the config entry (Devices & services → Reload)."""
    get_entry(hass, entry_id)
    await hass.config_entries.async_reload(entry_id)
    reloaded = get_entry(hass, entry_id)
    return config_snapshot(hass, reloaded)


async def list_available_models(
    hass: HomeAssistant,
    entry_id: str,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return model ids from the entry's LLM server (GET /v1/models)."""
    entry = get_entry(hass, entry_id)
    chat = get_llm_backend(entry)
    url = (base_url or chat.base_url or "").strip().rstrip("/")
    if not url:
        raise HomeAssistantError("LLM base URL is not configured")
    backend = LlmBackend(
        base_url=url,
        model=chat.model,
        api_key=chat.api_key,
        max_tokens=64,
        temperature=0.0,
        timeout=min(chat.timeout, 30),
        thinking_level=DEFAULT_THINKING_LEVEL,
    )
    session = async_get_clientsession(hass)
    llm = LlmClient(session)
    models = await llm.list_models(backend)
    # Keep currently configured chat model visible even if temporarily unloaded.
    if chat.model and chat.model not in models and url == chat.base_url.rstrip("/"):
        models = [chat.model, *models]
    return {"base_url": url, "models": models}

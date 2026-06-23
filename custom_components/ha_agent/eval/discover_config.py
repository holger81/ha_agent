"""Eval discover settings from the config entry."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry

from ..const import (
    CONF_EVAL_DISCOVER_MAX_MODELS,
    CONF_EVAL_DISCOVER_REQUIRE_DOWNLOAD_APPROVAL,
    CONF_EVAL_DISCOVER_REQUIRE_TRIAL_APPROVAL,
    CONF_EVAL_DOWNLOAD_WEBHOOK_URL,
    CONF_EVAL_MODELS_DIR,
    DEFAULT_EVAL_DISCOVER_MAX_MODELS,
)


@dataclass(slots=True)
class DiscoverConfig:
    """Phase-3 model discovery options."""

    models_dir: str | None
    download_webhook_url: str | None
    require_download_approval: bool
    require_trial_approval: bool
    max_models: int


def get_discover_config(entry: ConfigEntry) -> DiscoverConfig:
    data = entry.data
    models_dir = str(data.get(CONF_EVAL_MODELS_DIR, "") or "").strip() or None
    webhook = (
        str(data.get(CONF_EVAL_DOWNLOAD_WEBHOOK_URL, "") or "").strip() or None
    )
    return DiscoverConfig(
        models_dir=models_dir,
        download_webhook_url=webhook,
        require_download_approval=bool(
            data.get(CONF_EVAL_DISCOVER_REQUIRE_DOWNLOAD_APPROVAL, True)
        ),
        require_trial_approval=bool(
            data.get(CONF_EVAL_DISCOVER_REQUIRE_TRIAL_APPROVAL, True)
        ),
        max_models=int(
            data.get(CONF_EVAL_DISCOVER_MAX_MODELS, DEFAULT_EVAL_DISCOVER_MAX_MODELS)
        ),
    )

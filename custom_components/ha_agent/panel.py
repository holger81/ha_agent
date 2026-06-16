"""Register the HA Agent console sidebar panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

from homeassistant.components import panel_custom
from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DATA_KEY, DOMAIN, LOGGER

PANEL_URL_PATH = DOMAIN
PANEL_STATIC_URL = f"/{DOMAIN}_panel"
PANEL_TITLE = "HA Agent"
PANEL_ICON = "mdi:robot-happy"
PANEL_COMPONENT = "ha-agent-panel"


def _integration_version() -> str:
    manifest_path = Path(__file__).parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(manifest.get("version", "1"))
    except (OSError, json.JSONDecodeError):
        return "1"


async def async_register_panel(hass: HomeAssistant) -> None:
    """Register the console panel once per Home Assistant instance."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    if domain_data.get("panel_registered"):
        return

    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    await hass.http.async_register_static_paths(
        [
            StaticPathConfig(
                PANEL_STATIC_URL,
                frontend_dir,
                cache_headers=False,
            )
        ]
    )

    module_url = f"{PANEL_STATIC_URL}/ha-agent-panel.js?v={_integration_version()}"

    try:
        await panel_custom.async_register_panel(
            hass,
            webcomponent_name=PANEL_COMPONENT,
            frontend_url_path=PANEL_URL_PATH,
            sidebar_title=PANEL_TITLE,
            sidebar_icon=PANEL_ICON,
            module_url=module_url,
            embed_iframe=False,
            require_admin=True,
        )
    except ValueError as err:
        LOGGER.warning("HA Agent panel already registered: %s", err)
    domain_data["panel_registered"] = True

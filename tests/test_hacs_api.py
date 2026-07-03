"""Tests for HACS update helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_hacs_api():
    ha_error = types.ModuleType("homeassistant.exceptions")

    class HomeAssistantError(Exception):
        pass

    ha_error.HomeAssistantError = HomeAssistantError
    sys.modules.setdefault("homeassistant.exceptions", ha_error)

    entity_registry = types.ModuleType("homeassistant.helpers.entity_registry")

    def async_get(_hass):
        return MagicMock()

    entity_registry.async_get = async_get
    sys.modules.setdefault("homeassistant.helpers.entity_registry", entity_registry)

    helpers = types.ModuleType("homeassistant.helpers")
    helpers.entity_registry = entity_registry
    sys.modules.setdefault("homeassistant.helpers", helpers)

    core = types.ModuleType("homeassistant.core")
    sys.modules.setdefault("homeassistant.core", core)

    path = COMPONENT / "api" / "hacs.py"
    spec = importlib.util.spec_from_file_location("ha_agent_hacs_api", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_install_update_force_reinstall_downloads_directly() -> None:
    """Force reinstall should bypass the update entity when no update is pending."""
    hacs_api = _load_hacs_api()
    hass = MagicMock()
    repo = MagicMock()
    repo.async_download_repository = AsyncMock()
    repo.data = MagicMock(installed_version="1.13.8", last_version="1.13.8")

    hacs = MagicMock()
    hacs.repositories.get_by_full_name.return_value = repo
    hass.data = {"hacs": hacs}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            hacs_api,
            "_find_update_entity",
            lambda _hass: "update.ha_agent_update",
        )
        patch.setattr(
            hacs_api,
            "get_update_status",
            lambda _hass: {
                "update_available": False,
                "entity_id": "update.ha_agent_update",
                "installed_version": "1.13.8",
                "latest_version": "1.13.8",
            },
        )
        patch.setattr(
            hacs_api,
            "refresh_repository",
            AsyncMock(return_value={"installed_version": "1.13.10"}),
        )
        result = await hacs_api.install_update(hass, force_reinstall=True)

    repo.async_download_repository.assert_awaited_once()
    hass.services.async_call.assert_not_called()
    assert result["installed"] is True

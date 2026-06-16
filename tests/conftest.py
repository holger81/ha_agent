"""Pytest configuration for HA Agent tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_homeassistant_stubs() -> None:
    if "homeassistant.core" in sys.modules:
        return

    ha_pkg = types.ModuleType("homeassistant")
    ha_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["homeassistant"] = ha_pkg

    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistant:
        pass

    class ServiceCall:
        def __init__(self, data: dict | None = None) -> None:
            self.data = data or {}

    class SupportsResponse:
        ONLY = "only"

    def callback(func):
        return func

    ha_core.HomeAssistant = HomeAssistant
    ha_core.ServiceCall = ServiceCall
    ha_core.SupportsResponse = SupportsResponse
    ha_core.callback = callback
    sys.modules["homeassistant.core"] = ha_core

    ha_entries = types.ModuleType("homeassistant.config_entries")

    class ConfigEntry:
        entry_id = "test-entry"
        version = 1

        def __init__(self) -> None:
            self.data: dict = {}

    ha_entries.ConfigEntry = ConfigEntry
    sys.modules["homeassistant.config_entries"] = ha_entries

    ha_const = types.ModuleType("homeassistant.const")
    ha_const.Platform = types.SimpleNamespace(
        CONVERSATION="conversation",
        SELECT="select",
        SWITCH="switch",
        SENSOR="sensor",
    )
    sys.modules["homeassistant.const"] = ha_const

    ha_exc = types.ModuleType("homeassistant.exceptions")
    ha_exc.HomeAssistantError = type("HomeAssistantError", (Exception,), {})
    sys.modules["homeassistant.exceptions"] = ha_exc

    sys.modules["homeassistant.helpers"] = types.ModuleType("homeassistant.helpers")
    sys.modules["homeassistant.components"] = types.ModuleType(
        "homeassistant.components"
    )
    sys.modules["homeassistant.components.conversation"] = types.ModuleType(
        "homeassistant.components.conversation"
    )


_ensure_homeassistant_stubs()

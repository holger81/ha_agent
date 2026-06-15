"""Diagnostic sensors for HA Agent."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .config_helpers import get_llm_backend, get_mcp_config
from .const import (
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_MODEL_ENABLED,
    CONF_LLM_MODEL,
    DOMAIN,
    LOGGER,
)
from .llm_client import LlmClient
from .mcp_client import McpProxyClient
from .status import get_agent_status, update_agent_status

SCAN_INTERVAL = timedelta(minutes=5)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Agent diagnostic sensors."""
    session = async_get_clientsession(hass)
    llm = LlmClient(session)
    mcp = McpProxyClient(session, get_mcp_config(config_entry))

    async def _async_update() -> dict[str, object]:
        llm_ok = False
        mcp_ok = False
        try:
            await llm.check_connection(get_llm_backend(config_entry))
            llm_ok = True
        except Exception:
            llm_ok = False
        try:
            await mcp.check_health()
            mcp_ok = True
        except Exception:
            mcp_ok = False
        update_agent_status(
            hass,
            config_entry.entry_id,
            llm_reachable=llm_ok,
            mcp_reachable=mcp_ok,
        )
        return {"llm_reachable": llm_ok, "mcp_reachable": mcp_ok}

    coordinator = DataUpdateCoordinator(
        hass,
        logger=LOGGER,
        name=f"{DOMAIN}_{config_entry.entry_id}_health",
        update_method=_async_update,
        update_interval=SCAN_INTERVAL,
    )
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as err:
        LOGGER.warning("HA Agent health check failed during setup: %s", err)

    async_add_entities(
        [
            HaAgentLastRouteSensor(hass, config_entry),
            HaAgentMcpToolCountSensor(hass, config_entry),
            HaAgentChatModelSensor(hass, config_entry),
            HaAgentActionModelSensor(hass, config_entry),
            HaAgentLlmReachableSensor(coordinator, config_entry),
            HaAgentMcpReachableSensor(coordinator, config_entry),
            HaAgentSkillsTotalSensor(hass, config_entry),
            HaAgentSkillsEnabledSensor(hass, config_entry),
            HaAgentActiveSkillSensor(hass, config_entry),
            HaAgentLastSkillImprovedSensor(hass, config_entry),
        ]
    )


class _HaAgentDiagnosticSensor(SensorEntity):
    """Base diagnostic sensor."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

class HaAgentLastRouteSensor(_HaAgentDiagnosticSensor):
    """Show the route used for the last Assist turn."""

    _attr_icon = "mdi:routes"
    _attr_name = "Last route"
    _attr_translation_key = "last_route"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_route"

    @property
    def native_value(self) -> str | None:
        return get_agent_status(self.hass, self._entry.entry_id).get("last_route")


class HaAgentMcpToolCountSensor(_HaAgentDiagnosticSensor):
    """Show how many MCP session tools were loaded."""

    _attr_icon = "mdi:toolbox"
    _attr_name = "MCP tools"
    _attr_translation_key = "mcp_tool_count"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_mcp_tool_count"

    @property
    def native_value(self) -> int | None:
        value = get_agent_status(self.hass, self._entry.entry_id).get("mcp_tool_count")
        return int(value) if value is not None else None


class HaAgentChatModelSensor(_HaAgentDiagnosticSensor):
    """Show the configured chat model."""

    _attr_icon = "mdi:brain"
    _attr_name = "Chat model"
    _attr_translation_key = "chat_model_status"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_chat_model_status"

    @property
    def native_value(self) -> str | None:
        return self._entry.data.get(CONF_LLM_MODEL)


class HaAgentActionModelSensor(_HaAgentDiagnosticSensor):
    """Show the configured action model."""

    _attr_icon = "mdi:lightning-bolt"
    _attr_name = "Action model"
    _attr_translation_key = "action_model_status"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_action_model_status"

    @property
    def native_value(self) -> str | None:
        if not self._entry.data.get(CONF_ACTION_MODEL_ENABLED):
            return "disabled"
        return self._entry.data.get(CONF_ACTION_LLM_MODEL) or "not set"


class _HaAgentReachableSensor(SensorEntity):
    """Connectivity sensor backed by the health coordinator."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options: ClassVar[list[str]] = ["online", "offline"]

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[dict[str, object]],
        entry: ConfigEntry,
        *,
        key: str,
        unique_suffix: str,
        translation_key: str,
        name: str,
        icon: str,
    ) -> None:
        self.coordinator = coordinator
        self._entry = entry
        self._key = key
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_translation_key = translation_key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self._handle_coordinator_update)
        )

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success

    @property
    def native_value(self) -> str:
        if self.coordinator.data and self.coordinator.data.get(self._key):
            return "online"
        return "offline"


class HaAgentLlmReachableSensor(_HaAgentReachableSensor):
    """Show whether the LLM server is reachable."""

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[dict[str, object]],
        entry: ConfigEntry,
    ) -> None:
        super().__init__(
            coordinator,
            entry,
            key="llm_reachable",
            unique_suffix="llm_reachable",
            translation_key="llm_reachable",
            name="LLM server",
            icon="mdi:server",
        )


class HaAgentMcpReachableSensor(_HaAgentReachableSensor):
    """Show whether the MCP Proxy is reachable."""

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[dict[str, object]],
        entry: ConfigEntry,
    ) -> None:
        super().__init__(
            coordinator,
            entry,
            key="mcp_reachable",
            unique_suffix="mcp_reachable",
            translation_key="mcp_reachable",
            name="MCP Proxy",
            icon="mdi:lan-connect",
        )


class HaAgentSkillsTotalSensor(_HaAgentDiagnosticSensor):
    """Show total saved skills."""

    _attr_icon = "mdi:book-open-page-variant"
    _attr_name = "Skills total"
    _attr_translation_key = "skills_total"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_skills_total"

    @property
    def native_value(self) -> int | None:
        value = get_agent_status(self.hass, self._entry.entry_id).get("skills_total")
        return int(value) if value is not None else 0


class HaAgentSkillsEnabledSensor(_HaAgentDiagnosticSensor):
    """Show enabled skill count."""

    _attr_icon = "mdi:book-check"
    _attr_name = "Skills enabled"
    _attr_translation_key = "skills_enabled"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_skills_enabled"

    @property
    def native_value(self) -> int | None:
        value = get_agent_status(self.hass, self._entry.entry_id).get("skills_enabled")
        return int(value) if value is not None else 0


class HaAgentActiveSkillSensor(_HaAgentDiagnosticSensor):
    """Show the best-matching skill for the last turn."""

    _attr_icon = "mdi:star-circle"
    _attr_name = "Active skill"
    _attr_translation_key = "active_skill"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_active_skill"

    @property
    def native_value(self) -> str | None:
        return get_agent_status(self.hass, self._entry.entry_id).get("active_skill")


class HaAgentLastSkillImprovedSensor(_HaAgentDiagnosticSensor):
    """Show the last skill that was auto-improved."""

    _attr_icon = "mdi:auto-fix"
    _attr_name = "Last skill improved"
    _attr_translation_key = "last_skill_improved"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, entry)
        self._attr_unique_id = f"{entry.entry_id}_last_skill_improved"

    @property
    def native_value(self) -> str | None:
        return get_agent_status(self.hass, self._entry.entry_id).get(
            "last_skill_improved"
        )

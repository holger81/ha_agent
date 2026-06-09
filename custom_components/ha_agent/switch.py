"""Switch entities for HA Agent device configuration."""

from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_config_entry

from .const import CONF_ACTION_MODEL_ENABLED, DOMAIN


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Agent switch entities."""
    async_add_entities([HaAgentActionRoutingSwitch(hass, config_entry)])


class HaAgentActionRoutingSwitch(SwitchEntity):
    """Enable routing device commands to a separate action model."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:lightning-bolt"
    _attr_name = "Action model routing"
    _attr_translation_key = "action_model_routing"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the action routing switch."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_action_routing"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    @property
    def is_on(self) -> bool:
        """Return whether action routing is enabled."""
        return bool(self._entry.data.get(CONF_ACTION_MODEL_ENABLED))

    async def async_added_to_hass(self) -> None:
        """Track config entry updates."""
        await super().async_added_to_hass()
        self.async_on_remove(
            async_track_config_entry(
                self.hass,
                self._async_config_entry_updated,
                self._entry.entry_id,
            )
        )

    @callback
    def _async_config_entry_updated(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
    ) -> None:
        """Refresh state when the config entry changes."""
        if entry.entry_id != self._entry.entry_id:
            return
        self._entry = entry
        self.async_write_ha_state()

    async def _async_set_enabled(self, enabled: bool) -> None:
        data = dict(self._entry.data)
        data[CONF_ACTION_MODEL_ENABLED] = enabled
        self.hass.config_entries.async_update_entry(self._entry, data=data)
        await self.hass.config_entries.async_reload(self._entry.entry_id)

    async def async_turn_on(self, **kwargs) -> None:
        """Enable action model routing."""
        await self._async_set_enabled(True)

    async def async_turn_off(self, **kwargs) -> None:
        """Disable action model routing."""
        await self._async_set_enabled(False)

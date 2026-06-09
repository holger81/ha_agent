"""Select entities for HA Agent device configuration."""

from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_config_entry

from .const import (
    CONF_ACTION_LLM_MODEL,
    CONF_ACTION_MODEL_ENABLED,
    CONF_LLM_MODEL,
    DOMAIN,
    LOGGER,
)
from .llm_client import LlmClient
from .llm_models import ModelTarget, async_fetch_model_options


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Agent select entities."""
    async_add_entities(
        [
            HaAgentChatModelSelect(hass, config_entry),
            HaAgentActionModelSelect(hass, config_entry),
        ]
    )


class _HaAgentModelSelectBase(SelectEntity):
    """Base class for LLM model select entities."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_icon = "mdi:brain"

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        *,
        conf_key: str,
        unique_suffix: str,
        translation_key: str,
        target: ModelTarget,
    ) -> None:
        """Initialize the model select entity."""
        self.hass = hass
        self._entry = entry
        self._conf_key = conf_key
        self._target = target
        self._llm = LlmClient(async_get_clientsession(hass))
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_translation_key = translation_key
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }
        self._models: list[str] = []

    @property
    def current_option(self) -> str | None:
        """Return the configured model id."""
        return self._entry.data.get(self._conf_key)

    @property
    def options(self) -> list[str]:
        """Return discovered model ids."""
        current = self.current_option
        if current and current not in self._models:
            return [current, *self._models]
        return list(self._models)

    @property
    def available(self) -> bool:
        """Return whether the entity can be changed."""
        return super().available and self._is_enabled()

    def _is_enabled(self) -> bool:
        return True

    async def async_added_to_hass(self) -> None:
        """Load model list and track config entry updates."""
        await super().async_added_to_hass()
        await self._async_refresh_models()
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
        self.hass.async_create_task(self._async_refresh_models())

    async def _async_refresh_models(self) -> None:
        """Fetch the latest model list from the LLM server."""
        if not self._is_enabled():
            self._models = []
            self.async_write_ha_state()
            return
        try:
            self._models = await async_fetch_model_options(
                self._llm,
                self._entry.data,
                target=self._target,
                current_model=self.current_option,
            )
        except Exception as err:
            LOGGER.warning("Failed to refresh LLM model list: %s", err)
            current = self.current_option
            self._models = [current] if current else []
        self.async_write_ha_state()

    async def async_select_option(self, option: str) -> None:
        """Switch the active LLM model."""
        if option not in self.options:
            raise ValueError(f"Unknown model: {option}")

        data = dict(self._entry.data)
        data[self._conf_key] = option
        self.hass.config_entries.async_update_entry(self._entry, data=data)

        if self._conf_key == CONF_LLM_MODEL:
            device_registry = dr.async_get(self.hass)
            if device := device_registry.async_get_device(
                identifiers={(DOMAIN, self._entry.entry_id)},
            ):
                device_registry.async_update_device(
                    device.id,
                    model=option,
                )

        await self.hass.config_entries.async_reload(self._entry.entry_id)


class HaAgentChatModelSelect(_HaAgentModelSelectBase):
    """Select the chat LLM model."""

    _attr_name = "Chat model"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_LLM_MODEL,
            unique_suffix="chat_model",
            translation_key="chat_model",
            target="chat",
        )


class HaAgentActionModelSelect(_HaAgentModelSelectBase):
    """Select the action LLM model."""

    _attr_name = "Action model"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_ACTION_LLM_MODEL,
            unique_suffix="action_model",
            translation_key="action_model",
            target="action",
        )

    def _is_enabled(self) -> bool:
        return bool(self._entry.data.get(CONF_ACTION_MODEL_ENABLED))

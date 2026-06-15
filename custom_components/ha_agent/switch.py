"""Switch entities for HA Agent device configuration."""

from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_ACTION_MODEL_ENABLED,
    CONF_CONVERSATION_SHOW_REASONING,
    CONF_SKILLS_AUTO_SAVE,
    CONF_SKILLS_LEARNING_ENABLED,
    CONF_SKILLS_USE_ENABLED,
    DOMAIN,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Agent switch entities."""
    async_add_entities(
        [
            HaAgentActionRoutingSwitch(hass, config_entry),
            HaAgentSkillLearningSwitch(hass, config_entry),
            HaAgentSkillAutoSaveSwitch(hass, config_entry),
            HaAgentSkillUseSwitch(hass, config_entry),
            HaAgentShowReasoningSwitch(hass, config_entry),
        ]
    )


class _HaAgentConfigSwitch(SwitchEntity):
    """Base switch that persists a config_entry.data boolean."""

    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        *,
        conf_key: str,
        unique_suffix: str,
        translation_key: str,
        icon: str,
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._conf_key = conf_key
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_translation_key = translation_key
        self._attr_icon = icon
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
        }

    @property
    def is_on(self) -> bool:
        return bool(self._entry.data.get(self._conf_key))

    async def _async_set_enabled(self, enabled: bool) -> None:
        data = dict(self._entry.data)
        data[self._conf_key] = enabled
        self.hass.config_entries.async_update_entry(self._entry, data=data)
        await self.hass.config_entries.async_reload(self._entry.entry_id)

    async def async_turn_on(self, **kwargs) -> None:
        await self._async_set_enabled(True)

    async def async_turn_off(self, **kwargs) -> None:
        await self._async_set_enabled(False)


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


class HaAgentSkillLearningSwitch(_HaAgentConfigSwitch):
    """Enable learning new skills from successful multi-step turns."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_SKILLS_LEARNING_ENABLED,
            unique_suffix="skill_learning",
            translation_key="skill_learning",
            icon="mdi:school",
        )


class HaAgentSkillAutoSaveSwitch(_HaAgentConfigSwitch):
    """Automatically save learned skills without asking."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_SKILLS_AUTO_SAVE,
            unique_suffix="skill_auto_save",
            translation_key="skill_auto_save",
            icon="mdi:content-save-auto",
        )


class HaAgentSkillUseSwitch(_HaAgentConfigSwitch):
    """Use saved skills to speed up similar requests."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_SKILLS_USE_ENABLED,
            unique_suffix="skill_use",
            translation_key="skill_use",
            icon="mdi:lightning-bolt-outline",
        )


class HaAgentShowReasoningSwitch(_HaAgentConfigSwitch):
    """Show model reasoning and tool progress in the Assist chat log."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            entry,
            conf_key=CONF_CONVERSATION_SHOW_REASONING,
            unique_suffix="show_reasoning",
            translation_key="show_reasoning",
            icon="mdi:head-lightbulb-outline",
        )

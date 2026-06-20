"""Recovery-hint API for the HA Agent console."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..recovery_hints import RecoveryHint, get_recovery_hint_store
from .serialize import recovery_hint_to_dict


async def list_recovery_hints(
    hass: HomeAssistant,
    entry_id: str,
) -> list[dict[str, Any]]:
    """Return all editable recovery-hint rules for an entry."""
    store = get_recovery_hint_store(hass, entry_id)
    hints = await hass.async_add_executor_job(store.list_hints)
    return [recovery_hint_to_dict(hint) for hint in hints]


async def create_recovery_hint(
    hass: HomeAssistant,
    entry_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a custom recovery-hint rule."""
    title = str(payload.get("title", "")).strip()
    body = str(payload.get("body", "")).strip()
    tool_substring = str(payload.get("tool_substring", "")).strip()
    error_pattern = str(payload.get("error_pattern", "")).strip()
    if not title or not body:
        raise HomeAssistantError("title and body are required")
    if not tool_substring and not error_pattern:
        raise HomeAssistantError(
            "A tool-name substring or an error-text pattern is required"
        )
    store = get_recovery_hint_store(hass, entry_id)

    def _create() -> RecoveryHint:
        return store.create_hint(
            title=title,
            body=body,
            tool_substring=tool_substring,
            error_pattern=error_pattern,
            enabled=bool(payload.get("enabled", True)),
        )

    hint = await hass.async_add_executor_job(_create)
    return recovery_hint_to_dict(hint)


async def delete_recovery_hint(
    hass: HomeAssistant,
    entry_id: str,
    rule_id: str,
) -> bool:
    """Delete a custom recovery-hint rule (built-ins cannot be deleted)."""
    store = get_recovery_hint_store(hass, entry_id)
    deleted = await hass.async_add_executor_job(store.delete_hint, rule_id)
    if not deleted:
        raise HomeAssistantError(
            f"Cannot delete recovery hint (not found or built-in): {rule_id}"
        )
    return True


async def update_recovery_hint(
    hass: HomeAssistant,
    entry_id: str,
    rule_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Update an editable recovery-hint rule."""
    store = get_recovery_hint_store(hass, entry_id)
    title = payload.get("title")
    body = payload.get("body")
    tool_substring = payload.get("tool_substring")
    error_pattern = payload.get("error_pattern")
    enabled = payload.get("enabled")

    def _update() -> RecoveryHint | None:
        return store.update_hint(
            rule_id,
            title=None if title is None else str(title),
            body=None if body is None else str(body),
            tool_substring=None if tool_substring is None else str(tool_substring),
            error_pattern=None if error_pattern is None else str(error_pattern),
            enabled=None if enabled is None else bool(enabled),
        )

    hint = await hass.async_add_executor_job(_update)
    if hint is None:
        raise HomeAssistantError(f"Recovery hint not found: {rule_id}")
    return recovery_hint_to_dict(hint)


async def set_recovery_hint_enabled(
    hass: HomeAssistant,
    entry_id: str,
    rule_id: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a recovery-hint rule."""
    store = get_recovery_hint_store(hass, entry_id)

    def _set() -> RecoveryHint | None:
        return store.update_hint(rule_id, enabled=enabled)

    hint = await hass.async_add_executor_job(_set)
    if hint is None:
        raise HomeAssistantError(f"Recovery hint not found: {rule_id}")
    return recovery_hint_to_dict(hint)


async def reset_recovery_hint(
    hass: HomeAssistant,
    entry_id: str,
    rule_id: str,
) -> dict[str, Any]:
    """Restore a built-in recovery-hint rule to its shipped default."""
    store = get_recovery_hint_store(hass, entry_id)

    def _reset() -> RecoveryHint | None:
        return store.reset_hint(rule_id)

    hint = await hass.async_add_executor_job(_reset)
    if hint is None:
        raise HomeAssistantError(f"Recovery hint not found: {rule_id}")
    return recovery_hint_to_dict(hint)

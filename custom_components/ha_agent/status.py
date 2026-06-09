"""Runtime status exposed via diagnostic entities."""

from __future__ import annotations

from typing import Any, TypedDict

from homeassistant.core import HomeAssistant, callback

from .const import DATA_KEY
from .router import TaskRoute

STATUS_KEY = "status"


class AgentStatus(TypedDict, total=False):
    """Per-entry runtime status."""

    last_route: str
    mcp_tool_count: int
    llm_reachable: bool
    mcp_reachable: bool
    last_error: str | None


@callback
def _status_store(hass: HomeAssistant) -> dict[str, AgentStatus]:
    domain_data = hass.data.setdefault(DATA_KEY, {})
    return domain_data.setdefault(STATUS_KEY, {})


@callback
def get_agent_status(hass: HomeAssistant, entry_id: str) -> AgentStatus:
    """Return stored status for a config entry."""
    return dict(_status_store(hass).get(entry_id, {}))


@callback
def update_agent_status(
    hass: HomeAssistant,
    entry_id: str,
    **fields: Any,
) -> None:
    """Merge fields into the stored status for a config entry."""
    store = _status_store(hass)
    current: AgentStatus = dict(store.get(entry_id, {}))
    current.update(fields)
    store[entry_id] = current


@callback
def record_route(
    hass: HomeAssistant,
    entry_id: str,
    route: TaskRoute,
) -> None:
    """Store the route used for the latest agent turn."""
    update_agent_status(hass, entry_id, last_route=route.value)

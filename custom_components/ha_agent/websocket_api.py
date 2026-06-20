"""WebSocket API for the HA Agent console."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError

from .activity import list_turns
from .api import chat as chat_api
from .api import config as config_api
from .api import playbooks as playbooks_api
from .api import skills as skills_api
from .api.helpers import (
    config_snapshot,
    entry_summaries,
    get_entry,
    require_admin,
)
from .status import get_agent_status
from .threads import (
    async_delete_thread,
    async_save_threads,
    list_threads,
    search_threads,
    upsert_thread,
)


@callback
def async_register_handlers(hass: HomeAssistant) -> None:
    """Register HA Agent websocket commands."""
    websocket_api.async_register_command(hass, ws_subscribe)
    websocket_api.async_register_command(hass, ws_status)
    websocket_api.async_register_command(hass, ws_chat_send)
    websocket_api.async_register_command(hass, ws_chat_history_list)
    websocket_api.async_register_command(hass, ws_chat_history_clear)
    websocket_api.async_register_command(hass, ws_skills_list)
    websocket_api.async_register_command(hass, ws_skills_search)
    websocket_api.async_register_command(hass, ws_skills_get)
    websocket_api.async_register_command(hass, ws_skills_set_enabled)
    websocket_api.async_register_command(hass, ws_skills_delete)
    websocket_api.async_register_command(hass, ws_skills_create)
    websocket_api.async_register_command(hass, ws_skills_update)
    websocket_api.async_register_command(hass, ws_skills_pending_get)
    websocket_api.async_register_command(hass, ws_skills_pending_confirm)
    websocket_api.async_register_command(hass, ws_skills_pending_dismiss)
    websocket_api.async_register_command(hass, ws_skills_export)
    websocket_api.async_register_command(hass, ws_skills_import)
    websocket_api.async_register_command(hass, ws_playbooks_list)
    websocket_api.async_register_command(hass, ws_playbooks_create)
    websocket_api.async_register_command(hass, ws_playbooks_update)
    websocket_api.async_register_command(hass, ws_playbooks_delete)
    websocket_api.async_register_command(hass, ws_playbooks_set_enabled)
    websocket_api.async_register_command(hass, ws_playbooks_reset)
    websocket_api.async_register_command(hass, ws_config_get)
    websocket_api.async_register_command(hass, ws_config_set)
    websocket_api.async_register_command(hass, ws_activity_list)
    websocket_api.async_register_command(hass, ws_threads_list)
    websocket_api.async_register_command(hass, ws_threads_update)
    websocket_api.async_register_command(hass, ws_threads_delete)


def _entry_id_schema(extra: dict | None = None) -> dict:
    schema = {vol.Required("entry_id"): str}
    if extra:
        schema.update(extra)
    return schema


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/subscribe",
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_subscribe(hass: HomeAssistant, connection, msg: dict) -> None:
    """Return entry list and config snapshot."""
    require_admin(connection)
    entries = entry_summaries(hass)
    entry_id = msg.get("entry_id") or (entries[0]["entry_id"] if entries else None)
    config = config_snapshot(hass, get_entry(hass, entry_id)) if entry_id else None
    connection.send_message(
        websocket_api.result_message(
            msg["id"],
            {
                "entries": entries,
                "entry_id": entry_id,
                "config": config,
                "status": get_agent_status(hass, entry_id) if entry_id else {},
            },
        )
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/status",
        **_entry_id_schema(),
    }
)
@websocket_api.async_response
async def ws_status(hass: HomeAssistant, connection, msg: dict) -> None:
    """Return runtime status for an entry."""
    require_admin(connection)
    entry_id = msg["entry_id"]
    connection.send_message(
        websocket_api.result_message(
            msg["id"],
            get_agent_status(hass, entry_id),
        )
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/chat/send",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
        vol.Required("text"): str,
    }
)
@websocket_api.async_response
async def ws_chat_send(hass: HomeAssistant, connection, msg: dict) -> None:
    """Start a chat turn; stream deltas via events, ack immediately."""
    require_admin(connection)
    entry_id = msg["entry_id"]
    conversation_id = msg["conversation_id"]
    chat_api.start_chat(
        hass,
        entry_id=entry_id,
        conversation_id=conversation_id,
        text=msg["text"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"started": True})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/chat/history/list",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_chat_history_list(hass: HomeAssistant, connection, msg: dict) -> None:
    """Return stored conversation history."""
    require_admin(connection)
    history = chat_api.list_history(
        hass,
        msg["entry_id"],
        msg["conversation_id"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"history": history})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/chat/history/clear",
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_chat_history_clear(hass: HomeAssistant, connection, msg: dict) -> None:
    """Clear conversation history."""
    require_admin(connection)
    chat_api.clear_history(hass, msg["conversation_id"])
    connection.send_message(websocket_api.result_message(msg["id"], {"success": True}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/list",
        **_entry_id_schema(
            {
                vol.Optional("limit", default=50): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=200)
                ),
                vol.Optional("offset", default=0): vol.All(
                    vol.Coerce(int), vol.Range(min=0)
                ),
            }
        ),
    }
)
@websocket_api.async_response
async def ws_skills_list(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    result = await skills_api.list_skills(
        hass,
        msg["entry_id"],
        limit=msg.get("limit", 50),
        offset=msg.get("offset", 0),
    )
    connection.send_message(websocket_api.result_message(msg["id"], result))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/search",
        vol.Required("entry_id"): str,
        vol.Required("query"): str,
        vol.Optional("limit", default=20): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
        vol.Optional("enabled_only", default=False): bool,
    }
)
@websocket_api.async_response
async def ws_skills_search(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skills = await skills_api.search_skills(
        hass,
        msg["entry_id"],
        msg["query"],
        limit=msg.get("limit", 20),
        enabled_only=msg.get("enabled_only", False),
    )
    connection.send_message(websocket_api.result_message(msg["id"], {"skills": skills}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/get",
        vol.Required("entry_id"): str,
        vol.Required("skill_id"): str,
    }
)
@websocket_api.async_response
async def ws_skills_get(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skill = await skills_api.get_skill(hass, msg["entry_id"], msg["skill_id"])
    connection.send_message(websocket_api.result_message(msg["id"], {"skill": skill}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/set_enabled",
        vol.Required("entry_id"): str,
        vol.Required("skill_id"): str,
        vol.Required("enabled"): bool,
    }
)
@websocket_api.async_response
async def ws_skills_set_enabled(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skill = await skills_api.set_skill_enabled(
        hass,
        msg["entry_id"],
        msg["skill_id"],
        enabled=msg["enabled"],
    )
    connection.send_message(websocket_api.result_message(msg["id"], {"skill": skill}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/delete",
        vol.Required("entry_id"): str,
        vol.Required("skill_id"): str,
    }
)
@websocket_api.async_response
async def ws_skills_delete(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    await skills_api.delete_skill(hass, msg["entry_id"], msg["skill_id"])
    connection.send_message(websocket_api.result_message(msg["id"], {"success": True}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/create",
        vol.Required("entry_id"): str,
        vol.Required("skill"): dict,
    }
)
@websocket_api.async_response
async def ws_skills_create(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skill = await skills_api.create_skill(hass, msg["entry_id"], msg["skill"])
    connection.send_message(websocket_api.result_message(msg["id"], {"skill": skill}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/update",
        vol.Required("entry_id"): str,
        vol.Required("skill_id"): str,
        vol.Required("skill"): dict,
    }
)
@websocket_api.async_response
async def ws_skills_update(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skill = await skills_api.update_skill(
        hass,
        msg["entry_id"],
        msg["skill_id"],
        msg["skill"],
    )
    connection.send_message(websocket_api.result_message(msg["id"], {"skill": skill}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/pending_get",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_skills_pending_get(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    draft = await skills_api.fetch_pending_draft(
        hass,
        msg["entry_id"],
        msg["conversation_id"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"draft": draft})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/pending_confirm",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_skills_pending_confirm(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skill = await skills_api.confirm_pending_draft(
        hass,
        msg["entry_id"],
        msg["conversation_id"],
    )
    connection.send_message(websocket_api.result_message(msg["id"], {"skill": skill}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/pending_dismiss",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_skills_pending_dismiss(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skills_api.dismiss_pending_draft(
        hass,
        msg["entry_id"],
        msg["conversation_id"],
    )
    connection.send_message(websocket_api.result_message(msg["id"], {"success": True}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/export",
        **_entry_id_schema(),
    }
)
@websocket_api.async_response
async def ws_skills_export(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    skills = await skills_api.export_skills(hass, msg["entry_id"])
    connection.send_message(websocket_api.result_message(msg["id"], {"skills": skills}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/skills/import",
        vol.Required("entry_id"): str,
        vol.Required("skills"): list,
    }
)
@websocket_api.async_response
async def ws_skills_import(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    count = await skills_api.import_skills(hass, msg["entry_id"], msg["skills"])
    connection.send_message(
        websocket_api.result_message(msg["id"], {"imported": count})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/list",
        **_entry_id_schema(),
    }
)
@websocket_api.async_response
async def ws_playbooks_list(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    playbooks = await playbooks_api.list_playbooks(hass, msg["entry_id"])
    connection.send_message(
        websocket_api.result_message(msg["id"], {"playbooks": playbooks})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/create",
        vol.Required("entry_id"): str,
        vol.Required("playbook"): dict,
    }
)
@websocket_api.async_response
async def ws_playbooks_create(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    playbook = await playbooks_api.create_playbook(
        hass,
        msg["entry_id"],
        msg["playbook"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"playbook": playbook})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/delete",
        vol.Required("entry_id"): str,
        vol.Required("route"): str,
    }
)
@websocket_api.async_response
async def ws_playbooks_delete(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    await playbooks_api.delete_playbook(hass, msg["entry_id"], msg["route"])
    connection.send_message(
        websocket_api.result_message(msg["id"], {"success": True})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/update",
        vol.Required("entry_id"): str,
        vol.Required("route"): str,
        vol.Required("playbook"): dict,
    }
)
@websocket_api.async_response
async def ws_playbooks_update(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    playbook = await playbooks_api.update_playbook(
        hass,
        msg["entry_id"],
        msg["route"],
        msg["playbook"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"playbook": playbook})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/set_enabled",
        vol.Required("entry_id"): str,
        vol.Required("route"): str,
        vol.Required("enabled"): bool,
    }
)
@websocket_api.async_response
async def ws_playbooks_set_enabled(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    playbook = await playbooks_api.set_playbook_enabled(
        hass,
        msg["entry_id"],
        msg["route"],
        enabled=msg["enabled"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"playbook": playbook})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/playbooks/reset",
        vol.Required("entry_id"): str,
        vol.Required("route"): str,
    }
)
@websocket_api.async_response
async def ws_playbooks_reset(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    playbook = await playbooks_api.reset_playbook(
        hass,
        msg["entry_id"],
        msg["route"],
    )
    connection.send_message(
        websocket_api.result_message(msg["id"], {"playbook": playbook})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/config/get",
        **_entry_id_schema(),
    }
)
@websocket_api.async_response
async def ws_config_get(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    config = config_api.get_config(hass, msg["entry_id"])
    connection.send_message(websocket_api.result_message(msg["id"], {"config": config}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/config/set",
        vol.Required("entry_id"): str,
        vol.Required("updates"): dict,
    }
)
@websocket_api.async_response
async def ws_config_set(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    config = await config_api.set_config(hass, msg["entry_id"], msg["updates"])
    connection.send_message(websocket_api.result_message(msg["id"], {"config": config}))


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/activity/list",
        **_entry_id_schema(
            {
                vol.Optional("limit", default=50): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=200)
                ),
                vol.Optional("offset", default=0): vol.All(
                    vol.Coerce(int), vol.Range(min=0)
                ),
            }
        ),
    }
)
@websocket_api.async_response
async def ws_activity_list(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    turns, total = list_turns(
        hass,
        msg["entry_id"],
        limit=msg.get("limit", 50),
        offset=msg.get("offset", 0),
    )
    connection.send_message(
        websocket_api.result_message(
            msg["id"],
            {"turns": turns, "total": total},
        )
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/threads/list",
        **_entry_id_schema({vol.Optional("query"): str}),
    }
)
@websocket_api.async_response
async def ws_threads_list(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    query = (msg.get("query") or "").strip()
    if query:
        threads = search_threads(hass, msg["entry_id"], query)
    else:
        threads = list_threads(hass, msg["entry_id"])
    connection.send_message(
        websocket_api.result_message(msg["id"], {"threads": threads})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/threads/update",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
        vol.Optional("title"): str,
        vol.Optional("pinned"): bool,
    }
)
@websocket_api.async_response
async def ws_threads_update(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    thread = upsert_thread(
        hass,
        msg["entry_id"],
        msg["conversation_id"],
        title=msg.get("title"),
        pinned=msg.get("pinned"),
    )
    await async_save_threads(hass, msg["entry_id"])
    connection.send_message(
        websocket_api.result_message(msg["id"], {"thread": thread})
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ha_agent/threads/delete",
        vol.Required("entry_id"): str,
        vol.Required("conversation_id"): str,
    }
)
@websocket_api.async_response
async def ws_threads_delete(hass: HomeAssistant, connection, msg: dict) -> None:
    require_admin(connection)
    entry_id = msg["entry_id"]
    conversation_id = msg["conversation_id"]
    chat_api.cancel_chat_task(hass, entry_id, conversation_id)
    deleted = await async_delete_thread(hass, entry_id, conversation_id)
    if not deleted:
        raise HomeAssistantError("Conversation not found")
    connection.send_message(
        websocket_api.result_message(msg["id"], {"success": True})
    )

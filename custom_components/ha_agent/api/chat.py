"""Chat API for the HA Agent console."""

from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..agent import run_agent
from ..config_helpers import (
    get_agent_config,
    get_llm_backend,
    get_mcp_config,
    get_router_config,
    get_skills_config,
)
from ..const import DATA_KEY
from ..conversation import collect_exposed_entities
from ..llm_client import LlmClient
from ..mcp_client import McpProxyClient
from ..memory import clear_conversation, get_history
from ..status import get_agent_status
from ..threads import upsert_thread
from .helpers import get_entry

CHAT_TASKS_KEY = "chat_tasks"


def _chat_tasks(hass: HomeAssistant) -> dict[tuple[str, str], asyncio.Task]:
    store = hass.data.setdefault(DATA_KEY, {})
    return store.setdefault(CHAT_TASKS_KEY, {})


def _cancel_chat_task(hass: HomeAssistant, entry_id: str, conversation_id: str) -> None:
    key = (entry_id, conversation_id)
    task = _chat_tasks(hass).pop(key, None)
    if task and not task.done():
        task.cancel()


async def stream_chat(
    hass: HomeAssistant,
    *,
    entry_id: str,
    conversation_id: str,
    text: str,
) -> None:
    """Run the agent and stream deltas to the websocket connection."""
    entry = get_entry(hass, entry_id)
    _cancel_chat_task(hass, entry_id, conversation_id)

    async def _run() -> None:
        session = async_get_clientsession(hass)
        llm = LlmClient(session)
        mcp = McpProxyClient(session, get_mcp_config(entry))
        backend = get_llm_backend(entry)
        agent_config = get_agent_config(entry)
        router_config = get_router_config(entry)
        skills_config = get_skills_config(entry)
        exposed = await collect_exposed_entities(hass)
        upsert_thread(
            hass,
            entry_id,
            conversation_id,
            title=text[:48] if text else None,
        )

        payload_base: dict[str, Any] = {
            "entry_id": entry_id,
            "conversation_id": conversation_id,
        }
        try:
            async for delta in run_agent(
                hass,
                llm=llm,
                mcp_client=mcp,
                backend=backend,
                agent_config=agent_config,
                router_config=router_config,
                skills_config=skills_config,
                entry_id=entry_id,
                conversation_id=conversation_id,
                user_text=text,
                exposed_entities=exposed,
            ):
                if not delta.content and not delta.thinking:
                    continue
                hass.bus.async_fire(
                    "ha_agent_chat_delta",
                    {
                        **payload_base,
                        "content": delta.content or None,
                        "thinking": delta.thinking or None,
                    },
                )
            status = get_agent_status(hass, entry_id)
            hass.bus.async_fire(
                "ha_agent_chat_done",
                {
                    **payload_base,
                    "last_route": status.get("last_route"),
                    "active_skill": status.get("active_skill"),
                },
            )
        except asyncio.CancelledError:
            hass.bus.async_fire(
                "ha_agent_chat_done",
                {
                    **payload_base,
                    "cancelled": True,
                },
            )
            raise
        except Exception as err:
            hass.bus.async_fire(
                "ha_agent_chat_done",
                {
                    **payload_base,
                    "error": str(err),
                },
            )

    task = hass.async_create_task(_run(), name=f"ha_agent_chat_{entry_id}")
    _chat_tasks(hass)[(entry_id, conversation_id)] = task
    try:
        await task
    finally:
        _chat_tasks(hass).pop((entry_id, conversation_id), None)


def list_history(
    hass: HomeAssistant,
    entry: str,
    conversation_id: str,
) -> list[dict[str, str]]:
    """Return conversation history for the console."""
    config_entry = get_entry(hass, entry)
    agent_config = get_agent_config(config_entry)
    return get_history(
        hass,
        conversation_id,
        max_turns=agent_config.history_turns,
    )


def clear_history(hass: HomeAssistant, conversation_id: str) -> None:
    """Clear conversation history."""
    clear_conversation(hass, conversation_id)

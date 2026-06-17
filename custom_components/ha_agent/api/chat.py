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
from ..const import DATA_KEY, LOGGER
from ..conversation import collect_exposed_entities
from ..llm_client import LlmClient
from ..mcp_client import McpProxyClient
from ..memory import append_user_message, clear_conversation, get_history
from ..status import get_agent_status
from ..threads import async_save_threads, upsert_thread
from .helpers import get_entry

CHAT_TASKS_KEY = "chat_tasks"
CHAT_TURN_TIMEOUT_PADDING = 60


def _chat_turn_timeout_seconds(entry) -> float:
    """Upper bound for one console chat turn (MCP + agent loop)."""
    llm_timeout = get_llm_backend(entry).timeout
    mcp_timeout = get_mcp_config(entry).timeout
    max_iterations = get_agent_config(entry).max_iterations
    return (
        mcp_timeout
        + llm_timeout * max(max_iterations, 1)
        + CHAT_TURN_TIMEOUT_PADDING
    )


def _chat_tasks(hass: HomeAssistant) -> dict[tuple[str, str], asyncio.Task]:
    store = hass.data.setdefault(DATA_KEY, {})
    return store.setdefault(CHAT_TASKS_KEY, {})


def _cancel_chat_task(hass: HomeAssistant, entry_id: str, conversation_id: str) -> None:
    key = (entry_id, conversation_id)
    task = _chat_tasks(hass).pop(key, None)
    if task and not task.done():
        task.cancel()


def cancel_chat_task(hass: HomeAssistant, entry_id: str, conversation_id: str) -> None:
    """Cancel an in-flight console chat turn for a conversation."""
    _cancel_chat_task(hass, entry_id, conversation_id)


def start_chat(
    hass: HomeAssistant,
    *,
    entry_id: str,
    conversation_id: str,
    text: str,
) -> asyncio.Task[None]:
    """Schedule a chat turn; deltas and completion are sent on the event bus."""
    entry = get_entry(hass, entry_id)
    _cancel_chat_task(hass, entry_id, conversation_id)
    turn_timeout = _chat_turn_timeout_seconds(entry)

    async def _run() -> None:
        session = async_get_clientsession(hass)
        llm = LlmClient(session)
        mcp = McpProxyClient(session, get_mcp_config(entry))
        backend = get_llm_backend(entry)
        agent_config = get_agent_config(entry)
        router_config = get_router_config(entry)
        skills_config = get_skills_config(entry)
        exposed = await collect_exposed_entities(hass)
        append_user_message(
            hass,
            conversation_id,
            text,
            max_turns=agent_config.history_turns,
            entry_id=entry_id,
        )
        upsert_thread(
            hass,
            entry_id,
            conversation_id,
            title=text[:48] if text else None,
        )
        await async_save_threads(hass, entry_id)

        payload_base: dict[str, Any] = {
            "entry_id": entry_id,
            "conversation_id": conversation_id,
        }
        done_payload: dict[str, Any] = {}
        cancelled = False
        try:
            async with asyncio.timeout(turn_timeout):
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
                    if not delta.content and not delta.thinking and not delta.tool:
                        continue
                    hass.bus.async_fire(
                        "ha_agent_chat_delta",
                        {
                            **payload_base,
                            "content": delta.content or None,
                            "thinking": delta.thinking or None,
                            "tool": delta.tool,
                        },
                    )
            status = get_agent_status(hass, entry_id)
            done_payload = {
                "last_route": status.get("last_route"),
                "active_skill": status.get("active_skill"),
            }
        except TimeoutError:
            LOGGER.warning(
                "Console chat timed out after %ss for %s",
                turn_timeout,
                conversation_id,
            )
            done_payload = {
                "error": (
                    f"Chat timed out after {int(turn_timeout)}s. "
                    "Check LLM and MCP connectivity in Settings."
                ),
            }
        except asyncio.CancelledError:
            cancelled = True
            done_payload = {"cancelled": True}
        except Exception as err:
            LOGGER.exception("Console chat failed: %s", err)
            done_payload = {"error": str(err)}
        finally:
            hass.bus.async_fire(
                "ha_agent_chat_done",
                {
                    **payload_base,
                    **done_payload,
                },
            )
        if cancelled:
            raise

    key = (entry_id, conversation_id)
    task = hass.async_create_task(_run(), name=f"ha_agent_chat_{entry_id}")
    _chat_tasks(hass)[key] = task

    def _cleanup(_task: asyncio.Task[None]) -> None:
        _chat_tasks(hass).pop(key, None)

    task.add_done_callback(_cleanup)
    return task


async def stream_chat(
    hass: HomeAssistant,
    *,
    entry_id: str,
    conversation_id: str,
    text: str,
) -> None:
    """Run the agent to completion (used by tests and blocking callers)."""
    task = start_chat(
        hass,
        entry_id=entry_id,
        conversation_id=conversation_id,
        text=text,
    )
    await task


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

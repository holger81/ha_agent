#!/usr/bin/env python3
"""Watch HA Agent chat events via Home Assistant WebSocket API."""

from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from ha_agent_ws import HA_URL, HaAgentWsClient

WATCH_SECONDS = int(os.environ.get("HA_AGENT_WATCH_SECONDS", "300"))
POLL_INTERVAL = float(os.environ.get("HA_AGENT_WATCH_POLL", "2"))


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def watch() -> int:
    client = await HaAgentWsClient.connect()
    try:
        _log(f"Connected to {HA_URL}")
        for event_type in (
            "ha_agent_chat_delta",
            "ha_agent_chat_done",
            "ha_agent_turn_recorded",
        ):
            await client.subscribe_events(event_type)
            _log(f"Subscribed: {event_type}")

        entry_id = client.entry_id
        _log(f"Watching entry_id={entry_id} for {WATCH_SECONDS}s — send a chat now")

        deadline = time.time() + WATCH_SECONDS
        last_poll = 0.0
        active_seen: set[str] = set()
        ws = client._ws

        while time.time() < deadline:
            timeout = min(1.0, max(0.1, deadline - time.time()))
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=timeout)
            except TimeoutError:
                raw = None

            if raw is not None:
                if raw.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(raw.data)
                    if data.get("type") == "event":
                        event = data.get("event", {})
                        etype = event.get("event_type", "")
                        edata = event.get("data", {})
                        if (
                            edata.get("entry_id")
                            and edata.get("entry_id") != entry_id
                        ):
                            continue
                        if etype == "ha_agent_chat_delta":
                            parts = []
                            if edata.get("content"):
                                parts.append(f"content={edata['content'][:120]!r}")
                            if edata.get("thinking"):
                                parts.append(f"thinking={edata['thinking'][:80]!r}")
                            if edata.get("tool"):
                                tool = edata["tool"]
                                parts.append(
                                    f"tool={tool.get('phase')}:{tool.get('name')}"
                                )
                            if edata.get("meta") and edata["meta"].get("route"):
                                parts.append(f"route={edata['meta']['route']}")
                            _log(
                                f"DELTA conv={edata.get('conversation_id')} "
                                + " ".join(parts)
                            )
                        elif etype == "ha_agent_chat_done":
                            _log(
                                f"DONE conv={edata.get('conversation_id')} "
                                f"error={edata.get('error')} "
                                f"route={edata.get('turn_meta', {}).get('route')}"
                            )
                        elif etype == "ha_agent_turn_recorded":
                            turn = edata.get("turn", {})
                            _log(
                                f"RECORDED conv={edata.get('conversation_id')} "
                                f"user={turn.get('user_text', '')[:60]!r} "
                                f"errors={turn.get('tool_errors')} "
                                f"outcome={turn.get('outcome')}"
                            )
                            ts = edata.get("timestamp")
                            if ts is not None:
                                try:
                                    analysis = await client.call(
                                        {
                                            "type": (
                                                "ha_agent/diagnostics/analyze_turn"
                                            ),
                                            "entry_id": entry_id,
                                            "timestamp": ts,
                                        }
                                    )
                                    a = analysis.get("analysis", {})
                                    _log(
                                        f"ANALYSIS severity={a.get('severity')} "
                                        f"summary={a.get('summary')}"
                                    )
                                    for issue in a.get("issues") or []:
                                        if issue.get("kind") != "ok":
                                            _log(
                                                f"  issue [{issue.get('kind')}]: "
                                                f"{issue.get('detail')}"
                                            )
                                except Exception as err:
                                    _log(f"  analyze_turn failed: {err}")
                elif raw.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    _log("WebSocket closed")
                    break

            now = time.time()
            if now - last_poll >= POLL_INTERVAL:
                last_poll = now
                try:
                    obs = await client.call(
                        {
                            "type": "ha_agent/diagnostics/observe",
                            "entry_id": entry_id,
                        }
                    )
                    for active in obs.get("active") or []:
                        cid = active.get("conversation_id", "")
                        key = f"{cid}:{len(active.get('deltas') or [])}"
                        if key not in active_seen and active.get("deltas"):
                            active_seen.add(key)
                            _log(
                                f"LIVE conv={cid} source={active.get('source')} "
                                f"user={active.get('user_text', '')[:50]!r} "
                                f"deltas={len(active.get('deltas') or [])}"
                            )
                except Exception as err:
                    _log(f"observe poll error: {err}")

        _log("Watch ended")
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(watch()))

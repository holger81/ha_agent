#!/usr/bin/env python3
"""Shared Home Assistant WebSocket helpers for HA Agent scripts."""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
WS_URL = (
    HA_URL.replace("https://", "wss://").replace("http://", "ws://")
    + "/api/websocket"
)


class HaAgentWsClient:
    """Minimal HA WebSocket client for HA Agent diagnostics."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
    ):
        self._session = session
        self._ws = ws
        self._msg_id = 0
        self.entry_id: str | None = None

    @classmethod
    async def connect(cls) -> HaAgentWsClient:
        if not HA_TOKEN:
            raise RuntimeError("HA_TOKEN not set")
        session = aiohttp.ClientSession()
        ws = await session.ws_connect(WS_URL)
        hello = await ws.receive_json()
        if hello.get("type") != "auth_required":
            await session.close()
            raise RuntimeError(f"Unexpected hello: {hello}")
        await ws.send_json({"type": "auth", "access_token": HA_TOKEN})
        auth = await ws.receive_json()
        if auth.get("type") != "auth_ok":
            await session.close()
            raise RuntimeError(f"Auth failed: {auth}")
        client = cls(session, ws)
        sub = await client.call({"type": "ha_agent/subscribe"})
        client.entry_id = sub.get("entry_id")
        if not client.entry_id:
            await client.close()
            raise RuntimeError("No ha_agent entry_id (is HA Agent configured?)")
        return client

    async def close(self) -> None:
        await self._ws.close()
        await self._session.close()

    async def call(self, payload: dict[str, Any]) -> Any:
        self._msg_id += 1
        msg_id = self._msg_id
        await self._ws.send_json({**payload, "id": msg_id})
        while True:
            raw = await self._ws.receive()
            if raw.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(raw.data)
            if data.get("id") == msg_id:
                if not data.get("success", True):
                    err = data.get("error", {})
                    raise RuntimeError(err.get("message", str(data)))
                return data.get("result")

    async def subscribe_events(self, event_type: str) -> None:
        await self.call({"type": "subscribe_events", "event_type": event_type})

    async def inject_turn(
        self,
        text: str,
        *,
        conversation_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "ha_agent/diagnostics/inject_turn",
            "entry_id": self.entry_id,
            "text": text,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if timeout is not None:
            payload["timeout"] = timeout
        return await self.call(payload)

    async def analyze_latest(self) -> dict[str, Any]:
        return await self.call(
            {
                "type": "ha_agent/diagnostics/analyze_turn",
                "entry_id": self.entry_id,
                "latest": True,
            }
        )

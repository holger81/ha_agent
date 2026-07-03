"""Tests for diagnostics inject API."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_diagnostics_api():
    mod_name = "ha_agent.api.diagnostics"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.api" not in sys.modules:
        api_pkg = types.ModuleType("ha_agent.api")
        api_pkg.__path__ = [str(COMPONENT / "api")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.api"] = api_pkg

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")
        ha_core.HomeAssistant = object

        def callback(func):
            return func

        ha_core.callback = callback
        sys.modules["homeassistant.core"] = ha_core

    if "homeassistant.helpers.event" not in sys.modules:
        ha_event = types.ModuleType("homeassistant.helpers.event")
        ha_event.async_track_homeassistant_event = lambda *args, **kwargs: lambda: None
        sys.modules["homeassistant.helpers.event"] = ha_event

    chat_stub = types.ModuleType("ha_agent.api.chat")
    chat_stub.start_chat = MagicMock()
    chat_stub._chat_turn_timeout_seconds = MagicMock(return_value=30.0)
    sys.modules["ha_agent.api.chat"] = chat_stub

    helpers_stub = types.ModuleType("ha_agent.api.helpers")
    helpers_stub.get_entry = MagicMock()
    sys.modules["ha_agent.api.helpers"] = helpers_stub

    for rel in (
        "skills/models.py",
        "api/serialize.py",
        "activity.py",
        "diagnostics/analyze.py",
        "diagnostics/live.py",
    ):
        dep_name = "ha_agent." + rel.replace("/", ".").removesuffix(".py")
        if dep_name in sys.modules:
            continue
        path = COMPONENT / rel
        spec = importlib.util.spec_from_file_location(dep_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[dep_name] = module
        spec.loader.exec_module(module)

    path = COMPONENT / "api" / "diagnostics.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_inject_console_turn_waits_for_events() -> None:
    diag = _load_diagnostics_api()
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "entry-1"
    diag.get_entry = MagicMock(return_value=entry)
    hass.bus.async_listen = MagicMock(side_effect=lambda _event, cb: cb)

    handlers: dict[str, object] = {}

    def capture_listen(event_type, callback):
        handlers[event_type] = callback
        return lambda: None

    hass.bus.async_listen.side_effect = capture_listen

    turn_payload = {
        "user_text": "turn on lights",
        "assistant_text": "Done.",
        "tool_errors": 0,
        "outcome": "success",
        "conversation_id": "inject-abc",
    }

    with patch.object(diag, "start_chat") as start_chat_mock:
        task = asyncio.create_task(
            diag.inject_console_turn(
                hass,
                entry_id="entry-1",
                text="turn on lights",
                conversation_id="inject-abc",
            )
        )
        await asyncio.sleep(0)
        assert start_chat_mock.called
        handlers["ha_agent_chat_done"](
            MagicMock(
                data={
                    "entry_id": "entry-1",
                    "conversation_id": "inject-abc",
                    "turn_meta": {"route": "chat"},
                }
            )
        )
        handlers["ha_agent_turn_recorded"](
            MagicMock(
                data={
                    "entry_id": "entry-1",
                    "conversation_id": "inject-abc",
                    "timestamp": 1.0,
                    "turn": turn_payload,
                }
            )
        )
        result = await task

    assert result["conversation_id"] == "inject-abc"
    assert result["analysis"]["severity"] == "ok"
    assert result["turn"]["assistant_text"] == "Done."

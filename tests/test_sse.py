"""Unit tests for shared SSE parsing helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_sse():
    mod_name = "ha_agent.sse"
    sys.modules.pop(mod_name, None)
    if "ha_agent" not in sys.modules:
        import types

        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    path = COMPONENT / "sse.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


sse = _load_sse()


@pytest.mark.asyncio
async def test_iter_sse_events_joins_multiline_data_and_flushes_tail() -> None:
    async def chunks():
        yield b"event: ping\n"
        yield b"data: hello\n"
        yield b"data: world\n"
        yield b"\n"
        # Trailing event without blank line should flush at EOF.
        yield b"data: tail"

    events = [event async for event in sse.iter_sse_events(chunks())]
    assert len(events) == 2
    assert events[0].event == "ping"
    assert events[0].data == "hello\nworld"
    assert events[1].event == ""
    assert events[1].data == "tail"


@pytest.mark.asyncio
async def test_iter_sse_events_handles_split_utf8_and_crlf() -> None:
    async def chunks():
        # Split multi-byte UTF-8 across chunks (emoji).
        text = "data: café 😀\r\n\r\n"
        encoded = text.encode("utf-8")
        yield encoded[:10]
        yield encoded[10:]

    events = [event async for event in sse.iter_sse_events(chunks())]
    assert len(events) == 1
    assert "café" in events[0].data
    assert "😀" in events[0].data


@pytest.mark.asyncio
async def test_iter_sse_data_lines_from_byte_iterable() -> None:
    async def chunks():
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    lines = [line async for line in sse.iter_sse_data_lines(chunks())]
    assert lines == ["one", "two"]

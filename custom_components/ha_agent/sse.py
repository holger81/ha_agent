"""Incremental Server-Sent Events (SSE) parsing for streamed HTTP bodies.

Shared by the LLM chat stream, the llama.cpp ``/models/sse`` watcher and the
MCP streamable-HTTP client. Bytes are decoded with an incremental UTF-8
decoder so multi-byte characters split across network chunks are never
corrupted, multi-line ``data:`` payloads are joined per the SSE spec, CRLF
line endings are accepted, and a trailing event without a final blank line
is flushed at EOF.
"""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class SseEvent:
    """One parsed SSE event."""

    event: str
    data: str


class _SseParser:
    """Line-oriented SSE state machine (spec: WHATWG HTML, section 9.2)."""

    __slots__ = ("_data_lines", "_event_name", "_has_data")

    def __init__(self) -> None:
        self._event_name = ""
        self._data_lines: list[str] = []
        self._has_data = False

    def flush(self) -> SseEvent | None:
        """Dispatch the pending event, if it carried any ``data:`` lines."""
        event: SseEvent | None = None
        if self._has_data:
            event = SseEvent(event=self._event_name, data="\n".join(self._data_lines))
        self._event_name = ""
        self._data_lines = []
        self._has_data = False
        return event

    def feed_line(self, raw_line: str) -> SseEvent | None:
        """Consume one line (without its ``\\n``); return a completed event."""
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if not line:
            return self.flush()
        if line.startswith(":"):
            return None
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "data":
            self._data_lines.append(value)
            self._has_data = True
        elif field == "event":
            self._event_name = value
        return None


async def iter_sse_events(chunks: AsyncIterable[bytes]) -> AsyncIterator[SseEvent]:
    """Yield SSE events from an async iterable of raw byte chunks."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parser = _SseParser()
    buffer = ""

    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            event = parser.feed_line(raw_line)
            if event is not None:
                yield event

    buffer += decoder.decode(b"", final=True)
    if buffer:
        event = parser.feed_line(buffer)
        if event is not None:
            yield event
    tail = parser.flush()
    if tail is not None:
        yield tail


def _response_chunks(response: Any) -> AsyncIterable[bytes]:
    content = getattr(response, "content", None)
    if content is not None and hasattr(content, "iter_any"):
        return content.iter_any()
    if hasattr(response, "iter_any"):
        return response.iter_any()
    return response


async def iter_sse_data_lines(response: Any) -> AsyncIterator[str]:
    """Yield the ``data:`` payload of each SSE event from an aiohttp response.

    Multi-line ``data:`` fields are joined with newlines. Accepts an
    ``aiohttp.ClientResponse`` (reads ``response.content.iter_any()``) or any
    async iterable of ``bytes``.
    """
    async for event in iter_sse_events(_response_chunks(response)):
        yield event.data

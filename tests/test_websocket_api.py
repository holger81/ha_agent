"""Ensure websocket handlers are registered."""

from __future__ import annotations

import ast
import re
from pathlib import Path

WS_FILE = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ha_agent"
    / "websocket_api.py"
)


def _handler_functions(source: str) -> set[str]:
    tree = ast.parse(source)
    handlers: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("ws_"):
            handlers.add(node.name)
    return handlers


def _registered_handlers(source: str) -> set[str]:
    return set(re.findall(r"async_register_command\(hass, (ws_[a-z0-9_]+)\)", source))


def test_all_websocket_handlers_are_registered() -> None:
    source = WS_FILE.read_text(encoding="utf-8")
    defined = _handler_functions(source)
    registered = _registered_handlers(source)
    missing = sorted(defined - registered)
    assert not missing, f"Unregistered websocket handlers: {missing}"

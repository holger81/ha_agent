"""AST checks for websocket admin + entry_id validation (SEC-3 / IN-3)."""

from __future__ import annotations

import ast
from pathlib import Path

WS = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "ha_agent"
    / "websocket_api.py"
)


def test_ws_handlers_with_entry_id_call_get_or_require_entry() -> None:
    source = WS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    missing: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("ws_"):
            continue
        start = max(node.lineno - 25, 1)
        deco = "\n".join(lines[start - 1 : node.lineno])
        if "entry_id" not in deco:
            continue
        body = ast.dump(node)
        if "get_entry" not in body and "_require_entry" not in body:
            if "Optional" in deco and "entry_id" in deco:
                continue
            missing.append(node.name)
    assert missing == [], f"handlers missing entry validation: {missing}"


def test_ws_handlers_call_require_admin() -> None:
    tree = ast.parse(WS.read_text(encoding="utf-8"))
    missing: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("ws_"):
            continue
        body = ast.dump(node)
        if "require_admin" not in body:
            missing.append(node.name)
    assert missing == [], f"handlers missing require_admin: {missing}"

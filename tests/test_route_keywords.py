"""Unit tests for editable action-route trigger keywords."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_route_keywords():
    mod_name = "ha_agent.route_keywords"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "homeassistant.core" not in sys.modules:
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistant:
            pass

        def callback(func):
            return func

        ha_core.HomeAssistant = HomeAssistant
        ha_core.callback = callback
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.core"] = ha_core

    if "ha_agent.const" not in sys.modules:
        const_spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert const_spec and const_spec.loader
        const_mod = importlib.util.module_from_spec(const_spec)
        sys.modules["ha_agent.const"] = const_mod
        const_spec.loader.exec_module(const_mod)

    path = COMPONENT / "route_keywords.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _store(tmp_path):
    rk = _load_route_keywords()
    store = rk.RouteKeywordStore(tmp_path / "route_keywords.db")
    store.connect()
    return rk, store


def test_seeds_default_routes(tmp_path) -> None:
    """Connecting seeds only the action keyword row."""
    rk, store = _store(tmp_path)
    try:
        items = store.list_route_keywords()
        routes = [item.route for item in items]
        assert routes == list(rk.ROUTE_KEYWORD_ROUTES) == ["action"]
        assert all(item.enabled and item.is_default for item in items)
        action = store.get_route_keywords("action")
        assert action is not None
        assert "turn on" in action.keywords
        assert store.get_route_keywords("email") is None
        assert store.get_route_keywords("news") is None
    finally:
        store.close()


def test_purges_obsolete_email_news_rows(tmp_path) -> None:
    """Retired email/news keyword rows are dropped on connect."""
    _rk, store = _store(tmp_path)
    try:
        conn = store._conn
        assert conn is not None
        conn.execute(
            "INSERT INTO route_keywords "
            "(route, title, keywords, enabled, updated_at) "
            "VALUES (?, ?, ?, 1, ?)",
            ("email", "Email", '["inbox"]', 1.0),
        )
        conn.execute(
            "INSERT INTO route_keywords "
            "(route, title, keywords, enabled, updated_at) "
            "VALUES (?, ?, ?, 1, ?)",
            ("news", "News", '["news"]', 1.0),
        )
        conn.commit()
        store._purge_obsolete_routes()
        conn.commit()
        remaining = {
            row["route"]
            for row in conn.execute("SELECT route FROM route_keywords").fetchall()
        }
        assert remaining == {"action"}
        assert [item.route for item in store.list_route_keywords()] == ["action"]
    finally:
        store.close()


def test_default_route_unchanged_falls_back(tmp_path) -> None:
    """A seeded, unchanged route uses the shipped default (active is None)."""
    _rk, store = _store(tmp_path)
    try:
        assert store.active_keywords("action") is None
        assert store.active_keyword_map() == {}
    finally:
        store.close()


def test_update_keywords_active_values(tmp_path) -> None:
    """Customized, enabled keywords become the active override."""
    _rk, store = _store(tmp_path)
    try:
        updated = store.update_route_keywords(
            "action", keywords=["dim", "brighten"]
        )
        assert updated is not None
        assert updated.keywords == ["dim", "brighten"]
        assert updated.is_default is False
        assert store.active_keywords("action") == ["dim", "brighten"]
        assert store.active_keyword_map()["action"] == ["dim", "brighten"]
    finally:
        store.close()


def test_disabled_keywords_fall_back(tmp_path) -> None:
    """Disabling a customized route falls back to the default matcher."""
    _rk, store = _store(tmp_path)
    try:
        store.update_route_keywords("action", keywords=["dim"])
        store.update_route_keywords("action", enabled=False)
        assert store.active_keywords("action") is None
    finally:
        store.close()


def test_empty_keywords_fall_back(tmp_path) -> None:
    """An empty keyword list falls back to the default matcher."""
    _rk, store = _store(tmp_path)
    try:
        store.update_route_keywords("action", keywords=[])
        assert store.active_keywords("action") is None
    finally:
        store.close()


def test_reset_restores_default(tmp_path) -> None:
    """Reset returns the keywords to the shipped default and re-enables."""
    rk, store = _store(tmp_path)
    try:
        store.update_route_keywords("action", keywords=["dim"], enabled=False)
        reset = store.reset_route_keywords("action")
        assert reset is not None
        assert reset.enabled is True
        assert reset.is_default is True
        assert reset.keywords == rk.default_route_keywords("action")
        assert store.active_keywords("action") is None
    finally:
        store.close()


def test_update_normalizes_keywords(tmp_path) -> None:
    """Whitespace is stripped and case-insensitive duplicates removed."""
    _rk, store = _store(tmp_path)
    try:
        updated = store.update_route_keywords(
            "action", keywords=["  Dim ", "dim", "", "Brighten"]
        )
        assert updated is not None
        assert updated.keywords == ["Dim", "Brighten"]
    finally:
        store.close()

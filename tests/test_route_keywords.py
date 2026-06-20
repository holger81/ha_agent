"""Unit tests for editable route trigger keywords."""

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
    """Connecting seeds one keyword row per built-in route."""
    rk, store = _store(tmp_path)
    try:
        items = store.list_route_keywords()
        routes = [item.route for item in items]
        assert routes == list(rk.ROUTE_KEYWORD_ROUTES)
        assert all(item.enabled and item.is_default for item in items)
        email = store.get_route_keywords("email")
        assert email is not None
        assert "inbox" in email.keywords
    finally:
        store.close()


def test_default_route_unchanged_falls_back(tmp_path) -> None:
    """A seeded, unchanged route uses the shipped default (active is None)."""
    _rk, store = _store(tmp_path)
    try:
        assert store.active_keywords("news") is None
        assert store.active_keyword_map() == {}
    finally:
        store.close()


def test_update_keywords_active_values(tmp_path) -> None:
    """Customized, enabled keywords become the active override."""
    _rk, store = _store(tmp_path)
    try:
        updated = store.update_route_keywords(
            "news", keywords=["scoop", "bulletin"]
        )
        assert updated is not None
        assert updated.keywords == ["scoop", "bulletin"]
        assert updated.is_default is False
        assert store.active_keywords("news") == ["scoop", "bulletin"]
        assert store.active_keyword_map()["news"] == ["scoop", "bulletin"]
    finally:
        store.close()


def test_disabled_keywords_fall_back(tmp_path) -> None:
    """Disabling a customized route falls back to the default matcher."""
    _rk, store = _store(tmp_path)
    try:
        store.update_route_keywords("email", keywords=["postbox"])
        store.update_route_keywords("email", enabled=False)
        assert store.active_keywords("email") is None
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
        store.update_route_keywords("news", keywords=["scoop"], enabled=False)
        reset = store.reset_route_keywords("news")
        assert reset is not None
        assert reset.enabled is True
        assert reset.is_default is True
        assert reset.keywords == rk.default_route_keywords("news")
        assert store.active_keywords("news") is None
    finally:
        store.close()


def test_update_normalizes_keywords(tmp_path) -> None:
    """Whitespace is stripped and case-insensitive duplicates removed."""
    _rk, store = _store(tmp_path)
    try:
        updated = store.update_route_keywords(
            "news", keywords=["  Scoop ", "scoop", "", "Bulletin"]
        )
        assert updated is not None
        assert updated.keywords == ["Scoop", "Bulletin"]
    finally:
        store.close()

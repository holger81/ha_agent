"""Unit tests for editable route playbooks."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load_playbooks():
    mod_name = "ha_agent.playbooks"
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

    # Load const dependency under the package namespace.
    if "ha_agent.const" not in sys.modules:
        const_spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert const_spec and const_spec.loader
        const_mod = importlib.util.module_from_spec(const_spec)
        sys.modules["ha_agent.const"] = const_mod
        const_spec.loader.exec_module(const_mod)

    path = COMPONENT / "playbooks.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _store(tmp_path):
    pb = _load_playbooks()
    store = pb.PlaybookStore(tmp_path / "playbooks.db")
    store.connect()
    return pb, store


def test_seeds_default_playbooks(tmp_path) -> None:
    """Connecting seeds one playbook per known route."""
    pb, store = _store(tmp_path)
    try:
        playbooks = store.list_playbooks()
        routes = [p.route for p in playbooks]
        assert routes == list(pb.PLAYBOOK_ROUTES)
        assert all(p.enabled and p.is_default for p in playbooks)
    finally:
        store.close()


def test_playbook_key_for_route_maps_chat_to_general(tmp_path) -> None:
    """Unknown/chat routes fall back to the general playbook."""
    pb = _load_playbooks()
    assert pb.playbook_key_for_route("news") == "news"
    assert pb.playbook_key_for_route("action") == "action"
    assert pb.playbook_key_for_route("chat") == "general"
    assert pb.playbook_key_for_route("unknown") == "general"


def test_update_marks_customized_and_active_body(tmp_path) -> None:
    """Editing a body persists and flips is_default off."""
    _pb, store = _store(tmp_path)
    try:
        updated = store.update_playbook("news", body="Custom news flow")
        assert updated is not None
        assert updated.body == "Custom news flow"
        assert updated.is_default is False
        assert store.active_body("news") == "Custom news flow"
    finally:
        store.close()


def test_disabled_playbook_yields_empty_body(tmp_path) -> None:
    """A disabled playbook injects nothing."""
    _pb, store = _store(tmp_path)
    try:
        store.update_playbook("email", enabled=False)
        assert store.active_body("email") == ""
    finally:
        store.close()


def test_reset_restores_default(tmp_path) -> None:
    """Reset returns the body to the shipped default and re-enables it."""
    pb, store = _store(tmp_path)
    try:
        store.update_playbook("general", body="changed", enabled=False)
        reset = store.reset_playbook("general")
        assert reset is not None
        assert reset.enabled is True
        assert reset.is_default is True
        assert reset.body == pb.default_playbook_body("general")
    finally:
        store.close()


def test_create_and_delete_custom_playbook(tmp_path) -> None:
    """Custom rules can be added and removed; built-ins cannot be deleted."""
    _pb, store = _store(tmp_path)
    try:
        created = store.create_playbook(
            title="Vacuum",
            body="Start the robot vacuum.",
            match_text="The user wants to vacuum or clean the floors.",
        )
        assert created.is_builtin is False
        assert created.route.startswith("custom-")
        assert store.custom_count() == 1

        routes = [p.route for p in store.list_playbooks()]
        assert created.route in routes

        assert store.delete_playbook("news") is False  # built-in protected
        assert store.delete_playbook(created.route) is True
        assert store.custom_count() == 0
    finally:
        store.close()


def test_update_custom_match_text(tmp_path) -> None:
    """A custom rule's when-to-apply text persists."""
    _pb, store = _store(tmp_path)
    try:
        created = store.create_playbook(
            title="Vacuum", body="b", match_text="clean"
        )
        updated = store.update_playbook(created.route, match_text="vacuum the floor")
        assert updated is not None
        assert updated.match_text == "vacuum the floor"
    finally:
        store.close()


def test_parse_playbook_selection(tmp_path) -> None:
    """The selection parser accepts plain and fenced JSON and validates routes."""
    pb = _load_playbooks()
    valid = {"news", "email"}
    assert pb.parse_playbook_selection('{"route": "news"}', valid) == "news"
    assert (
        pb.parse_playbook_selection('```json\n{"route":"email"}\n```', valid)
        == "email"
    )
    assert pb.parse_playbook_selection('{"route": null}', valid) is None
    assert pb.parse_playbook_selection('{"route": "unknown"}', valid) is None
    assert pb.parse_playbook_selection("not json", valid) is None

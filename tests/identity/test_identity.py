"""Unit tests for agent identity registry and resolution."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


def _ensure_ha_stubs() -> None:
    if "homeassistant.core" in sys.modules:
        return
    ha_pkg = types.ModuleType("homeassistant")
    ha_core = types.ModuleType("homeassistant.core")

    class HomeAssistant:
        pass

    ha_core.HomeAssistant = HomeAssistant
    sys.modules["homeassistant"] = ha_pkg
    sys.modules["homeassistant.core"] = ha_core


def _load_module(relative: str):
    name = f"ha_agent.{relative.replace('/', '.')}"
    if name in sys.modules:
        return sys.modules[name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if relative.startswith("identity/"):
        if "ha_agent.identity" not in sys.modules:
            pkg = types.ModuleType("ha_agent.identity")
            pkg.__path__ = [str(COMPONENT / "identity")]  # type: ignore[attr-defined]
            sys.modules["ha_agent.identity"] = pkg
        path = COMPONENT / f"{relative}.py"
    else:
        path = COMPONENT / f"{relative}.py"
    _ensure_ha_stubs()
    if "ha_agent.const" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.const"] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = _load_module("identity/models")
store_mod = _load_module("identity/store")
runtime_mod = _load_module("identity/runtime")
resolver_mod = _load_module("identity/resolver")
voicebm_mod = _load_module("identity/voicebm")

UserKind = models.UserKind
IdentitySource = models.IdentitySource
IdentityStore = store_mod.IdentityStore
set_identity_override = runtime_mod.set_identity_override
get_identity_override = runtime_mod.get_identity_override
resolve_agent_user = resolver_mod.resolve_agent_user
parse_voice_identity = voicebm_mod.parse_voice_identity


@pytest.fixture
def identity_store() -> IdentityStore:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "identity.db"
        store = IdentityStore(db_path)
        store.connect()
        yield store
        store.close()


def test_store_seeds_four_registered_and_assist_guest(
    identity_store: IdentityStore,
) -> None:
    users = identity_store.list_users()
    registered = [user for user in users if user.kind == UserKind.REGISTERED]
    guests = [user for user in users if user.kind == UserKind.GUEST]
    assert len(registered) == 4
    assert len(guests) == 1
    assert any(user.is_default for user in registered)


def test_update_user_links_ha_login(identity_store: IdentityStore) -> None:
    user = identity_store.list_users(kind=UserKind.REGISTERED)[0]
    updated = identity_store.update_user(
        user.id,
        display_name="Holger",
        ha_user_id="ha-user-1",
        is_default=True,
    )
    assert updated is not None
    assert updated.display_name == "Holger"
    assert updated.ha_user_id == "ha-user-1"
    assert identity_store.get_by_ha_user_id("ha-user-1") is not None


def test_parse_voice_identity_reads_json_block() -> None:
    payload = parse_voice_identity(
        'Some prompt.\nHA_AGENT_IDENTITY: {"speaker_id": "abc", "confidence": 0.91}'
    )
    assert payload is not None
    assert payload["speaker_id"] == "abc"
    assert payload["confidence"] == 0.91


@pytest.mark.asyncio
async def test_resolve_console_user_from_login(
    identity_store: IdentityStore,
) -> None:
    user = identity_store.list_users(kind=UserKind.REGISTERED)[0]
    identity_store.update_user(user.id, ha_user_id="login-1", display_name="Holger")
    hass = MagicMock()

    async def _executor(job):
        return job()

    hass.async_add_executor_job = _executor
    hass.data = {}
    hass.config = MagicMock()
    hass.config.path.return_value = str(identity_store._db_path.parent)

    store_map = {"entry-1": identity_store}
    hass.data["ha_agent"] = {"identity_stores": store_map}

    resolved = await resolve_agent_user(
        hass,
        "entry-1",
        channel="console",
        ha_user_id="login-1",
    )
    assert resolved.user.display_name == "Holger"
    assert resolved.source == IdentitySource.LOGIN


@pytest.mark.asyncio
async def test_resolve_console_admin_override(
    identity_store: IdentityStore,
) -> None:
    users = identity_store.list_users(kind=UserKind.REGISTERED)
    target = users[1]
    hass = MagicMock()

    async def _executor(job):
        return job()

    hass.async_add_executor_job = _executor
    hass.data = {"ha_agent": {"identity_stores": {"entry-1": identity_store}}}
    hass.config = MagicMock()
    hass.config.path.return_value = str(identity_store._db_path.parent)

    set_identity_override(hass, "entry-1", target.id)
    resolved = await resolve_agent_user(
        hass,
        "entry-1",
        channel="console",
        ha_user_id="login-1",
    )
    assert resolved.user.id == target.id
    assert resolved.source == IdentitySource.OVERRIDE


@pytest.mark.asyncio
async def test_resolve_assist_without_voice_uses_guest(
    identity_store: IdentityStore,
) -> None:
    hass = MagicMock()

    async def _executor(job):
        return job()

    hass.async_add_executor_job = _executor
    hass.data = {"ha_agent": {"identity_stores": {"entry-1": identity_store}}}
    hass.config = MagicMock()
    hass.config.path.return_value = str(identity_store._db_path.parent)

    resolved = await resolve_agent_user(
        hass,
        "entry-1",
        channel="assist",
    )
    assert resolved.user.kind == UserKind.GUEST
    assert resolved.source == IdentitySource.ASSIST_GUEST

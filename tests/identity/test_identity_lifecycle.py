"""Tests for guest promote and merge lifecycle."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import uuid
from pathlib import Path

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


store_mod = _load_module("identity/store")

IdentityStore = store_mod.IdentityStore
UserKind = _load_module("identity/models").UserKind

VECTOR_A = [1.0, 0.0, 0.0]
VECTOR_B = [0.8, 0.2, 0.0]


@pytest.fixture
def identity_store() -> IdentityStore:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "identity.db"
        store = IdentityStore(db_path)
        store.connect()
        yield store
        store.close()


def _create_guest_with_voice(store: IdentityStore, name: str, vector: list[float]):
    guest = store.create_guest(name)
    store.create_voice_profile(
        profile_id=str(uuid.uuid4()),
        agent_user_id=guest.id,
        backend="sherpa-onnx",
        model="test",
        centroid=vector,
        match_confidence=0.9,
    )
    return guest


def test_promote_guest_moves_voice_profile_to_registered(
    identity_store: IdentityStore,
) -> None:
    guest = _create_guest_with_voice(identity_store, "Guest 2", VECTOR_A)
    registered = identity_store.list_users(kind=UserKind.REGISTERED)[0]

    updated = identity_store.promote_guest(
        guest.id,
        registered.id,
        display_name="Holger",
    )

    assert updated.display_name == "Holger"
    assert identity_store.get_voice_profile_for_user(guest.id) is None
    profile = identity_store.get_voice_profile_for_user(registered.id)
    assert profile is not None
    assert profile.sample_count == 1
    archived = identity_store.get_user(guest.id)
    assert archived is not None
    assert archived.merged_into == registered.id


def test_promote_guest_merges_existing_registered_voice_profile(
    identity_store: IdentityStore,
) -> None:
    guest = _create_guest_with_voice(identity_store, "Guest 2", VECTOR_A)
    registered = identity_store.list_users(kind=UserKind.REGISTERED)[0]
    identity_store.create_voice_profile(
        profile_id=str(uuid.uuid4()),
        agent_user_id=registered.id,
        backend="sherpa-onnx",
        model="test",
        centroid=VECTOR_B,
        match_confidence=0.88,
    )

    identity_store.promote_guest(guest.id, registered.id)
    profile = identity_store.get_voice_profile_for_user(registered.id)
    assert profile is not None
    assert profile.sample_count == 2


def test_merge_guests_combines_voice_profiles(
    identity_store: IdentityStore,
) -> None:
    guest_a = _create_guest_with_voice(identity_store, "Guest 2", VECTOR_A)
    guest_b = _create_guest_with_voice(identity_store, "Guest 3", VECTOR_B)

    survivor = identity_store.merge_guests(
        [guest_a.id, guest_b.id],
        survivor_id=guest_b.id,
    )

    assert survivor.id == guest_b.id
    profile = identity_store.get_voice_profile_for_user(guest_b.id)
    assert profile is not None
    assert profile.sample_count == 2
    archived = identity_store.get_user(guest_a.id)
    assert archived is not None
    assert archived.merged_into == guest_b.id
    assert identity_store.get_voice_profile_for_user(guest_a.id) is None


def test_merge_guests_requires_two_profiles(identity_store: IdentityStore) -> None:
    guest = _create_guest_with_voice(identity_store, "Guest 2", VECTOR_A)
    with pytest.raises(ValueError, match="At least two guests"):
        identity_store.merge_guests([guest.id])

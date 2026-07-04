"""Tests for Phase 9 identity polish features."""

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
        data: dict

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
    if relative.startswith("identity/") and "ha_agent.identity" not in sys.modules:
        pkg = types.ModuleType("ha_agent.identity")
        pkg.__path__ = [str(COMPONENT / "identity")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.identity"] = pkg
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


naming_mod = _load_module("identity/naming")
store_mod = _load_module("identity/store")
clustering_mod = _load_module("identity/clustering")
config_mod = _load_module("identity/config")
runtime_mod = _load_module("identity/runtime")
activity_mod = _load_module("activity")
models = _load_module("identity/models")

IdentityStore = store_mod.IdentityStore
IdentityVoiceConfig = config_mod.IdentityVoiceConfig
SpeakerMatch = models.SpeakerMatch
UserKind = models.UserKind
extract_self_intro_name = naming_mod.extract_self_intro_name
is_default_guest_name = naming_mod.is_default_guest_name
resolve_speaker_embedding = clustering_mod.resolve_speaker_embedding
enroll_speaker_embedding = clustering_mod.enroll_speaker_embedding
set_enrollment_target = runtime_mod.set_enrollment_target
record_turn = activity_mod.record_turn
update_turn_identity = activity_mod.update_turn_identity
TurnTrace = _load_module("skills/models").TurnTrace

VECTOR_A = [1.0, 0.0, 0.0]
VECTOR_B = [0.9, 0.1, 0.0]
VECTOR_LOW = [0.7, 0.71414284, 0.0]


@pytest.fixture
def identity_store() -> IdentityStore:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "identity.db"
        store = IdentityStore(db_path)
        store.connect()
        yield store
        store.close()


def test_extract_self_intro_name() -> None:
    assert extract_self_intro_name("Hi, I'm Dave") == "Dave"
    assert extract_self_intro_name("My name is Anna") == "Anna"
    assert extract_self_intro_name("turn on the lights") is None
    assert extract_self_intro_name("I'm sorry") is None


def test_is_default_guest_name() -> None:
    assert is_default_guest_name("Guest 1") is True
    assert is_default_guest_name("Dave") is False


def test_auto_names_new_guest_from_stt(identity_store: IdentityStore) -> None:
    resolved = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_A,
            quality="ok",
            duration_ms=1200,
        ),
        config=IdentityVoiceConfig(
            guest_create_threshold=0.75,
            auto_name_enabled=True,
        ),
        user_text="Hello, I'm Dave",
    )
    assert resolved is not None
    assert resolved.user.display_name == "Dave"


def test_low_confidence_match_skips_centroid_update(
    identity_store: IdentityStore,
) -> None:
    config = IdentityVoiceConfig(
        guest_match_threshold=0.80,
        guest_create_threshold=0.55,
    )
    first = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_A,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert first is not None
    profile = identity_store.get_voice_profile_for_user(first.user.id)
    assert profile is not None
    assert profile.sample_count == 1

    second = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_LOW,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert second is not None
    assert second.user.id == first.user.id
    profile = identity_store.get_voice_profile_for_user(first.user.id)
    assert profile is not None
    assert profile.sample_count == 1


def test_enroll_voice_sample_for_registered_member(
    identity_store: IdentityStore,
) -> None:
    registered = identity_store.list_users(kind=UserKind.REGISTERED)[0]
    resolved = enroll_speaker_embedding(
        identity_store,
        registered.id,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_A,
            quality="ok",
            duration_ms=1200,
        ),
        config=IdentityVoiceConfig(),
    )
    assert resolved is not None
    assert resolved.user.id == registered.id
    profile = identity_store.get_voice_profile_for_user(registered.id)
    assert profile is not None
    assert profile.sample_count == 1


def test_update_turn_identity() -> None:
    hass = MagicMock()
    hass.data = {"ha_agent": {}}
    hass.bus = MagicMock()
    trace = TurnTrace(user_text="hello", history_len=0, conversation_id="conv-1")
    trace.agent_user_id = "guest-1"
    trace.agent_user_display_name = "Guest 1"
    trace.agent_user_kind = "guest"
    record_turn(hass, "entry-1", trace)
    turns, _ = activity_mod.list_turns(hass, "entry-1")
    timestamp = turns[0]["timestamp"]

    updated = update_turn_identity(
        hass,
        "entry-1",
        timestamp,
        agent_user_id="member-1",
        agent_user_display_name="Member 1",
        agent_user_kind="registered",
        corrected_by_ha_user_id="admin",
        original_user_id="guest-1",
        original_display_name="Guest 1",
    )
    assert updated is not None
    assert updated["agent_user_display_name"] == "Member 1"
    assert updated["identity_source"] == "corrected"


def test_enrollment_runtime() -> None:
    hass = MagicMock()
    hass.data = {"ha_agent": {}}
    session = set_enrollment_target(hass, "entry-1", "member-1")
    assert session is not None
    assert session.agent_user_id == "member-1"
    assert runtime_mod.get_enrollment_session(hass, "entry-1") is session

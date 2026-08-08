"""Tests for speaker embedding clustering."""

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
embeddings_mod = _load_module("identity/embeddings")
clustering_mod = _load_module("identity/clustering")
config_mod = _load_module("identity/config")
resolver_mod = _load_module("identity/resolver")

UserKind = models.UserKind
IdentitySource = models.IdentitySource
SpeakerMatch = models.SpeakerMatch
IdentityStore = store_mod.IdentityStore
IdentityVoiceConfig = config_mod.IdentityVoiceConfig
cosine_similarity = embeddings_mod.cosine_similarity
resolve_speaker_embedding = clustering_mod.resolve_speaker_embedding
resolve_agent_user = resolver_mod.resolve_agent_user

VECTOR_A = [1.0, 0.0, 0.0]
VECTOR_B = [0.9, 0.1, 0.0]
VECTOR_C = [0.0, 1.0, 0.0]


@pytest.fixture
def identity_store() -> IdentityStore:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "identity.db"
        store = IdentityStore(db_path)
        store.connect()
        yield store
        store.close()


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity(VECTOR_A, VECTOR_A) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity(VECTOR_A, VECTOR_C) == pytest.approx(0.0)


def test_cluster_creates_first_voice_guest(identity_store: IdentityStore) -> None:
    match = SpeakerMatch(
        backend="sherpa-onnx",
        embedding=VECTOR_A,
        quality="ok",
        duration_ms=1200,
    )
    resolved = resolve_speaker_embedding(
        identity_store,
        match,
        config=IdentityVoiceConfig(guest_create_threshold=0.75),
    )
    assert resolved is not None
    assert resolved.user.kind == UserKind.GUEST
    assert resolved.user.display_name == "Guest 1"
    assert resolved.source == IdentitySource.VOICE
    profile = identity_store.get_voice_profile_for_user(resolved.user.id)
    assert profile is not None
    assert profile.sample_count == 1


def test_cluster_reuses_same_guest_for_similar_voice(
    identity_store: IdentityStore,
) -> None:
    config = IdentityVoiceConfig(guest_create_threshold=0.75)
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
    second = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_B,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert first is not None and second is not None
    assert first.user.id == second.user.id
    profile = identity_store.get_voice_profile_for_user(first.user.id)
    assert profile is not None
    assert profile.sample_count == 2


def test_cluster_prefers_recent_profile_on_close_scores(
    identity_store: IdentityStore,
) -> None:
    config = IdentityVoiceConfig(
        guest_create_threshold=0.50,
        guest_tie_margin=0.05,
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
    second = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_C,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert first is not None and second is not None
    assert first.user.id != second.user.id

    profile_first = identity_store.get_voice_profile_for_user(first.user.id)
    profile_second = identity_store.get_voice_profile_for_user(second.user.id)
    assert profile_first is not None and profile_second is not None
    identity_store.update_voice_profile_centroid(
        profile_first.id,
        embedding=VECTOR_A,
        match_confidence=0.99,
    )
    identity_store.update_voice_profile_centroid(
        profile_second.id,
        embedding=VECTOR_C,
        match_confidence=0.99,
    )

    blended = [0.70710678, 0.70710678, 0.0]
    resolved = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=blended,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert resolved is not None
    assert resolved.user.id == second.user.id


def test_cluster_creates_second_guest_for_different_voice(
    identity_store: IdentityStore,
) -> None:
    config = IdentityVoiceConfig(guest_create_threshold=0.75)
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
    second = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_C,
            quality="ok",
            duration_ms=1200,
        ),
        config=config,
    )
    assert first is not None and second is not None
    assert first.user.id != second.user.id
    assert second.user.display_name == "Guest 2"


def test_cluster_skips_too_short_utterance(identity_store: IdentityStore) -> None:
    resolved = resolve_speaker_embedding(
        identity_store,
        SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_A,
            quality="ok",
            duration_ms=400,
        ),
        config=IdentityVoiceConfig(min_utterance_ms=800),
    )
    assert resolved is None


@pytest.mark.asyncio
async def test_resolve_assist_with_speaker_match(
    identity_store: IdentityStore,
) -> None:
    hass = MagicMock()

    async def _executor(job):
        return job()

    hass.async_add_executor_job = _executor
    hass.data = {"ha_agent": {"identity_stores": {"entry-1": identity_store}}}
    hass.config = MagicMock()
    hass.config.path.return_value = str(identity_store._db_path.parent)

    match = SpeakerMatch(
        backend="sherpa-onnx",
        embedding=VECTOR_A,
        quality="ok",
        duration_ms=1500,
    )
    resolved = await resolve_agent_user(
        hass,
        "entry-1",
        channel="assist",
        speaker_match=match,
        voice_config=IdentityVoiceConfig(guest_create_threshold=0.75),
    )
    assert resolved.source == IdentitySource.VOICE
    assert resolved.user.display_name == "Guest 1"

    resolved_again = await resolve_agent_user(
        hass,
        "entry-1",
        channel="assist",
        speaker_match=SpeakerMatch(
            backend="sherpa-onnx",
            embedding=VECTOR_B,
            quality="ok",
            duration_ms=1500,
        ),
        voice_config=IdentityVoiceConfig(guest_create_threshold=0.75),
    )
    assert resolved_again.user.id == resolved.user.id

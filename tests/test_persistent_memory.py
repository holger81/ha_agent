"""Unit tests for durable persistent memory."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


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
    if relative.startswith("persistent_memory/"):
        if "ha_agent.persistent_memory" not in sys.modules:
            pkg = types.ModuleType("ha_agent.persistent_memory")
            pkg.__path__ = [str(COMPONENT / "persistent_memory")]  # type: ignore[attr-defined]
            sys.modules["ha_agent.persistent_memory"] = pkg
        if "ha_agent.identity" not in sys.modules:
            id_pkg = types.ModuleType("ha_agent.identity")
            id_pkg.__path__ = [str(COMPONENT / "identity")]  # type: ignore[attr-defined]
            sys.modules["ha_agent.identity"] = id_pkg
    _ensure_ha_stubs()
    if "ha_agent.const" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.const"] = mod
        spec.loader.exec_module(mod)
    # identity deps for inject guest gating
    for rel in ("identity/models", "identity/config"):
        dep_name = f"ha_agent.{rel.replace('/', '.')}"
        if dep_name not in sys.modules:
            dep_path = COMPONENT / f"{rel}.py"
            dep_spec = importlib.util.spec_from_file_location(dep_name, dep_path)
            assert dep_spec and dep_spec.loader
            dep_mod = importlib.util.module_from_spec(dep_spec)
            sys.modules[dep_name] = dep_mod
            dep_spec.loader.exec_module(dep_mod)
    path = COMPONENT / f"{relative}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = _load_module("persistent_memory/models")
store_mod = _load_module("persistent_memory/store")
intent_mod = _load_module("persistent_memory/intent")
extract_mod = _load_module("persistent_memory/extract")
inject_mod = _load_module("persistent_memory/inject")
identity_models = _load_module("identity/models")

MemoryScope = models.MemoryScope
PersistentMemoryStore = store_mod.PersistentMemoryStore
detect_memory_intent = intent_mod.detect_memory_intent
MemoryIntentKind = intent_mod.MemoryIntentKind
extract_memory_writes = extract_mod.extract_memory_writes
load_merged_memory = inject_mod.load_merged_memory
format_memory_context = inject_mod.format_memory_context
should_include_user_memory = inject_mod.should_include_user_memory
UserKind = identity_models.UserKind
IdentitySource = identity_models.IdentitySource
AgentUser = identity_models.AgentUser
ResolvedIdentity = identity_models.ResolvedIdentity


@pytest.fixture
def memory_store(tmp_path: Path) -> PersistentMemoryStore:
    store = PersistentMemoryStore(tmp_path / "memory.db")
    store.connect()
    yield store
    store.close()


def _identity(
    *,
    kind: UserKind = UserKind.REGISTERED,
    source: IdentitySource = IdentitySource.LOGIN,
    confidence: float | None = None,
) -> ResolvedIdentity:
    user = AgentUser(
        id="user-1",
        kind=kind,
        display_name="Alex",
        created_at=1.0,
        updated_at=1.0,
    )
    return ResolvedIdentity(
        user=user,
        source=source,
        speaker_confidence=confidence,
    )


def test_system_and_user_precedence(memory_store: PersistentMemoryStore) -> None:
    memory_store.set_system("news.digest_scope", "national", route_scope="news")
    memory_store.set_user(
        "user-1",
        "news.digest_scope",
        "local",
        route_scope="news",
    )
    merged = memory_store.merge_for_turn(
        agent_user_id="user-1",
        include_user=True,
        route="news",
    )
    assert merged.values["news.digest_scope"] == "local"
    assert merged.sources["news.digest_scope"] == MemoryScope.USER


def test_guest_skips_user_memory(memory_store: PersistentMemoryStore) -> None:
    memory_store.set_system("news.digest_scope", "local", route_scope="news")
    memory_store.set_user("user-1", "news.digest_scope", "national", route_scope="news")
    guest = _identity(kind=UserKind.GUEST, source=IdentitySource.ASSIST_GUEST)
    assert should_include_user_memory(guest) is False
    merged = load_merged_memory(
        memory_store,
        identity=guest,
        route="news",
    )
    assert merged.values["news.digest_scope"] == "local"


def test_remember_and_forget_intent() -> None:
    remember = detect_memory_intent("Remember that I prefer local news")
    assert remember.kind in {MemoryIntentKind.REMEMBER, MemoryIntentKind.PREFER}
    forget = detect_memory_intent("Forget my news preference")
    assert forget.kind == MemoryIntentKind.FORGET
    skill = detect_memory_intent("Remember this as a skill")
    assert skill.kind == MemoryIntentKind.NONE
    assert skill.is_workflow is True


def test_extract_local_news_and_alias() -> None:
    news = extract_memory_writes("I prefer local news briefings")
    assert any(
        item.key == "news.digest_scope" and item.value == "local" for item in news
    )
    alias = extract_memory_writes(
        "Remember dining light is light.dining_room_lights_ceiling"
    )
    assert any(
        item.key.startswith("entity.alias.")
        and item.value == "light.dining_room_lights_ceiling"
        for item in alias
    )


def test_extract_sensor_alias_and_this_entity_from_history() -> None:
    """Sensor aliases and 'this entity is for …' use the prior lookup id."""
    entity_ids_from_history = extract_mod.entity_ids_from_history
    explicit = extract_memory_writes(
        "Remember outdoor air quality is sensor.home_outdoor_aqi_5min_mean"
    )
    assert any(
        item.key == "entity.alias.outdoor_air_quality"
        and item.value == "sensor.home_outdoor_aqi_5min_mean"
        and item.route_scope is None
        for item in explicit
    )

    intent = detect_memory_intent("Remember this entity is for outdoor air quality")
    assert intent.kind == MemoryIntentKind.REMEMBER
    history = [
        {"role": "user", "content": "what is the outdoor air quality"},
        {
            "role": "assistant",
            "content": (
                "The outdoor air quality (AQI) is currently 63.71. "
                "Controlled: sensor.home_outdoor_aqi_5min_mean."
            ),
            "turn_meta": {
                "referenced_entity_ids": ["sensor.home_outdoor_aqi_5min_mean"]
            },
        },
    ]
    prior = entity_ids_from_history(history)
    assert prior == ["sensor.home_outdoor_aqi_5min_mean"]
    writes = extract_memory_writes(
        "Remember this entity is for outdoor air quality",
        fragment=intent.fragment,
        controlled_entity_ids=prior,
    )
    assert any(
        item.key == "entity.alias.outdoor_air_quality"
        and item.value == "sensor.home_outdoor_aqi_5min_mean"
        and item.route_scope is None
        for item in writes
    )


def test_reject_air_quality_alias_to_light() -> None:
    """Reading aliases must not latch onto unrelated control entities."""
    assert not extract_memory_writes(
        "Remember outdoor air quality is light.dining_room"
    )
    # Older light control must not win over a later AQI lookup.
    history = [
        {
            "role": "assistant",
            "content": "OK. Controlled: light.dining_room.",
            "turn_meta": {"controlled_entity_ids": ["light.dining_room"]},
        },
        {
            "role": "assistant",
            "content": "AQI is 63. Controlled: sensor.home_outdoor_aqi_5min_mean.",
            "turn_meta": {
                "referenced_entity_ids": ["sensor.home_outdoor_aqi_5min_mean"],
                "controlled_entity_ids": ["light.dining_room"],
            },
        },
    ]
    prior = extract_mod.entity_ids_from_history(history)
    assert prior[0] == "sensor.home_outdoor_aqi_5min_mean"
    writes = extract_memory_writes(
        "Remember this entity is for outdoor air quality",
        controlled_entity_ids=prior,
    )
    assert len(writes) == 1
    assert writes[0].value == "sensor.home_outdoor_aqi_5min_mean"
    # Only a light in history → refuse rather than store a bad alias.
    assert not extract_memory_writes(
        "Remember this entity is for outdoor air quality",
        controlled_entity_ids=["light.dining_room"],
    )


def test_format_memory_context_compact(memory_store: PersistentMemoryStore) -> None:
    memory_store.set_system("email.default_mailbox", "INBOX", route_scope="email")
    merged = load_merged_memory(
        memory_store,
        identity=_identity(),
        route="email",
    )
    text = format_memory_context(merged)
    assert "DURABLE MEMORY" in text
    assert "email.default_mailbox" in text


def test_delete_user_memory(memory_store: PersistentMemoryStore) -> None:
    memory_store.set_user("user-1", "email.default_mailbox", "Work")
    assert memory_store.delete_user("user-1", "email.default_mailbox") is True
    assert memory_store.get_user("user-1", "email.default_mailbox") is None


def test_normalize_key_case_and_spaces(tmp_path):
    """Keys are normalized once in get/set/delete."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
    if "ha_agent" not in sys.modules:
        pkg = types.ModuleType("ha_agent")
        pkg.__path__ = [str(COMPONENT)]
        sys.modules["ha_agent"] = pkg
    if "ha_agent.persistent_memory" not in sys.modules:
        pm = types.ModuleType("ha_agent.persistent_memory")
        pm.__path__ = [str(COMPONENT / "persistent_memory")]
        sys.modules["ha_agent.persistent_memory"] = pm
    if "homeassistant.core" not in sys.modules:
        ha = types.ModuleType("homeassistant.core")
        ha.HomeAssistant = object
        sys.modules["homeassistant.core"] = ha
    if "homeassistant.exceptions" not in sys.modules:
        hexc = types.ModuleType("homeassistant.exceptions")
        class HomeAssistantError(Exception):
            pass
        hexc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant.exceptions"] = hexc
    if "ha_agent.skills" not in sys.modules:
        sk = types.ModuleType("ha_agent.skills")
        sk.__path__ = [str(COMPONENT / "skills")]
        sys.modules["ha_agent.skills"] = sk
    if "ha_agent.const" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "ha_agent.const", COMPONENT / "const.py"
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.const"] = m
        spec.loader.exec_module(m)
    if "ha_agent.skills.store" not in sys.modules:
        stub = types.ModuleType("ha_agent.skills.store")
        stub.validate_entry_id = lambda x: x
        sys.modules["ha_agent.skills.store"] = stub
    for name in ("models", "store"):
        mod = f"ha_agent.persistent_memory.{name}"
        if mod not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                mod, COMPONENT / "persistent_memory" / f"{name}.py"
            )
            m = importlib.util.module_from_spec(spec)
            sys.modules[mod] = m
            spec.loader.exec_module(m)
    PersistentMemoryStore = sys.modules[
        "ha_agent.persistent_memory.store"
    ].PersistentMemoryStore
    store = PersistentMemoryStore(tmp_path / "pm.db")
    store.connect()
    store.set_system("My Key", "value")
    assert store.get_system("my_key") is not None
    assert store.get_system("My Key").value == "value"
    assert store.delete_system("MY KEY") is True
    store.close()

"""Corpus tests that pin skill selection across ~500 household phrasings.

The fixtures under ``tests/fixtures/skill_selection`` hold LLM-generated
paraphrases for the asks this agent sees around a house (readings, device
control, and soft-domain workflows like email/news). Every phrase runs through
the real selection code so a wording change in one domain cannot silently
hijack another one.

Regenerate or extend the corpus with ``scripts/skill_selection_bench.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "skill_selection"


def _load(name: str):
    """Import a component module without installing Home Assistant."""
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    relative = name.replace(".", "/")
    spec = importlib.util.spec_from_file_location(
        module_name, COMPONENT / f"{relative}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True, slots=True)
class Case:
    """One corpus phrase and the behavior it pins."""

    text: str
    group: str
    kind: str
    expect_skill: str | None
    expect_route: str
    expect_soft_domain: str | None
    expect_reading_kind: str | None
    marked: bool

    def __str__(self) -> str:
        return f"[{self.group}] {self.text!r}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_catalog_spec() -> list[dict]:
    return _read_json(CORPUS_DIR / "catalog.json")["skills"]


def _load_cases() -> list[Case]:
    cases: list[Case] = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        if path.name == "catalog.json":
            continue
        data = _read_json(path)
        unmarked = {text.lower() for text in data.get("unmarked_phrases", [])}
        for phrase in data["phrases"]:
            cases.append(
                Case(
                    text=phrase,
                    group=data["group"],
                    kind=data["kind"],
                    expect_skill=data["expect_skill"],
                    expect_route=data["expect_route"],
                    expect_soft_domain=data["expect_soft_domain"],
                    expect_reading_kind=data["expect_reading_kind"],
                    marked=phrase.lower() not in unmarked,
                )
            )
    return cases


CATALOG_SPEC = _load_catalog_spec()
CASES = _load_cases()
SOFT_DOMAIN_GROUPS = {"email", "news"}


def _report(failures: list[str], total: int) -> str:
    shown = "\n".join(f"  - {line}" for line in failures[:25])
    more = "" if len(failures) <= 25 else f"\n  … {len(failures) - 25} more"
    return f"{len(failures)}/{total} corpus phrases failed:\n{shown}{more}"


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> object:
    """Real FTS-backed skill store loaded with the corpus catalog."""
    store_mod = _load("skills.store")
    db = tmp_path_factory.mktemp("corpus") / "skills.db"
    skill_store = store_mod.SkillStore(db)
    skill_store.connect()
    for spec in CATALOG_SPEC:
        skill_store.insert_skill(
            title=spec["title"],
            description=spec["description"],
            triggers=spec["triggers"],
            body=spec["body"],
            tool_steps=spec["tool_steps"],
            slug=spec["slug"],
            route_scope=spec["route_scope"],
        )
    yield skill_store
    skill_store.close()


@pytest.fixture(scope="module")
def skills_by_slug(store) -> dict:
    """Catalog skills keyed by slug, as loaded back from the store."""
    return {skill.slug: skill for skill in store.list_enabled(limit=50)}


def test_corpus_shape() -> None:
    """The corpus stays large, unique, and consistent with the catalog."""
    slugs = {spec["slug"] for spec in CATALOG_SPEC}
    assert len(CASES) >= 450
    assert sum(1 for case in CASES if case.group == "temperature") >= 100

    texts = [case.text.lower() for case in CASES]
    duplicates = {text for text in texts if texts.count(text) > 1}
    assert not duplicates, f"duplicate phrases: {sorted(duplicates)}"

    for case in CASES:
        assert case.expect_skill is None or case.expect_skill in slugs, str(case)
        assert case.expect_route in {"chat", "action"}, str(case)


def test_soft_domain_hint_has_no_false_positives() -> None:
    """No phrase may infer a domain other than its own."""
    selection = _load("skills.selection")
    failures = [
        f"{case} → inferred {hint!r}, expected {case.expect_soft_domain!r}"
        for case in CASES
        if (hint := selection.infer_soft_domain_hint(case.text))
        != (case.expect_soft_domain if case.marked else None)
    ]
    assert not failures, _report(failures, len(CASES))


def test_soft_domain_groups_infer_their_domain() -> None:
    """Email/news asks carrying domain vocabulary resolve to that domain."""
    selection = _load("skills.selection")
    cases = [case for case in CASES if case.group in SOFT_DOMAIN_GROUPS and case.marked]
    assert len(cases) >= 60
    failures = [
        f"{case} → {selection.infer_soft_domain_hint(case.text)!r}"
        for case in cases
        if selection.infer_soft_domain_hint(case.text) != case.expect_soft_domain
    ]
    assert not failures, _report(failures, len(cases))


def test_small_talk_skips_the_classifier() -> None:
    """Greetings and small talk never reach skill selection at all."""
    context = _load("context")
    phrases = _read_json(CORPUS_DIR / "chitchat.json")["casual_phrases"]
    assert phrases
    failures = [
        f"[chitchat] {phrase!r} reaches the classifier"
        for phrase in phrases
        if not context.is_casual_chat_query(phrase)
    ]
    assert not failures, _report(failures, len(phrases))


def test_reading_kind_inference() -> None:
    """Reading asks resolve to their sensor kind; control asks resolve to none."""
    policy = _load("loop_policy")
    cases = [case for case in CASES if case.kind in {"reading", "control"}]
    failures = [
        f"{case} → {kind!r}, expected {case.expect_reading_kind!r}"
        for case in cases
        if (kind := policy._infer_reading_kind(case.text)) != case.expect_reading_kind
    ]
    assert not failures, _report(failures, len(cases))


# Words that must never become the ha_search query for a reading: they describe
# the measurement or the request, not the place or device being measured.
_NOT_A_PLACE = frozenset(
    {
        "temperature",
        "temp",
        "temps",
        "degree",
        "degrees",
        "celsius",
        "fahrenheit",
        "thermometer",
        "warm",
        "warmer",
        "warmest",
        "cold",
        "colder",
        "coldest",
        "hot",
        "hotter",
        "hottest",
        "chilly",
        "freezing",
        "cool",
        "cooler",
        "humidity",
        "humid",
        "damp",
        "moisture",
        "dry",
        "aqi",
        "quality",
        "particulate",
        "co2",
        "ppm",
        "pressure",
        "index",
        "level",
        "levels",
        "percentage",
        "percent",
        "tell",
        "show",
        "give",
        "know",
        "want",
        "idea",
        "say",
        "says",
        "report",
        "whats",
        "does",
        "did",
        "any",
        "anything",
        "moment",
        "right",
        "now",
        "all",
        "much",
        "many",
        "high",
        "low",
        "above",
        "below",
        "still",
        "too",
        "very",
        "get",
        "got",
        "like",
        "out",
        "and",
    }
)


def test_reading_place_tokens_stay_locations() -> None:
    """Search terms derived from a reading ask contain no measurement vocabulary."""
    policy = _load("loop_policy")
    cases = [
        case
        for case in CASES
        if case.kind == "reading" and case.expect_reading_kind is not None
    ]
    failures = []
    for case in cases:
        tokens = policy._goal_place_tokens(case.text, case.expect_reading_kind)
        bad = [token for token in tokens if token in _NOT_A_PLACE]
        if bad:
            failures.append(f"{case} → place tokens {tokens} include {bad}")
    assert not failures, _report(failures, len(cases))


def test_specialized_skill_needs_domain_support(skills_by_slug) -> None:
    """An email/news skill is rejected unless the ask uses that domain's words."""
    selection = _load("skills.selection")
    specialized = [
        spec["slug"] for spec in CATALOG_SPEC if spec["group"] in SOFT_DOMAIN_GROUPS
    ]
    failures = []
    for case in CASES:
        hint = selection.infer_soft_domain_hint(case.text)
        for slug in specialized:
            skill = skills_by_slug[slug]
            if case.group == skill.route_scope and case.marked:
                continue
            if selection.skill_matches_route(
                skill,
                case.expect_route,
                domain_hint=hint,
                user_text=case.text,
            ):
                failures.append(f"{case} kept {slug!r}")
    assert not failures, _report(failures, len(CASES))


def test_ha_skills_rejected_on_soft_domain_asks(skills_by_slug) -> None:
    """Reading/control skills never serve an email or news ask."""
    selection = _load("skills.selection")
    ha_slugs = [
        spec["slug"] for spec in CATALOG_SPEC if spec["group"] in {"reading", "control"}
    ]
    cases = [case for case in CASES if case.group in SOFT_DOMAIN_GROUPS and case.marked]
    failures = [
        f"{case} kept {slug!r}"
        for case in cases
        for slug in ha_slugs
        if selection.skill_matches_route(
            skills_by_slug[slug],
            case.expect_route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not failures, _report(failures, len(cases))


def test_state_changing_skills_rejected_for_questions(skills_by_slug) -> None:
    """A status question never keeps a workflow that would change device state.

    Only asks phrased as questions are covered. A verbless ask ("current aqi")
    carries no read/command signal, and treating it as a read would also reject
    verbless commands ("lights off downstairs"), so those stay eligible.
    """
    selection = _load("skills.selection")
    context = _load("context")
    changing = [
        slug
        for slug, skill in skills_by_slug.items()
        if selection.skill_changes_state(skill)
    ]
    assert changing, "corpus catalog has no state-changing skill to test"
    cases = [
        case
        for case in CASES
        if case.kind in {"reading", "soft_domain"}
        and context.is_state_question(case.text)
    ]
    assert len(cases) >= 150
    failures = [
        f"{case} kept {slug!r}"
        for case in cases
        for slug in changing
        if selection.skill_matches_route(
            skills_by_slug[slug],
            case.expect_route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not failures, _report(failures, len(cases))


def test_expected_skill_survives_the_filters(skills_by_slug) -> None:
    """The right skill is never dropped by the route/domain gates."""
    selection = _load("skills.selection")
    cases = [case for case in CASES if case.expect_skill and case.marked]
    failures = [
        f"{case} dropped {case.expect_skill!r}"
        for case in cases
        if not selection.skill_matches_route(
            skills_by_slug[case.expect_skill],
            case.expect_route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not failures, _report(failures, len(cases))


def _prepass_result(prepass, *, case: Case, slug: str, skill):
    keyword = SimpleNamespace(
        summary="corpus",
        domain_hint=None,
        route=prepass.TaskRoute.CHAT,
    )
    return prepass._parse_prepass_payload(
        {
            "route": case.expect_route,
            "domain_hint": "",
            "complexity": "single",
            "skill_slug": slug,
            "slot_bindings": {},
        },
        catalog_by_slug={slug: skill},
        keyword_decision=keyword,
        heuristic=prepass.Complexity.SINGLE,
        user_text=case.text,
    )


def test_prepass_keeps_the_expected_skill(skills_by_slug) -> None:
    """When the classifier agrees, prepass keeps the skill for every phrasing."""
    prepass = _load_prepass()
    cases = [case for case in CASES if case.expect_skill and case.marked]
    failures = []
    for case in cases:
        assert case.expect_skill is not None
        result = _prepass_result(
            prepass,
            case=case,
            slug=case.expect_skill,
            skill=skills_by_slug[case.expect_skill],
        )
        if result is None or result.skill_selection is None:
            failures.append(f"{case} lost {case.expect_skill!r}")
    assert not failures, _report(failures, len(cases))


def test_prepass_drops_a_cross_domain_pick(skills_by_slug) -> None:
    """A classifier pick from another domain is dropped for every phrasing."""
    prepass = _load_prepass()
    failures = []
    for case in CASES:
        for slug in _conflicting_slugs(case):
            result = _prepass_result(
                prepass,
                case=case,
                slug=slug,
                skill=skills_by_slug[slug],
            )
            if result is not None and result.skill_selection is not None:
                failures.append(f"{case} kept conflicting {slug!r}")
    assert not failures, _report(failures, len(CASES))


def _conflicting_slugs(case: Case) -> list[str]:
    """Slugs from a different domain than the case, which must never be kept."""
    if case.group in SOFT_DOMAIN_GROUPS:
        if not case.marked:
            return []
        return [
            spec["slug"]
            for spec in CATALOG_SPEC
            if spec["group"] in SOFT_DOMAIN_GROUPS and spec["route_scope"] != case.group
        ] + ["look-up-sensor-or-entity-status"]
    return [
        spec["slug"] for spec in CATALOG_SPEC if spec["group"] in SOFT_DOMAIN_GROUPS
    ]


def _load_prepass():
    """Load prepass with the extra HA stubs its imports need."""
    if "homeassistant.components" not in sys.modules:
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
    if "homeassistant.components.conversation" not in sys.modules:
        sys.modules["homeassistant.components.conversation"] = types.ModuleType(
            "homeassistant.components.conversation"
        )
    return _load("prepass")


def _hass_for(store) -> MagicMock:
    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    return hass


def _llm_returning(slugs: list[str]) -> MagicMock:
    llm_client = _load("llm_client")
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=llm_client.ChatResult(
            content=json.dumps({"skill_slugs": slugs}),
            tool_calls=[],
            assistant_message={},
        )
    )
    return llm


def _backend():
    config_helpers = _load("config_helpers")
    return config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )


# One representative phrasing per group for the end-to-end selection path.
_INTEGRATION_SAMPLE = (
    ("temperature", "how warm is it in the kitchen"),
    ("temperature", "what is the temperature in Jonathans room"),
    ("humidity", "how humid is it in the basement"),
    ("air_quality", "what is the outdoor air quality"),
    ("co2", "what is the co2 level in the office"),
    ("energy", "how much power is the dishwasher using"),
    ("device_status", "is the front door locked"),
    ("email", "do I have new emails?"),
    ("email", "did I get any mail today"),
    ("news", "give me the headlines"),
    ("lights", "turn on the dining room lights"),
    ("covers", "open the living room blinds"),
    ("locks", "lock the front door"),
    ("climate_set", "set the thermostat to 21 degrees"),
    ("media", "play music in the kitchen"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("group", "text"), _INTEGRATION_SAMPLE)
async def test_resolve_never_returns_a_cross_domain_skill(
    monkeypatch, store, group: str, text: str
) -> None:
    """Even when the classifier picks a foreign skill, selection refuses it."""
    selection = _load("skills.selection")
    case = next(item for item in CASES if item.text == text)
    conflicting = _conflicting_slugs(case)
    if not conflicting:
        pytest.skip("no cross-domain conflict for this phrasing")

    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    result = await selection.resolve_skills_for_turn(
        _hass_for(store),
        "entry",
        _llm_returning(conflicting),
        _backend(),
        text,
        route=case.expect_route,
    )
    assert not [skill for skill in result.skills if skill.slug in conflicting], (
        f"{case} selected {[skill.slug for skill in result.skills]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("group", "text"), _INTEGRATION_SAMPLE)
async def test_resolve_returns_the_expected_skill_or_nothing(
    monkeypatch, store, group: str, text: str
) -> None:
    """Selection returns the expected skill or none — never a different one.

    Chat turns whose wording misses keyword search deliberately skip the
    classifier, so recall for paraphrases is pinned by the prepass corpus test
    above (prepass hands the whole catalog to the classifier every turn).
    """
    selection = _load("skills.selection")
    case = next(item for item in CASES if item.text == text)
    assert case.expect_skill is not None

    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    result = await selection.resolve_skills_for_turn(
        _hass_for(store),
        "entry",
        _llm_returning([case.expect_skill]),
        _backend(),
        text,
        route=case.expect_route,
    )
    slugs = [skill.slug for skill in result.skills]
    assert slugs in ([], [case.expect_skill]), (
        f"{case} → {result.summary} ({result.detail})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "what is the temperature in Jonathans room",
        "how warm is it in the kitchen",
        "how humid is it in the basement",
    ],
)
async def test_resolve_pins_the_status_skill_for_reading_asks(
    monkeypatch, store, text: str
) -> None:
    """Common reading phrasings reach the status skill through selection."""
    selection = _load("skills.selection")
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    result = await selection.resolve_skills_for_turn(
        _hass_for(store),
        "entry",
        _llm_returning(["look-up-sensor-or-entity-status"]),
        _backend(),
        text,
        route="chat",
    )
    assert [skill.slug for skill in result.skills] == [
        "look-up-sensor-or-entity-status"
    ], f"{text!r} → {result.summary} ({result.detail})"

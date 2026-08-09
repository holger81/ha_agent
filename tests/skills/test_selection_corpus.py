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
        if "phrases" not in data:  # composite/oddity fixtures have their own shape
            continue
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


@dataclass(frozen=True, slots=True)
class Composite:
    """A connected, multi-step ask and the skills it may or may not use."""

    text: str
    route: str
    expect_soft_domain: str | None
    expect_reading_kind: str | None
    eligible: tuple[str, ...]
    forbidden: tuple[str, ...]
    known_dropped: tuple[str, ...]

    def __str__(self) -> str:
        return f"[composite] {self.text!r}"


def _load_composites() -> list[Composite]:
    data = _read_json(CORPUS_DIR / "composite.json")
    return [
        Composite(
            text=entry["text"],
            route=entry["route"],
            expect_soft_domain=entry["expect_soft_domain"],
            expect_reading_kind=entry["expect_reading_kind"],
            eligible=tuple(entry["eligible"]),
            forbidden=tuple(entry["forbidden"]),
            known_dropped=tuple(entry.get("known_dropped", ())),
        )
        for entry in data["cases"]
    ]


CATALOG_SPEC = _load_catalog_spec()
CASES = _load_cases()
COMPOSITES = _load_composites()
ODDITIES = _read_json(CORPUS_DIR / "oddities.json")
SOFT_DOMAIN_GROUPS = {"email", "news"}
HA_GROUPS = {"reading", "control", "composite"}


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
    ha_slugs = [spec["slug"] for spec in CATALOG_SPEC if spec["group"] in HA_GROUPS]
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


def test_composite_shape(skills_by_slug) -> None:
    """Composite cases stay unique and reference catalog skills."""
    assert len(COMPOSITES) >= 15
    texts = [case.text.lower() for case in COMPOSITES]
    assert len(set(texts)) == len(texts)
    for case in COMPOSITES:
        assert case.route in {"chat", "action"}, str(case)
        for slug in case.eligible + case.forbidden + case.known_dropped:
            assert slug in skills_by_slug, f"{case} references unknown {slug!r}"


def test_composite_infers_domain_and_reading_kind() -> None:
    """A connected ask still resolves its domain and any reading it implies."""
    selection = _load("skills.selection")
    policy = _load("loop_policy")
    failures = []
    for case in COMPOSITES:
        hint = selection.infer_soft_domain_hint(case.text)
        if hint != case.expect_soft_domain:
            want = case.expect_soft_domain
            failures.append(f"{case} → domain {hint!r}, want {want!r}")
        kind = policy._infer_reading_kind(case.text)
        if kind != case.expect_reading_kind:
            want_kind = case.expect_reading_kind
            failures.append(f"{case} → kind {kind!r}, want {want_kind!r}")
    assert not failures, _report(failures, len(COMPOSITES))


def test_composite_keeps_the_skills_the_turn_needs(skills_by_slug) -> None:
    """The gates must not drop a skill a multi-step ask genuinely needs.

    This is the inverse of the cross-domain tests: over-rejection is just as
    much a bug as picking a foreign skill, and compound asks are where a
    single-domain gate is most likely to overreach.
    """
    selection = _load("skills.selection")
    failures = [
        f"{case} dropped {slug!r}"
        for case in COMPOSITES
        for slug in case.eligible
        if not selection.skill_matches_route(
            skills_by_slug[slug],
            case.route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not failures, _report(failures, len(COMPOSITES))


def test_composite_rejects_foreign_domain_skills(skills_by_slug) -> None:
    """A connected ask still refuses skills from an unrelated domain."""
    selection = _load("skills.selection")
    failures = [
        f"{case} kept {slug!r}"
        for case in COMPOSITES
        for slug in case.forbidden
        if selection.skill_matches_route(
            skills_by_slug[slug],
            case.route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not failures, _report(failures, len(COMPOSITES))


def test_composite_documented_gate_limitations(skills_by_slug) -> None:
    """Record where a single-domain hint drops a skill a compound ask needs.

    "email me the temperature in the nursery" carries an email hint, so the HA
    lookup skill is dropped as cross-domain even though the turn must read a
    sensor. Keeping it would need a multi-intent signal in the gate; a plain
    "shares a word with the skill" rule is not enough, because unrelated asks
    share generic verbs like look/check/read. This test pins today's behavior so
    the day it changes, the change is deliberate — update the fixture then.
    """
    selection = _load("skills.selection")
    cases = [case for case in COMPOSITES if case.known_dropped]
    assert cases, "no documented limitation left; drop this test"
    unexpected = [
        f"{case} now keeps {slug!r} (good — move it to 'eligible')"
        for case in cases
        for slug in case.known_dropped
        if selection.skill_matches_route(
            skills_by_slug[slug],
            case.route,
            domain_hint=selection.infer_soft_domain_hint(case.text),
            user_text=case.text,
        )
    ]
    assert not unexpected, _report(unexpected, len(cases))


def test_multi_step_skill_counts_as_state_changing(skills_by_slug) -> None:
    """A read step plus a write step is a write: snapshot-then-notify changes state."""
    selection = _load("skills.selection")
    snapshot = skills_by_slug["send-camera-snapshot"]
    steps = [step["toolName"] for step in snapshot.tool_steps]
    assert any("camera" in name for name in steps)
    assert selection.skill_changes_state(snapshot) is True
    # So it stays out of a plain question, without a camera-specific rule.
    assert (
        selection.skill_matches_route(
            snapshot, "chat", user_text="is anyone at the front door"
        )
        is False
    )


def test_oddities_never_raise(skills_by_slug) -> None:
    """Messy and degenerate input must not break any selection helper."""
    selection = _load("skills.selection")
    policy = _load("loop_policy")
    context = _load("context")
    texts = list(ODDITIES["texts"]) + list(ODDITIES["degenerate_texts"])
    assert len(texts) >= 40
    for text in texts:
        context.is_state_question(text)
        context.is_casual_chat_query(text)
        hint = selection.infer_soft_domain_hint(text)
        assert hint is None or isinstance(hint, str)
        assert isinstance(selection.soft_domains_in_text(text), frozenset)
        kind = policy._infer_reading_kind(text)
        assert isinstance(policy._goal_place_tokens(text, kind), list)
        for skill in skills_by_slug.values():
            assert isinstance(
                selection.skill_matches_route(
                    skill, "chat", domain_hint=hint, user_text=text
                ),
                bool,
            )


def test_degenerate_input_infers_nothing() -> None:
    """Empty, whitespace, and punctuation-only asks carry no intent."""
    selection = _load("skills.selection")
    policy = _load("loop_policy")
    failures = []
    for text in ODDITIES["degenerate_texts"]:
        if selection.infer_soft_domain_hint(text) is not None:
            failures.append(f"{text!r} inferred a domain")
        if policy._infer_reading_kind(text) is not None:
            failures.append(f"{text!r} inferred a reading kind")
        if policy._goal_place_tokens(text, "temperature"):
            failures.append(f"{text!r} produced place tokens")
    assert not failures, _report(failures, len(ODDITIES["degenerate_texts"]))


def test_oddity_questions_refuse_control_skills(skills_by_slug) -> None:
    """Typos and shouting do not turn a question into a command."""
    selection = _load("skills.selection")
    context = _load("context")
    changing = [
        slug
        for slug, skill in skills_by_slug.items()
        if selection.skill_changes_state(skill)
    ]
    failures = []
    for text in ODDITIES["question_texts"]:
        if not context.is_state_question(text):
            failures.append(f"{text!r} is not read as a question")
            continue
        failures.extend(
            f"{text!r} kept {slug!r}"
            for slug in changing
            if selection.skill_matches_route(
                skills_by_slug[slug],
                "chat",
                domain_hint=selection.infer_soft_domain_hint(text),
                user_text=text,
            )
        )
    assert not failures, _report(failures, len(ODDITIES["question_texts"]))


def test_oddity_commands_keep_control_skills(skills_by_slug) -> None:
    """Shouting, typos, and other languages still reach a control workflow."""
    selection = _load("skills.selection")
    lights = skills_by_slug["control-lights"]
    covers = skills_by_slug["control-covers"]
    failures = [
        f"{text!r} dropped every control skill"
        for text in ODDITIES["command_texts"]
        if not (
            selection.skill_matches_route(lights, "action", user_text=text)
            or selection.skill_matches_route(covers, "action", user_text=text)
        )
    ]
    assert not failures, _report(failures, len(ODDITIES["command_texts"]))


def test_oddity_search_term_is_the_place() -> None:
    """Filler, shouting, rambling, and umlauts still yield the place to search.

    The first place token becomes the retry ``ha_search`` query, so a greeting
    or hedge winning that slot sends the agent looking for "hey".
    """
    policy = _load("loop_policy")
    expected = ODDITIES["expected_place_head"]
    assert len(expected) >= 8
    failures = []
    for text, place in expected.items():
        kind = policy._infer_reading_kind(text)
        tokens = policy._goal_place_tokens(text, kind)
        if tokens[:1] != [place]:
            failures.append(f"{text[:48]!r} → {tokens[:3]}, want {place!r} first")
    for text in ODDITIES["no_place_texts"]:
        kind = policy._infer_reading_kind(text)
        if policy._goal_place_tokens(text, kind):
            failures.append(f"{text[:48]!r} invented a place")
    for text, place in ODDITIES["place_in_tokens"].items():
        kind = policy._infer_reading_kind(text)
        if place not in policy._goal_place_tokens(text, kind):
            failures.append(f"{text[:48]!r} lost the place {place!r}")
    assert not failures, _report(failures, len(expected))


def test_non_ascii_place_stays_one_token() -> None:
    """A word like "küche" must not be chopped into a meaningless fragment."""
    policy = _load("loop_policy")
    tokens = policy._goal_place_tokens("wie warm ist es in der küche", "temperature")
    assert "küche" in tokens
    assert "che" not in tokens


def test_oddity_place_tokens_have_no_junk() -> None:
    """Filler, shouting, and punctuation never become the entity search term."""
    policy = _load("loop_policy")
    failures = []
    for text in ODDITIES["texts"]:
        kind = policy._infer_reading_kind(text)
        if kind is None:
            continue
        tokens = policy._goal_place_tokens(text, kind)
        bad = [token for token in tokens if token in _NOT_A_PLACE]
        if bad:
            failures.append(f"{text!r} → {tokens} include {bad}")
    assert not failures, _report(failures, len(ODDITIES["texts"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "send a picture from my front door camera to my iphone",
        "take a snapshot of the driveway camera and text it to me",
    ],
)
async def test_resolve_pins_the_composite_skill(monkeypatch, store, text: str) -> None:
    """A connected camera-to-phone ask reaches its multi-step skill."""
    selection = _load("skills.selection")
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    result = await selection.resolve_skills_for_turn(
        _hass_for(store),
        "entry",
        _llm_returning(["send-camera-snapshot"]),
        _backend(),
        text,
        route="action",
    )
    assert [skill.slug for skill in result.skills] == ["send-camera-snapshot"], (
        f"{text!r} → {result.summary} ({result.detail})"
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

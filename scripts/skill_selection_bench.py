#!/usr/bin/env python3
"""Generate and benchmark the skill-selection phrase corpus.

The corpus lives in ``tests/fixtures/skill_selection`` and is asserted by
``tests/skills/test_selection_corpus.py``. This script has two jobs:

``generate``
    Ask an LLM for fresh paraphrases of one corpus group and append the new,
    deduplicated phrases to that group's fixture.

``bench``
    Score the corpus. Offline by default (marker inference plus the route and
    domain gates, exactly what CI asserts). With ``--llm`` it also asks a live
    model to pick a skill per phrase and reports how often the classifier plus
    the gates land on the expected skill.

    The ``--llm`` pass hands the whole catalog to the model for every phrase,
    like prepass does. That is the worst case: on a chat turn without a keyword
    hit, ``resolve_skills_for_turn`` never calls the classifier at all, so a
    general-knowledge question ("what is the capital of france") cannot pick up
    a skill there even if the model would have proposed one here.

Examples:
    scripts/skill_selection_bench.py bench
    scripts/skill_selection_bench.py bench --llm --group temperature
    scripts/skill_selection_bench.py generate --group humidity --count 30
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "ha_agent"
CORPUS_DIR = ROOT / "tests" / "fixtures" / "skill_selection"


def _stub_homeassistant() -> None:
    """Install the minimal Home Assistant modules the component imports."""
    if "homeassistant.core" in sys.modules:
        return

    package = types.ModuleType("homeassistant")
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["homeassistant"] = package

    core = types.ModuleType("homeassistant.core")
    core.HomeAssistant = type("HomeAssistant", (), {})
    core.callback = lambda func: func
    sys.modules["homeassistant.core"] = core

    exceptions = types.ModuleType("homeassistant.exceptions")
    exceptions.HomeAssistantError = type("HomeAssistantError", (Exception,), {})
    sys.modules["homeassistant.exceptions"] = exceptions

    for name in (
        "homeassistant.helpers",
        "homeassistant.components",
        "homeassistant.components.conversation",
    ):
        sys.modules[name] = types.ModuleType(name)


def _load(name: str):
    """Import a component module without installing Home Assistant."""
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _stub_homeassistant()
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    path = COMPONENT / f"{name.replace('.', '/')}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _group_files() -> list[Path]:
    return [
        path
        for path in sorted(CORPUS_DIR.glob("*.json"))
        if path.name != "catalog.json"
    ]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_path(group: str) -> Path:
    for path in _group_files():
        if _read(path)["group"] == group:
            return path
    raise SystemExit(f"unknown group {group!r}; have: {', '.join(_group_names())}")


def _group_names() -> list[str]:
    return [_read(path)["group"] for path in _group_files()]


def _catalog_skills() -> list[Any]:
    """Build Skill objects for the corpus catalog."""
    models = _load("skills.models")
    specs = _read(CORPUS_DIR / "catalog.json")["skills"]
    return [
        models.Skill(
            id=str(index),
            slug=spec["slug"],
            title=spec["title"],
            description=spec["description"],
            triggers=spec["triggers"],
            body=spec["body"],
            tool_steps=spec["tool_steps"],
            route_scope=spec["route_scope"],
        )
        for index, spec in enumerate(specs)
    ]


def _backend(args: argparse.Namespace):
    config_helpers = _load("config_helpers")
    return config_helpers.LlmBackend(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        thinking_level="off",
    )


_GENERATE_PROMPT = (
    "You write test data for a smart-home voice assistant.\n"
    'Return ONLY a JSON object: {"phrases": ["..."]}.\n'
    "Each phrase is one natural thing a person would say to their home "
    "assistant for the given intent. Vary sentence shape, politeness, room "
    "names, articles, and word order. Use lowercase, no trailing punctuation "
    "except a question mark, and no quotes inside phrases. Never repeat an "
    "existing phrase or restate one with only a different room name."
)


async def _generate(args: argparse.Namespace) -> int:
    path = _group_path(args.group)
    data = _read(path)
    existing = list(data["phrases"])
    llm_client = _load("llm_client")

    user = json.dumps(
        {
            "intent": data["group"],
            "intent_kind": data["kind"],
            "example_phrases": existing[:15],
            "count": args.count,
            "avoid": existing,
        }
    )
    async with aiohttp.ClientSession() as session:
        llm = llm_client.LlmClient(session)
        result = await llm.chat(
            [
                {"role": "system", "content": _GENERATE_PROMPT},
                {"role": "user", "content": user},
            ],
            _backend(args),
        )

    try:
        payload = json.loads((result.content or "").strip().strip("`"))
        phrases = [str(item).strip() for item in payload["phrases"]]
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        print(f"could not parse model output: {err}\n{result.content!r}")
        return 1

    seen = {phrase.lower() for phrase in existing}
    added = []
    for phrase in phrases:
        if phrase and phrase.lower() not in seen:
            seen.add(phrase.lower())
            added.append(phrase)

    if not added:
        print("no new phrases")
        return 0
    data["phrases"] = existing + added
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"added {len(added)} phrase(s) to {path.name}:")
    for phrase in added:
        print(f"  + {phrase}")
    print("\nRun the corpus tests to check the new phrases:")
    print("  pytest tests/skills/test_selection_corpus.py -q")
    return 0


def _offline_bench(groups: list[str]) -> int:
    """Score marker inference and the route/domain gates over the corpus."""
    selection = _load("skills.selection")
    policy = _load("loop_policy")
    skills = {skill.slug: skill for skill in _catalog_skills()}

    print(f"{'group':<15}{'phrases':>8}{'domain':>9}{'kind':>8}{'gate':>7}")
    print("-" * 47)
    failures: list[str] = []
    totals = [0, 0, 0, 0]
    for path in _group_files():
        data = _read(path)
        if groups and data["group"] not in groups:
            continue
        unmarked = {text.lower() for text in data.get("unmarked_phrases", [])}
        counts = [0, 0, 0]
        phrases = data["phrases"]
        for phrase in phrases:
            marked = phrase.lower() not in unmarked
            hint = selection.infer_soft_domain_hint(phrase)
            expected_hint = data["expect_soft_domain"] if marked else None
            if hint == expected_hint:
                counts[0] += 1
            else:
                failures.append(f"[{data['group']}] {phrase!r} domain → {hint!r}")

            if data["kind"] in {"reading", "control"}:
                kind = policy._infer_reading_kind(phrase)
                if kind == data["expect_reading_kind"]:
                    counts[1] += 1
                else:
                    failures.append(f"[{data['group']}] {phrase!r} kind → {kind!r}")
            else:
                counts[1] += 1

            expected = data["expect_skill"]
            if expected is None:
                counts[2] += 1
                continue
            kept = selection.skill_matches_route(
                skills[expected],
                data["expect_route"],
                domain_hint=hint,
                user_text=phrase,
            )
            # Phrases listed as unmarked carry no domain vocabulary, so the
            # gates are expected to drop their skill (a documented blind spot).
            if kept is marked:
                counts[2] += 1
            elif marked:
                failures.append(f"[{data['group']}] {phrase!r} dropped {expected!r}")
            else:
                failures.append(
                    f"[{data['group']}] {phrase!r} kept {expected!r} "
                    "but is listed as unmarked"
                )

        total = len(phrases)
        totals[0] += total
        for index, value in enumerate(counts):
            totals[index + 1] += value
        print(
            f"{data['group']:<15}{total:>8}"
            f"{counts[0] / total:>8.0%}{counts[1] / total:>8.0%}"
            f"{counts[2] / total:>7.0%}"
        )

    print("-" * 47)
    print(
        f"{'total':<15}{totals[0]:>8}"
        f"{totals[1] / totals[0]:>8.0%}{totals[2] / totals[0]:>8.0%}"
        f"{totals[3] / totals[0]:>7.0%}"
    )
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for line in failures[:40]:
            print(f"  - {line}")
    return 1 if failures else 0


async def _llm_bench(args: argparse.Namespace) -> int:
    """Ask a live model to pick a skill per phrase and score the outcome."""
    selection = _load("skills.selection")
    llm_client = _load("llm_client")
    catalog = _catalog_skills()
    backend = _backend(args)

    print(f"model={backend.model} base_url={backend.base_url}")
    print(f"{'group':<15}{'phrases':>8}{'correct':>9}{'wrong':>7}{'none':>7}")
    print("-" * 46)
    wrong_examples: list[str] = []
    async with aiohttp.ClientSession() as session:
        llm = llm_client.LlmClient(session)
        for path in _group_files():
            data = _read(path)
            if args.group and data["group"] not in args.group:
                continue
            phrases = data["phrases"][: args.limit] if args.limit else data["phrases"]
            correct = wrong = none = 0
            for phrase in phrases:
                # Mirror production: greetings never reach the classifier.
                if selection.is_casual_chat_query(phrase):
                    if data["expect_skill"] is None:
                        correct += 1
                    else:
                        none += 1
                    continue
                hint = selection.infer_soft_domain_hint(phrase)
                selected, _raw = await selection.select_skills_with_llm(
                    llm,
                    backend,
                    user_text=phrase,
                    route=data["expect_route"],
                    catalog=catalog,
                    max_select=1,
                    domain_hint=hint,
                )
                kept = [
                    skill
                    for skill in selected
                    if selection.skill_matches_route(
                        skill,
                        data["expect_route"],
                        domain_hint=hint,
                        user_text=phrase,
                    )
                ]
                slug = kept[0].slug if kept else None
                if slug == data["expect_skill"]:
                    correct += 1
                elif slug is None:
                    none += 1
                else:
                    wrong += 1
                    wrong_examples.append(
                        f"[{data['group']}] {phrase!r} → {slug!r} "
                        f"(want {data['expect_skill']!r})"
                    )
            total = len(phrases) or 1
            print(
                f"{data['group']:<15}{len(phrases):>8}{correct / total:>8.0%}"
                f"{wrong / total:>7.0%}{none / total:>7.0%}"
            )

    if wrong_examples:
        print(f"\n{len(wrong_examples)} cross-skill pick(s):")
        for line in wrong_examples[:40]:
            print(f"  - {line}")
    return 0


def main() -> int:
    """Parse arguments and run the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def _llm_args(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--base-url",
            default=os.environ.get("HA_AGENT_LLM_URL", "http://192.168.10.31:9292/v1"),
        )
        target.add_argument(
            "--model", default=os.environ.get("HA_AGENT_LLM_MODEL", "local-model")
        )
        target.add_argument("--api-key", default=os.environ.get("HA_AGENT_LLM_API_KEY"))
        target.add_argument("--max-tokens", type=int, default=2048)
        target.add_argument("--temperature", type=float, default=0.7)
        target.add_argument("--timeout", type=int, default=180)

    gen = sub.add_parser("generate", help="append LLM-written phrases to a group")
    gen.add_argument("--group", required=True, choices=_group_names())
    gen.add_argument("--count", type=int, default=25)
    _llm_args(gen)

    bench = sub.add_parser("bench", help="score the corpus")
    bench.add_argument("--group", action="append", default=[])
    bench.add_argument("--llm", action="store_true", help="also bench a live model")
    bench.add_argument("--limit", type=int, default=0, help="phrases per group")
    _llm_args(bench)

    args = parser.parse_args()
    if args.command == "generate":
        return asyncio.run(_generate(args))
    status = _offline_bench(args.group)
    if args.llm:
        args.temperature = min(args.temperature, 0.1)
        status = asyncio.run(_llm_bench(args)) or status
    return status


if __name__ == "__main__":
    sys.exit(main())

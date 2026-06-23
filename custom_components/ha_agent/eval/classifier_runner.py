"""Playbook classifier benchmarks for the eval system."""

from __future__ import annotations

import time

from ..config_helpers import LlmBackend
from ..llm_client import LlmClient
from ..playbooks import Playbook, select_playbook_with_llm
from .models import EvalCase, EvalCaseScore

_CLASSIFIER_CATALOG: tuple[Playbook, ...] = (
    Playbook(
        route="email",
        title="Email",
        match_text="The user asks about email, mail, inbox, or unread messages.",
        body="EMAIL PLAYBOOK",
        is_default=True,
    ),
    Playbook(
        route="news",
        title="News",
        match_text="The user asks for news, headlines, or a briefing.",
        body="NEWS PLAYBOOK",
        is_default=True,
    ),
    Playbook(
        route="action",
        title="Device action",
        match_text=(
            "The user asks to control devices: lights, covers, locks, climate, "
            "or camera snapshots."
        ),
        body="DEVICE PLAYBOOK",
        is_default=True,
    ),
    Playbook(
        route="movie_night",
        title="Movie night",
        match_text=(
            "The user wants a cozy movie night: dim living-room lights, "
            "close blinds, or similar ambience."
        ),
        body="MOVIE NIGHT PLAYBOOK",
        is_default=False,
        is_builtin=False,
    ),
    Playbook(
        route="general",
        title="General",
        match_text="Fallback for general requests that still need tools or evidence.",
        body="GENERAL PLAYBOOK",
        is_default=True,
    ),
)


def classifier_catalog() -> list[Playbook]:
    """Return the fixed playbook catalog used for classifier eval."""
    return list(_CLASSIFIER_CATALOG)


async def run_classifier_case(
    llm: LlmClient,
    backend: LlmBackend,
    case: EvalCase,
) -> EvalCaseScore:
    """Benchmark playbook selection for one classifier case."""
    started = time.perf_counter()
    catalog = classifier_catalog()
    selected = await select_playbook_with_llm(
        llm,
        backend,
        user_text=case.user_text,
        history=None,
        catalog=catalog,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    expected = case.expected_playbook_route
    selected_route = selected.route if selected else None
    details: list[str] = []
    if expected:
        if selected_route == expected:
            route_match = True
        else:
            route_match = False
            details.append(
                f"expected playbook route {expected!r}, got {selected_route!r}"
            )
    else:
        route_match = selected is not None

    passed = route_match
    score = 1.0 if passed else 0.0
    return EvalCaseScore(
        case_id=case.id,
        task=case.task,
        model=backend.model,
        score=score,
        passed=passed,
        latency_ms=latency_ms,
        iterations=1,
        outcome="success" if passed else "failed",
        tool_match=route_match,
        text_match=True,
        details=details,
    )

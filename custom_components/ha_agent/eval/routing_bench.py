"""Route-classifier microbench (no full agent loop)."""

from __future__ import annotations

from ..config_helpers import LlmBackend, RouterConfig
from ..llm_client import LlmClient
from ..router import (
    RouteResolution,
    TaskRoute,
    align_route_to_skill,
    resolve_route_with_classifier,
)
from ..skills.selection import infer_soft_domain_hint
from .models import EvalCase


async def run_routing_case(
    llm: LlmClient,
    backend: LlmBackend,
    case: EvalCase,
    *,
    router_config: RouterConfig,
    route_keywords: dict[str, list[str]] | None = None,
    structured_output_enabled: bool = True,
) -> RouteResolution:
    """Mirror production post-classifier soft-hint fill (skills/prepass off).

    Scores the model under test as the route classifier. Soft domain hints are
    filled the same way ``agent.run_agent`` does when the classifier omits one.
    """
    history = list(case.history) or None
    resolution = await resolve_route_with_classifier(
        llm,
        backend,
        user_text=case.user_text,
        exposed_entities=list(case.exposed_entities),
        router_config=router_config,
        route_keywords=route_keywords,
        history=history,
        structured_output_enabled=structured_output_enabled,
    )
    domain_hint = resolution.domain_hint
    route = resolution.route
    if not domain_hint:
        inferred = infer_soft_domain_hint(case.user_text, history)
        if inferred:
            if route == TaskRoute.HA_ACTION:
                route = TaskRoute.CHAT
            domain_hint = inferred
    route, domain_hint, align_reason = align_route_to_skill(
        route,
        domain_hint=domain_hint,
    )
    summary = resolution.classifier_summary
    detail = resolution.classifier_detail
    if align_reason:
        summary = f"{resolution.method} → {route.value}"
        detail = f"{detail} {align_reason[0].upper() + align_reason[1:]}."
    return RouteResolution(
        route=route,
        method=resolution.method,
        classifier_summary=summary,
        classifier_detail=detail,
        keyword_hint=resolution.keyword_hint,
        classifier_raw=resolution.classifier_raw,
        domain_hint=domain_hint,
    )

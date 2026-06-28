"""Post-use skill evaluation and hourly-gated improvement."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from homeassistant.core import HomeAssistant

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from ..status import update_agent_status
from .models import Skill, SkillRunResult, TurnTrace
from .observer import is_discovery_tool
from .repair import detect_repairable_issues
from .store import get_skill_store

_EVAL_PROMPT = (
    "You evaluate how well a Home Assistant assistant skill performed.\n"
    "Return ONLY valid JSON with keys: succeeded, followed_steps, improve, "
    "title, description, triggers, body, tool_steps, slots.\n"
    "- succeeded: boolean whether the run completed without tool errors\n"
    "- followed_steps: boolean whether tool calls matched the skill tool_steps\n"
    "- improve: boolean whether the skill text should be updated\n"
    "- if improve is true, include revised title, description, triggers, "
    "body, tool_steps, slots\n"
    "When tool errors mention missing parameters, set improve=true and add "
    "required arguments to tool_steps.\n"
    "Keep entity_id values from the trace; do not invent new devices."
)


def build_run_result(
    skill_id: str,
    trace: TurnTrace,
    skill: Skill,
) -> SkillRunResult:
    """Score a turn that used a matched skill."""
    followed = _trace_follows_steps(trace.tool_calls, skill.tool_steps)
    discovery_violation = _used_discovery_with_concrete_skill(trace, skill)
    succeeded = (
        not trace.fallback
        and trace.tool_errors == 0
        and bool(trace.assistant_text.strip())
        and trace.verifier_verdict != "fail"
        and trace.skill_followed is not False
        and not discovery_violation
    )
    return SkillRunResult(
        skill_id=skill_id,
        iterations=trace.iterations,
        tool_errors=trace.tool_errors,
        followed_steps=followed and not discovery_violation,
        succeeded=succeeded,
    )


def _used_discovery_with_concrete_skill(trace: TurnTrace, skill: Skill) -> bool:
    concrete = [
        step
        for step in skill.tool_steps
        if str(step.get("toolName") or step.get("name") or "").strip()
        and not is_discovery_tool(str(step.get("toolName") or step.get("name") or ""))
    ]
    if len(concrete) < 2:
        return False
    return any(
        is_discovery_tool(str(call.get("toolName") or call.get("name") or ""))
        for call in trace.tool_calls
    )


def _trace_follows_steps(
    tool_calls: list[dict[str, Any]],
    tool_steps: list[dict[str, Any]],
) -> bool:
    """Return True when executed tools roughly match the skill steps."""
    if not tool_steps:
        return bool(tool_calls)
    if not tool_calls:
        return False
    non_discovery = [
        call
        for call in tool_calls
        if not is_discovery_tool(str(call.get("toolName") or call.get("name") or ""))
    ]
    for index, step in enumerate(tool_steps[: len(non_discovery)]):
        if is_discovery_tool(str(step.get("toolName") or step.get("name") or "")):
            continue
        call = non_discovery[index] if index < len(non_discovery) else None
        if call is None:
            return False
        step_name = step.get("toolName") or step.get("name")
        call_name = call.get("toolName") or call.get("name")
        if step_name and call_name and step_name != call_name:
            return False
    return True


async def evaluate_skill_use(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    skill: Skill,
    trace: TurnTrace,
) -> None:
    """Record usage and optionally improve a skill (hourly cooldown)."""
    if skill.is_builtin:
        return
    store = get_skill_store(hass, entry_id)
    run_result = build_run_result(skill.id, trace, skill)

    def _record() -> Skill | None:
        return store.record_use(skill.id, succeeded=run_result.succeeded)

    updated = await hass.async_add_executor_job(_record)
    if updated is None:
        return

    score_delta = 0.15 if run_result.succeeded else -0.2
    if trace.verifier_verdict == "pass":
        score_delta += 0.05
    elif trace.verifier_verdict == "fail":
        score_delta -= 0.15
    if not run_result.followed_steps:
        score_delta -= 0.25
    if trace.skill_followed is False:
        score_delta -= 0.2

    def _adjust_score() -> Skill | None:
        return store.adjust_score(skill.id, score_delta)

    await hass.async_add_executor_job(_adjust_score)

    repair_issues = detect_repairable_issues(trace, skill)
    bypass_cooldown = any(
        issue.kind in {"missing_param", "not_followed"} for issue in repair_issues
    )

    def _can_improve() -> bool:
        return store.can_improve(skill.id, bypass_cooldown=bypass_cooldown)

    if not await hass.async_add_executor_job(_can_improve):
        return

    force_improve = (
        not run_result.followed_steps
        or trace.skill_followed is False
        or trace.tool_errors > 0
    )
    improvement = await _request_improvement(
        llm,
        backend,
        skill=skill,
        trace=trace,
        force_improve=force_improve,
    )
    if improvement is None:
        return

    skill.title = improvement.get("title", skill.title)
    skill.description = improvement.get("description", skill.description)
    if (triggers := improvement.get("triggers")) and isinstance(triggers, list):
        skill.triggers = [str(item) for item in triggers]
    if body := improvement.get("body"):
        skill.body = str(body)
    if (tool_steps := improvement.get("tool_steps")) and isinstance(
        tool_steps, list
    ):
        skill.tool_steps = [item for item in tool_steps if isinstance(item, dict)]
    skill.version += 1
    skill.last_improved_at = time.time()
    skill.last_evaluation_at = time.time()

    def _save() -> Skill:
        return store.update_skill(skill)

    saved = await hass.async_add_executor_job(_save)
    update_agent_status(
        hass,
        entry_id,
        last_skill_improved=saved.title,
    )


async def _request_improvement(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    skill: Skill,
    trace: TurnTrace,
    force_improve: bool = False,
) -> dict[str, Any] | None:
    """Ask the LLM whether and how to improve a skill."""
    payload = {
        "skill": {
            "title": skill.title,
            "description": skill.description,
            "triggers": skill.triggers,
            "body": skill.body,
            "tool_steps": skill.tool_steps,
            "slots": [
                {
                    "name": s.name,
                    "description": s.description,
                    "default": s.default,
                }
                for s in skill.slots
            ],
            "version": skill.version,
        },
        "run": {
            "user_text": trace.user_text,
            "tool_calls": trace.tool_calls,
            "tool_errors": trace.tool_errors,
            "iterations": trace.iterations,
            "assistant_text": trace.assistant_text,
            "verifier_verdict": trace.verifier_verdict,
            "verifier_detail": trace.verifier_detail,
            "skill_followed": trace.skill_followed,
            "slot_bindings": trace.slot_bindings,
            "recovery_hints": trace.recovery_hints,
        },
        "force_improve": force_improve,
    }
    messages = [
        {"role": "system", "content": _EVAL_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill evaluation failed: %s", err)
        return None

    content = (result.content or "").strip()
    if not content:
        return None
    return _parse_eval_response(content, force_improve=force_improve)


def _parse_eval_response(
    content: str, *, force_improve: bool = False
) -> dict[str, Any] | None:
    """Parse evaluator JSON."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("improve") and not force_improve:
        return None
    if force_improve and not data.get("improve"):
        data["improve"] = True
    return data

"""Merged LLM observer for skill learning: gate + distillation in one pass."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from .body import normalize_skill_draft
from .models import SkillDraft, SkillSlot, TurnTrace

_DISCOVERY_TOOL = re.compile(
    r"(searchToolsForDomain|searchTool|tools/list|tools_list)",
    re.IGNORECASE,
)

_OBSERVER_PROMPT = (
    "You are an observer for a Home Assistant voice assistant. Decide whether a "
    "completed turn should become a reusable skill, and if so extract ONLY the "
    "durable workflow — not one-off facts, email bodies, news text, or discovery "
    "noise.\n"
    "Return ONLY valid JSON with keys:\n"
    "- learn (boolean)\n"
    "- reason (short string)\n"
    "When learn=true, also include:\n"
    "- title (max 64 chars)\n"
    "- description (third-person WHAT + WHEN, max 512 chars)\n"
    "- triggers (3-8 example user phrases)\n"
    "- body (markdown workflow — primary instructions; mention tools in backticks)\n"
    "- tool_steps (optional; omit or [] to derive tool names from body on save)\n"
    "Rules for learn=true:\n"
    "- Repeatable procedure the user may ask again (device control, email check "
    "workflow, etc.), not a single factual answer.\n"
    "- Prefer a clear markdown body; tool_steps are optional when body names "
    "tools in backticks (e.g. `mcp_news__news_curate`).\n"
    "- slots (optional): [{name, description, source, default}] for parameterized "
    "workflows with {{slot}} placeholders in body/tool_steps.\n"
    "- parent_id (optional): when forking a variant of an existing skill.\n"
    "- EXCLUDE discovery/searchToolsForDomain/searchTool from body unless core.\n"
    "- Do not copy email bodies, headlines, or assistant reply text into body.\n"
    "- Use entity_id values only from controlled_entity_ids or tool arguments.\n"
    "Rules for learn=false:\n"
    "- One-off Q&A, chit-chat, news/email summaries (unless manual_save_requested "
    "and a clear reusable procedure exists).\n"
    "- Single lookup with no durable workflow.\n"
    "- Failed or empty runs with nothing reusable.\n"
    "When subtask_results are present, distill a multi-step orchestration "
    "procedure across domains."
)


@dataclass(frozen=True, slots=True)
class SkillObserverResult:
    """Outcome of the skill-learning observer."""

    learn: bool
    reason: str
    draft: SkillDraft | None = None


def is_discovery_tool(tool_name: str) -> bool:
    """Return True for MCP discovery/list tools."""
    return bool(_DISCOVERY_TOOL.search(tool_name or ""))


def build_observer_payload(
    trace: TurnTrace,
    history: list[dict[str, str]],
    *,
    manual_save: bool = False,
) -> dict[str, Any]:
    """Build structured trace for the skill observer."""
    tools: list[dict[str, Any]] = []
    for call in trace.tool_calls:
        name = str(call.get("toolName") or call.get("name") or "").strip()
        if not name:
            continue
        tools.append(
            {
                "toolName": name,
                "arguments": call.get("arguments")
                if isinstance(call.get("arguments"), dict)
                else {},
                "succeeded": bool(call.get("succeeded", True)),
                "discovery": bool(
                    call.get("discovery") or is_discovery_tool(name)
                ),
                "error": call.get("error"),
                "error_kind": call.get("error_kind"),
                "missing_fields": call.get("missing_fields") or [],
            }
        )

    return {
        "user_goal": trace.user_text,
        "route": trace.route,
        "manual_save_requested": manual_save,
        "assistant_summary": (trace.assistant_text or "")[:1500],
        "history": history[-6:],
        "tools": tools,
        "tool_errors": trace.tool_errors,
        "iterations": trace.iterations,
        "outcome": trace.outcome,
        "controlled_entity_ids": trace.controlled_entity_ids,
        "matched_existing_skills": bool(trace.matched_skill_ids),
        "slot_bindings": trace.slot_bindings,
        "verifier_verdict": trace.verifier_verdict,
        "skill_followed": trace.skill_followed,
        "recovery_hints": trace.recovery_hints[-4:],
        "complexity": trace.complexity,
        "subtask_results": trace.subtask_results[:8],
        "orchestration_plan": trace.orchestration_plan[:8],
    }


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def parse_observer_response(content: str) -> SkillObserverResult | None:
    """Parse observer JSON. None when unusable."""
    try:
        data = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    learn = data.get("learn")
    if not isinstance(learn, bool):
        return None
    reason = str(data.get("reason", "")).strip() or (
        "Approved for learning." if learn else "Not worth learning."
    )

    if not learn:
        return SkillObserverResult(learn=False, reason=reason)

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    body = str(data.get("body", "")).strip()
    if not title or not description or not body:
        return SkillObserverResult(
            learn=True,
            reason=f"{reason} (distillation incomplete)",
            draft=None,
        )

    triggers_raw = data.get("triggers", [])
    tool_steps_raw = data.get("tool_steps", [])
    triggers = (
        [str(item).strip() for item in triggers_raw if str(item).strip()]
        if isinstance(triggers_raw, list)
        else []
    )
    tool_steps = (
        [
            item
            for item in tool_steps_raw
            if isinstance(item, dict)
            and str(item.get("toolName") or item.get("name") or "").strip()
        ]
        if isinstance(tool_steps_raw, list)
        else []
    )
    if not triggers:
        triggers = [title]

    slots_raw = data.get("slots", [])
    slots = []
    if isinstance(slots_raw, list):
        for item in slots_raw:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            slots.append(
                SkillSlot(
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                    source=str(item.get("source", "user")),
                    default=item.get("default"),
                )
            )

    draft = normalize_skill_draft(
        SkillDraft(
            title=title,
            description=description,
            triggers=triggers,
            body=body,
            tool_steps=tool_steps,
            slots=slots,
            parent_id=(
                str(data["parent_id"]).strip() if data.get("parent_id") else None
            ),
            route_scope=(
                str(data["route_scope"]).strip()
                if data.get("route_scope")
                else None
            ),
        ),
        explicit_tool_steps=bool(tool_steps),
    )

    return SkillObserverResult(
        learn=True,
        reason=reason,
        draft=draft,
    )


async def observe_skill_fork(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    parent_skill: Any,
    trace: TurnTrace,
    history: list[dict[str, str]],
) -> SkillObserverResult | None:
    """Propose a child skill when an existing skill was adapted with new slots."""
    from .models import Skill
    from .params import bindings_diverge_from_defaults

    if not isinstance(parent_skill, Skill):
        return None
    if not bindings_diverge_from_defaults(parent_skill, trace.slot_bindings):
        return None

    fork_prompt = (
        f"{_OBSERVER_PROMPT}\n"
        "FORK MODE: The user reused skill "
        f'"{parent_skill.title}" with different slot values. '
        "Propose a child variant (learn=true) with parent_id set to the parent "
        "skill id, updated triggers/body/tool_steps for the new intent, and "
        "slots reflecting the adapted parameters."
    )
    payload = build_observer_payload(trace, history)
    payload["parent_skill"] = {
        "id": parent_skill.id,
        "title": parent_skill.title,
        "body": parent_skill.body[:1200],
        "tool_steps": parent_skill.tool_steps,
    }
    payload["fork_slot_bindings"] = trace.slot_bindings
    messages = [
        {"role": "system", "content": fork_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill fork observer failed: %s", err)
        return None
    parsed = parse_observer_response(result.content or "")
    if parsed is None or not parsed.learn or parsed.draft is None:
        return None
    if not parsed.draft.parent_id:
        parsed.draft.parent_id = parent_skill.id
    return parsed


async def observe_skill_candidate(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    trace: TurnTrace,
    history: list[dict[str, str]],
    manual_save: bool = False,
) -> SkillObserverResult:
    """Run the merged observer (gate + distillation) on a turn trace."""
    payload = build_observer_payload(trace, history, manual_save=manual_save)
    messages = [
        {"role": "system", "content": _OBSERVER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill observer failed: %s", err)
        return SkillObserverResult(learn=False, reason="observer unavailable")

    content = (result.content or "").strip()
    if not content:
        return SkillObserverResult(learn=False, reason="empty observer response")

    parsed = parse_observer_response(content)
    if parsed is None:
        LOGGER.debug("Skill observer returned invalid JSON: %s", content[:200])
        return SkillObserverResult(learn=False, reason="invalid observer response")
    return parsed

"""LLM gate for whether a successful turn should become a learned skill."""

from __future__ import annotations

import json
import re
from typing import Any

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from .models import TurnTrace

_LEARN_GATE_PROMPT = (
    "You decide whether a Home Assistant assistant turn should be saved as a "
    "reusable skill.\n"
    "Return ONLY valid JSON with keys: learn (boolean), reason (short string).\n"
    "Say learn=true ONLY when ALL apply:\n"
    "- The user asked for a repeatable procedure or workflow, not a one-off fact.\n"
    "- The tool sequence is worth reusing for similar future requests.\n"
    "- Saving it would help handle triggers like this again without rediscovery.\n"
    "Say learn=false for one-off Q&A, news or email summaries, follow-up "
    "clarifications, chit-chat, single lookup answers, or generic information "
    "retrieval with no durable workflow."
)


def _parse_learn_gate_response(content: str) -> bool | None:
    """Parse learn-gate JSON. None when the response is unusable."""
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
    learn = data.get("learn")
    if not isinstance(learn, bool):
        return None
    return learn


async def assess_skill_worth_learning(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    trace: TurnTrace,
    history: list[dict[str, str]],
) -> bool:
    """Ask the LLM whether this turn merits skill creation."""
    payload: dict[str, Any] = {
        "user_goal": trace.user_text,
        "assistant_summary": trace.assistant_text,
        "history": history[-6:],
        "tool_calls": trace.tool_calls,
        "tool_errors": trace.tool_errors,
        "iterations": trace.iterations,
        "controlled_entity_ids": trace.controlled_entity_ids,
        "outcome": trace.outcome,
    }
    messages = [
        {"role": "system", "content": _LEARN_GATE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Skill learn gate failed: %s", err)
        return False

    content = (result.content or "").strip()
    if not content:
        return False

    learn = _parse_learn_gate_response(content)
    if learn is None:
        LOGGER.debug("Skill learn gate returned invalid JSON: %s", content[:200])
        return False
    return learn

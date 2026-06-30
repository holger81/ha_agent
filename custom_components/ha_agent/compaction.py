"""In-turn message compaction for bounded context on small models."""

from __future__ import annotations

import json
from typing import Any

_DISCOVERY_TOOLS = frozenset(
    {"searchtoolsfordomain", "searchtool", "tools/list", "tools_list"}
)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate from serialized message size."""
    try:
        blob = json.dumps(messages, ensure_ascii=True)
    except TypeError:
        blob = str(messages)
    return max(1, len(blob) // 4)


def compact_messages_if_needed(
    messages: list[dict[str, Any]],
    *,
    token_budget: int,
    keep_recent_tool_results: int = 2,
) -> bool:
    """Summarize older tool results when the turn exceeds the token budget.

    Returns True when compaction ran.
    """
    if token_budget <= 0 or estimate_message_tokens(messages) <= token_budget:
        return False

    tool_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    ]
    if len(tool_indexes) <= keep_recent_tool_results:
        return False

    compacted = False
    for index in tool_indexes[:-keep_recent_tool_results]:
        message = messages[index]
        content = str(message.get("content") or "")
        if content.startswith("[Earlier tool result summarized]"):
            continue
        preview = content.replace("\n", " ").strip()
        if len(preview) > 160:
            preview = f"{preview[:157]}..."
        messages[index] = {
            **message,
            "content": (
                "[Earlier tool result summarized] "
                f"{preview or 'tool output omitted to save context'}"
            ),
        }
        compacted = True
    return compacted

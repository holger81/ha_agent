"""Build LLM context from Home Assistant conversation input."""

from __future__ import annotations

import json
import re
from typing import Any

from homeassistant.components import conversation

_AFFIRMATIVE = re.compile(
    r"^(yes|yeah|yep|sure|please|ok|okay|go ahead|do it|try that)\.?$",
    re.IGNORECASE,
)
_NEWS_QUERY = re.compile(
    r"\b(news|headlines|briefing|nachrichten|headline)\b",
    re.IGNORECASE,
)
_DEVICE_ACTION = re.compile(
    r"\b(open|close|turn on|turn off|toggle|lock|unlock)\b",
    re.IGNORECASE,
)
_EMAIL_QUERY = re.compile(
    r"\b(emails?|e-mail|mail|inbox|unread)\b",
    re.IGNORECASE,
)


def parse_exposed_entities(raw: Any) -> list[dict[str, Any]]:
    """Parse exposed entities from webhook-style payloads or lists."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [entity for entity in raw if isinstance(entity, dict)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [entity for entity in parsed if isinstance(entity, dict)]
    return []


def format_exposed_entities(entities: list[dict[str, Any]]) -> str:
    """Format exposed entities for the system prompt."""
    lines: list[str] = []
    for entity in entities:
        entity_id = entity.get("entity_id")
        if not entity_id:
            continue
        name = entity.get("name") or entity_id
        state = entity.get("state")
        area = entity.get("area_name")
        suffix = ""
        if state is not None:
            suffix += f" state={state}"
        if area:
            suffix += f" area={area}"
        lines.append(f"- {entity_id} ({name}{suffix})")
    return "\n".join(lines)


def is_affirmative(query: str) -> bool:
    """Return True for short affirmative replies."""
    return bool(_AFFIRMATIVE.match(query.strip()))


def is_news_query(query: str) -> bool:
    """Return True when the user asks for news."""
    return bool(_NEWS_QUERY.search(query))


def is_device_action_query(query: str) -> bool:
    """Return True when the user asks for a device action."""
    return bool(_DEVICE_ACTION.search(query))


def is_email_query(query: str) -> bool:
    """Return True when the user asks about email."""
    return bool(_EMAIL_QUERY.search(query))


def entity_matches_query(entity: dict[str, Any], query: str) -> bool:
    """Return True when an exposed entity matches query tokens."""
    parts: list[str] = []
    for key in ("entity_id", "name", "area_name"):
        if value := entity.get(key):
            parts.append(str(value).lower())
    aliases = entity.get("aliases")
    if isinstance(aliases, list):
        parts.extend(str(alias).lower() for alias in aliases)

    tokens = [token for token in query.lower().split() if len(token) > 2]
    return any(
        token in part for token in tokens for part in parts if part
    )


def build_tool_context(query: str, exposed: list[dict[str, Any]]) -> str:
    """Build optional tool hints (not route classifiers)."""
    context_parts: list[str] = []

    if exposed:
        context_parts.append(
            "EXPOSED ENTITIES (use matching entity_id directly; "
            "do not search if one clearly fits):\n"
            + format_exposed_entities(exposed)
        )

    if is_affirmative(query) or is_news_query(query):
        context_parts.append(
            "NEWS: call mcp_news__news_curate with {} via mcp_call_tool once "
            "(no searchToolsForDomain, no searxng), then summarize headlines."
        )

    if is_email_query(query):
        context_parts.append(
            "EMAIL: use MCP mail tools via mcp_call_tool. Unread count: "
            '{"toolName":"imap_mailbox_status","arguments":{"mailbox":"INBOX"}}. '
            "Search messages: imap_search_messages with flat fields and required "
            'mailbox. Do not search HA entities for email.'
        )

    if is_device_action_query(query) and not any(
        entity_matches_query(entity, query) for entity in exposed
    ):
        context_parts.append(
            "DEVICE ACTION: no exposed entity clearly matches. Search once with "
            '{"toolName":"home_assistant__ha_search_entities","arguments":'
            '{"query":"<keywords from user>","domain_filter":"cover"}} for '
            "doors/covers, then ha_call_service. mcp_call_tool: top-level toolName "
            "+ flat arguments only — never \"value\", never nested toolName."
        )

    return "\n\n".join(context_parts)


def build_system_message(
    agent_system_prompt: str,
    tool_instructions: str,
    *,
    tool_context: str = "",
    extra_system_prompt: str | None = None,
) -> str:
    """Assemble the system message for the LLM."""
    parts = [agent_system_prompt.strip(), tool_instructions.strip()]
    if tool_context.strip():
        parts.append(tool_context.strip())
    if extra_system_prompt and extra_system_prompt.strip():
        parts.append(extra_system_prompt.strip())
    return "\n\n".join(part for part in parts if part)


def build_messages(
    *,
    system_message: str,
    history: list[dict[str, str]],
    user_text: str,
) -> list[dict[str, str]]:
    """Build OpenAI-style messages for the agent."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def user_text_from_input(user_input: conversation.ConversationInput) -> str:
    """Extract user text from conversation input."""
    if user_input.text:
        return user_input.text.strip()
    return ""

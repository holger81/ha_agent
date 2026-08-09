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
    r"\b("
    r"open|close|toggle|lock|unlock|"
    r"switch\s+(?:on|off)|"
    r"turn\s+(?:on|off)|"
    r"turn\b(?:\s+\w+){0,6}\s+(?:on|off)"
    r")\b",
    re.IGNORECASE,
)
_CAMERA_ACTION = re.compile(
    r"\b("
    r"snapshot|"
    r"take\s+(?:a\s+)?(?:photo|picture|pic|snapshot)|"
    r"capture\s+(?:an?\s+)?(?:image|photo|picture|snapshot)"
    r")\b|"
    r"\b(?:snap|take)\b.{0,40}\bcam(?:era)?\b",
    re.IGNORECASE,
)
_FOLLOW_UP_REF = re.compile(
    r"\b(them|those|these|it|that|again|back)\b",
    re.IGNORECASE,
)
_ENTITY_ID = re.compile(
    r"\b(?:light|switch|cover|fan|lock|climate|media_player|camera)\.[a-z0-9_]+\b",
    re.IGNORECASE,
)
_EMAIL_QUERY = re.compile(
    r"\b(emails?|e-mail|mail|inbox|unread)\b",
    re.IGNORECASE,
)
_CAPABILITY_QUERY = re.compile(
    r"\b(what tools?|which tools?|what can you|what do you have access|capabilities)\b",
    re.IGNORECASE,
)
_EXPOSED_ENTITIES_HEADER = (
    "EXPOSED ENTITIES (Assist shortcuts — incomplete inventory):\n"
    "This list is NOT complete and must NOT be treated as all entities in "
    "the home. Prefer a matching shortcut when it clearly fits; otherwise "
    "look up the entity with MCP tools (searchToolsForDomain / searchTool, "
    "then callTool — e.g. ha_search with domain_filter). Never invent tool "
    "names. Never tell the user an entity or room is missing solely because "
    "it is absent from this list — search first."
)
_DEVICE_DISCOVERY_FALLBACK = (
    "Discover tools with searchToolsForDomain / searchTool for the relevant "
    "MCP domain, then callTool using an exact toolName and arguments from "
    "that tool's schema. Never invent tool names."
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


def _keyword_regex(keywords: list[str] | None) -> re.Pattern[str] | None:
    """Build a case-insensitive whole-word regex from editable keywords.

    Returns ``None`` when no usable keywords are supplied so callers fall back
    to the shipped default regex.
    """
    if not keywords:
        return None
    parts = [re.escape(keyword.strip()) for keyword in keywords if keyword.strip()]
    if not parts:
        return None
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


def route_keyword_match(
    query: str,
    route_name: str,
    keywords: list[str] | None = None,
) -> str | None:
    """Return a short label when an action keyword matches, else None.

    Only the ``action`` route uses keyword matching. Soft domains (email,
    news, …) are selected via skills, not parallel keyword routes.
    """
    if route_name != "action":
        return None
    if override := _keyword_regex(keywords):
        pattern = override
    elif _CAMERA_ACTION.search(query):
        pattern = _CAMERA_ACTION
    elif _DEVICE_ACTION.search(query):
        pattern = _DEVICE_ACTION
    else:
        return None

    match = pattern.search(query)
    if not match:
        return None
    label = match.group(0).strip()
    if keywords:
        return label
    return f"{route_name}: {label}"


def is_affirmative(query: str) -> bool:
    """Return True for short affirmative replies."""
    return bool(_AFFIRMATIVE.match(query.strip()))


_GENERIC_CHITCHAT = re.compile(
    r"^(?:"
    r"hi|hello|hey|yo|howdy|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|thx|"
    r"ok|okay|bye|goodbye|see\s+ya"
    r")[!.?\s]*$",
    re.IGNORECASE,
)

_CASUAL_CHAT = re.compile(
    r"\b(?:"
    r"joke|jokes|funny|make\s+me\s+laugh|"
    r"say\s+something\s+(?:funny|random)|"
    r"tell\s+me\s+(?:a\s+)?(?:joke|story|riddle)|"
    r"who\s+are\s+you|what\s+can\s+you\s+do"
    r")\b",
    re.IGNORECASE,
)

_CHAT_ROUTES = frozenset({"chat", "general", ""})


def is_generic_chitchat(query: str) -> bool:
    """Return True for greetings and other non-task small talk."""
    return bool(_GENERIC_CHITCHAT.match(query.strip()))


def is_casual_chat_query(query: str) -> bool:
    """Return True when the user wants conversation, not a saved workflow."""
    text = query.strip()
    if not text:
        return False
    if is_generic_chitchat(text):
        return True
    return bool(_CASUAL_CHAT.search(text))


def is_chat_route(route: str | None) -> bool:
    """Return True for general conversation routes (not action)."""
    return (route or "").lower() in _CHAT_ROUTES


def is_news_query(query: str, keywords: list[str] | None = None) -> bool:
    """Return True when the user asks for news.

    When ``keywords`` is supplied (a UI override), a whole-word regex built
    from them replaces the shipped default matcher.
    """
    pattern = _keyword_regex(keywords) or _NEWS_QUERY
    return bool(pattern.search(query))


def is_device_action_query(query: str, keywords: list[str] | None = None) -> bool:
    """Return True when the user asks for a homeassistant service action.

    When ``keywords`` is supplied (a UI override), a single whole-word regex
    built from them replaces the shipped device + camera matchers.
    """
    if override := _keyword_regex(keywords):
        return bool(override.search(query))
    return bool(_DEVICE_ACTION.search(query) or _CAMERA_ACTION.search(query))


def is_camera_action_query(query: str) -> bool:
    """Return True when the user asks for a camera snapshot or photo."""
    return bool(_CAMERA_ACTION.search(query))


def is_email_query(query: str, keywords: list[str] | None = None) -> bool:
    """Return True when the user asks about email.

    When ``keywords`` is supplied (a UI override), a whole-word regex built
    from them replaces the shipped default matcher.
    """
    pattern = _keyword_regex(keywords) or _EMAIL_QUERY
    return bool(pattern.search(query))


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
    return any(token in part for token in tokens for part in parts if part)


def _format_entity_candidates(entities: list[dict[str, Any]]) -> list[str]:
    """Return prompt lines for candidate exposed entities (no tool names)."""
    lines: list[str] = []
    for entity in entities:
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id.strip():
            continue
        name = entity.get("name") or entity_id
        area = entity.get("area_name")
        detail = f" ({name}"
        if area:
            detail += f", area={area}"
        detail += ")"
        lines.append(f"- entity_id {entity_id}{detail}")
    return lines


def _entity_discovery_hint(_query: str) -> str:
    """Return discovery guidance when no exposed-entity shortcut matches."""
    return _DEVICE_DISCOVERY_FALLBACK


def _history_entity_ids(history: list[dict[str, str]]) -> list[str]:
    """Return entity ids mentioned in prior conversation turns."""
    combined = " ".join(message.get("content", "") for message in history[-6:])
    return _entity_ids_from_text(combined)


def _device_action_hint(
    query: str,
    exposed: list[dict[str, Any]],
    *,
    history: list[dict[str, str]] | None = None,
) -> str | None:
    """Return generic device-action guidance (no hardcoded upstream tools)."""
    if not is_device_action_query(query):
        return None

    prior_turns = history or []
    matches = [entity for entity in exposed if entity_matches_query(entity, query)]
    if is_camera_action_query(query):
        camera_matches = [
            entity
            for entity in matches
            if str(entity.get("entity_id", "")).startswith("camera.")
        ]
        if camera_matches:
            matches = camera_matches

    if matches:
        lines = [
            "DEVICE ACTION: matching exposed-entity shortcut(s) below. Prefer "
            "them with the appropriate MCP tool from discovery / session tools. "
            "Shortcuts are incomplete — if they are wrong or the target is not "
            "listed, search via MCP instead of assuming it does not exist. "
            "Never invent tool names.",
            *_format_entity_candidates(matches),
        ]
        return "\n".join(lines)

    if history_ids := _history_entity_ids(prior_turns):
        lines = [
            "DEVICE ACTION: reuse entity_id values from the prior turn in this "
            "conversation with the appropriate MCP tool. Never invent tool names.",
        ]
        for entity_id in history_ids:
            lines.append(f"- entity_id {entity_id}")
        return "\n".join(lines)

    return (
        "DEVICE ACTION: no exposed-entity shortcut clearly matches. "
        f"{_entity_discovery_hint(query)}"
    )


def _entity_ids_from_text(text: str) -> list[str]:
    """Return homeassistant entity ids mentioned in text."""
    return list(dict.fromkeys(match.group(0) for match in _ENTITY_ID.finditer(text)))


def _recent_device_context(history: list[dict[str, str]]) -> bool:
    """Return True when recent turns mention device actions or entity ids."""
    combined = " ".join(message.get("content", "") for message in history[-6:])
    return bool(
        _DEVICE_ACTION.search(combined)
        or _CAMERA_ACTION.search(combined)
        or _entity_ids_from_text(combined)
    )


def _recent_news_context(history: list[dict[str, str]]) -> bool:
    """Return True when recent turns were about news."""
    combined = " ".join(message.get("content", "") for message in history[-4:])
    return bool(is_news_query(combined))


def _recent_email_context(history: list[dict[str, str]]) -> bool:
    """Return True when recent turns were about email."""
    combined = " ".join(message.get("content", "") for message in history[-4:])
    return bool(is_email_query(combined))


_INFORMATIONAL_FOLLOW_UP = re.compile(
    r"\b("
    r"about|more|detail|details|tell me|explain|what happened|who|why|where|"
    r"this|these|that|those|it|them|again"
    r")\b",
    re.IGNORECASE,
)


def is_informational_follow_up(query: str) -> bool:
    """Return True when the user asks for more detail on a prior topic."""
    return bool(_INFORMATIONAL_FOLLOW_UP.search(query))


def is_short_follow_up_query(query: str) -> bool:
    """Return True for short retry/follow-up phrases that depend on history.

    Examples: "check again", "again", "what about that". These must not pin a
    skill from a shared verb alone (e.g. email "check …" matching "check again").
    """
    text = (query or "").strip()
    if not text:
        return False
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) > 5:
        return False
    return bool(_FOLLOW_UP_REF.search(text) or _INFORMATIONAL_FOLLOW_UP.search(text))


def resolve_turn_goal(
    user_text: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Return the substantive goal for this turn (prior user ask on short follow-ups).

    Short retries like "try again" must keep the previous reading/control goal so
    loop policy (reading kind, place tokens) and skill slots stay on-topic.
    """
    text = (user_text or "").strip()
    if not text:
        return ""
    if not history or not is_short_follow_up_query(text):
        return text
    for message in reversed(history):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        prior = str(message.get("content") or "").strip()
        if prior and not is_short_follow_up_query(prior):
            return prior
    return text


def _follow_up_device_hint(
    query: str,
    history: list[dict[str, str]],
) -> str | None:
    """Guide pronoun/retry follow-ups that rely on conversation memory."""
    if not history or not _FOLLOW_UP_REF.search(query):
        return None
    if not _recent_device_context(history):
        return None

    lines = [
        "FOLLOW-UP DEVICE ACTION: the user refers to an entity from earlier in "
        "this conversation. Reuse the same entity_id from the prior successful "
        "device command with the appropriate MCP tool. Never invent tool names "
        "or pass display names as entity_id.",
    ]
    history_text = " ".join(message.get("content", "") for message in history[-6:])
    if entity_ids := _entity_ids_from_text(history_text):
        lines.append(
            "Recent entity_id values from this conversation: " + ", ".join(entity_ids)
        )
    return "\n".join(lines)


def build_tool_context(
    query: str,
    exposed: list[dict[str, Any]],
    *,
    history: list[dict[str, str]] | None = None,
    skill_hints: str = "",
    route: str | None = None,
) -> str:
    """Build optional tool hints (not route classifiers)."""
    context_parts: list[str] = []
    prior_turns = history or []

    if skill_hints.strip():
        context_parts.append(skill_hints.strip())
        context_parts.append(
            "When ACTIVE SKILLS include a workflow for this request, follow that "
            "workflow first when it fits the user's goal. Use tool_steps only "
            "when present; otherwise follow the markdown workflow text. If the "
            "skill does not fit, declare SKILL_OVERRIDE: <reason> in your "
            "reasoning before discovery or other off-skill tools."
        )

    if exposed:
        context_parts.append(
            _EXPOSED_ENTITIES_HEADER + "\n" + format_exposed_entities(exposed)
        )

    if device_hint := _device_action_hint(query, exposed, history=prior_turns):
        context_parts.append(device_hint)

    if follow_up_hint := _follow_up_device_hint(query, prior_turns):
        context_parts.append(follow_up_hint)

    if is_email_query(query) and not skill_hints.strip():
        context_parts.append(
            "EMAIL: follow MCP SERVER INSTRUCTIONS. Discover tools in the email "
            "domain (searchToolsForDomain / searchTool), then callTool with an "
            "exact toolName from discovery. Never invent tool names."
        )

    if (
        is_news_query(query)
        or (is_affirmative(query) and _recent_news_context(prior_turns))
    ) and not skill_hints.strip():
        context_parts.append(
            "NEWS: follow MCP SERVER INSTRUCTIONS. Discover tools in the news "
            "domain (searchToolsForDomain / searchTool), then callTool with an "
            "exact toolName from discovery. Never invent tool names."
        )

    if _CAPABILITY_QUERY.search(query):
        context_parts.append(
            "CAPABILITIES: explain using MCP SERVER INSTRUCTIONS and MCP SESSION "
            "TOOLS. Describe discovery domains available in those instructions."
        )

    return "\n\n".join(context_parts)


def format_identity_context(identity: object) -> str:
    """Format resolved identity for LLM context."""
    from .identity.models import IdentitySource, ResolvedIdentity

    if not isinstance(identity, ResolvedIdentity):
        return ""
    override_note = (
        " Admin override is active for this turn."
        if identity.source == IdentitySource.OVERRIDE
        else ""
    )
    return (
        f"ACTIVE USER: {identity.user.display_name} ({identity.user.kind})."
        f"{override_note}"
    )


def build_system_message(
    agent_system_prompt: str,
    tool_instructions: str,
    *,
    mcp_session_prompt: str = "",
    tool_context: str = "",
    extra_system_prompt: str | None = None,
    identity_context: str = "",
    memory_context: str = "",
) -> str:
    """Assemble the system message for the LLM."""
    parts = [agent_system_prompt.strip(), tool_instructions.strip()]
    if identity_context.strip():
        parts.append(identity_context.strip())
    if memory_context.strip():
        parts.append(memory_context.strip())
    if mcp_session_prompt.strip():
        parts.append(mcp_session_prompt.strip())
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
    trimmed_user = user_text.strip()
    last = messages[-1] if messages else None
    if not (
        last
        and last.get("role") == "user"
        and str(last.get("content", "")).strip() == trimmed_user
    ):
        messages.append({"role": "user", "content": trimmed_user})
    return messages


def user_text_from_input(user_input: conversation.ConversationInput) -> str:
    """Extract user text from conversation input."""
    if user_input.text:
        return user_input.text.strip()
    return ""

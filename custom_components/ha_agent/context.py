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
_TURN_OFF = re.compile(r"\bturn\b(?:\s+\w+){0,6}\s+off\b", re.IGNORECASE)
_TURN_ON = re.compile(r"\bturn\b(?:\s+\w+){0,6}\s+on\b", re.IGNORECASE)
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
    "EXPOSED ENTITIES (Assist shortcuts — not a complete list):\n"
    "These are pre-matched entities for faster routing. The home may have "
    "many more devices. When no shortcut fits, or the task needs a different "
    "entity, discover in domain smart-home with searchToolsForDomain, then callTool."
)
_DEVICE_DISCOVERY_FALLBACK = (
    "Discover in domain smart-home with searchToolsForDomain, then callTool. "
    "For homeassistant service calls always pass domain, service, and entity_id."
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
    """Return a short label when a route keyword matches, else None."""
    if route_name == "email":
        pattern = _keyword_regex(keywords) or _EMAIL_QUERY
    elif route_name == "news":
        pattern = _keyword_regex(keywords) or _NEWS_QUERY
    elif route_name == "action":
        if override := _keyword_regex(keywords):
            pattern = override
        elif _CAMERA_ACTION.search(query):
            pattern = _CAMERA_ACTION
        elif _DEVICE_ACTION.search(query):
            pattern = _DEVICE_ACTION
        else:
            return None
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
    """Return True for general conversation routes (not email/news/action)."""
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
    return any(
        token in part for token in tokens for part in parts if part
    )


def _service_hint_for_query(query: str) -> str:
    """Return a homeassistant service hint for common device actions."""
    if is_camera_action_query(query):
        return "snapshot"
    lowered = query.lower()
    if _TURN_OFF.search(query) or "switch off" in lowered:
        return "turn_off"
    if _TURN_ON.search(query) or "switch on" in lowered:
        return "turn_on"
    if "toggle" in lowered:
        return "toggle"
    if "open" in lowered:
        return "open_cover"
    if "close" in lowered:
        return "close_cover"
    if "lock" in lowered:
        return "lock"
    if "unlock" in lowered:
        return "unlock"
    return "turn_on"


def _ha_service_domain_for_query(query: str) -> str:
    """Return the homeassistant domain most likely needed for this query."""
    if is_camera_action_query(query):
        return "camera"
    lowered = query.lower()
    if any(word in lowered for word in ("cover", "blind", "shade", "garage")):
        return "cover"
    if "lock" in lowered:
        return "lock"
    return "light"


def _ha_call_service_example(
    *,
    domain: str,
    service: str,
    entity_id: str,
) -> str:
    payload = {
        "toolName": "home_assistant__ha_call_service",
        "arguments": {
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
        },
    }
    return json.dumps(payload, ensure_ascii=True)


def _entity_discovery_hint(query: str) -> str:
    """Return discovery guidance when no exposed-entity shortcut matches."""
    if is_camera_action_query(query):
        return (
            "Find the camera with home_assistant__ha_search_entities using words "
            "from the user request (e.g. 'front door camera'), then call "
            "home_assistant__ha_call_service with domain camera, service snapshot, "
            "and the matching camera entity_id."
        )
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
    """Return explicit homeassistant call guidance for device actions."""
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
    service = _service_hint_for_query(query)
    domain = _ha_service_domain_for_query(query)
    entity_id = f"{domain}.example"
    call_example = _ha_call_service_example(
        domain=domain,
        service=service,
        entity_id=entity_id,
    )

    if matches:
        lines = [
            "DEVICE ACTION: a matching exposed-entity shortcut was found — use it "
            "directly with home_assistant__ha_call_service. Do NOT call "
            "home_assistant__ha_search_entities (unavailable on most setups). "
            "If the shortcut is wrong or insufficient, discover other entities "
            "in domain smart-home with searchToolsForDomain before calling "
            "home_assistant__ha_call_service. Always include domain, service, "
            f"and entity_id in arguments. Derive domain from the entity_id "
            f"prefix (light.* -> light). Suggested service: {service}. "
            f"Example: {call_example}",
        ]
        for entity in matches:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue
            domain = entity_id.split(".", 1)[0]
            lines.append(
                f"- Use entity_id {entity_id} with domain {domain} "
                f"and service {service}"
            )
        return "\n".join(lines)

    if history_ids := _history_entity_ids(prior_turns):
        lines = [
            "DEVICE ACTION: reuse entity_id values from the prior turn in this "
            f"conversation. Suggested service: {service}.",
        ]
        for entity_id in history_ids:
            domain = entity_id.split(".", 1)[0]
            lines.append(
                f"- Use entity_id {entity_id} with domain {domain} "
                f"and service {service}"
            )
        return "\n".join(lines)

    return (
        "DEVICE ACTION: no exposed-entity shortcut clearly matches. "
        f"{_entity_discovery_hint(query)} "
        f"Example: {call_example}"
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
        "device command and only change the service if needed (turn_on vs turn_off). "
        "Never pass display names as entity_id.",
    ]
    if is_device_action_query(query):
        service = _service_hint_for_query(query)
        lines.append(f"Suggested service for this follow-up: {service}")
    history_text = " ".join(message.get("content", "") for message in history[-6:])
    if entity_ids := _entity_ids_from_text(history_text):
        lines.append(
            "Recent entity_id values from this conversation: "
            + ", ".join(entity_ids)
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

    if route in {"email", "news"} and skill_hints.strip():
        context_parts.append(
            "When ACTIVE SKILLS include a workflow for this route, follow that "
            "workflow first. Use tool_steps only when present; otherwise follow "
            "the markdown workflow text."
        )

    if exposed:
        context_parts.append(
            _EXPOSED_ENTITIES_HEADER + "\n" + format_exposed_entities(exposed)
        )

    if device_hint := _device_action_hint(query, exposed, history=prior_turns):
        context_parts.append(device_hint)

    if follow_up_hint := _follow_up_device_hint(query, prior_turns):
        context_parts.append(follow_up_hint)

    if route == "email" or is_email_query(query):
        context_parts.append(
            "EMAIL: follow MCP SERVER INSTRUCTIONS. Discover in domain email "
            "with searchToolsForDomain, then callTool. Do not search HA entities."
        )

    if route == "news" or is_news_query(query) or (
        is_affirmative(query) and _recent_news_context(prior_turns)
    ):
        context_parts.append(
            "NEWS: call callTool with toolName mcp_news__news_curate. "
            "Use that exact toolName (underscores only, no extra server prefix). "
            "Call it with no arguments ({}) for today's briefing. "
            "Only use searchToolsForDomain if that call fails."
        )

    if _CAPABILITY_QUERY.search(query):
        context_parts.append(
            "CAPABILITIES: explain using MCP SERVER INSTRUCTIONS and MCP SESSION "
            "TOOLS. Mention discovery domains such as email, news, and smart-home."
        )

    return "\n\n".join(context_parts)


def build_system_message(
    agent_system_prompt: str,
    tool_instructions: str,
    *,
    mcp_session_prompt: str = "",
    tool_context: str = "",
    extra_system_prompt: str | None = None,
    route_playbook: str = "",
) -> str:
    """Assemble the system message for the LLM."""
    parts = [agent_system_prompt.strip(), tool_instructions.strip()]
    if route_playbook.strip():
        parts.append(route_playbook.strip())
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

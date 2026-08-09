"""Deterministic extractors for durable memory keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One structured memory write derived from user text / turn context."""

    key: str
    value: Any
    route_scope: str | None = None
    notes: str = ""
    confidence: float = 1.0


_LOCAL_NEWS = re.compile(
    r"\b(?:local(?:\s+news)?|nearby|bay area|around here)\b", re.IGNORECASE
)
_NATIONAL_NEWS = re.compile(
    r"\b(?:national(?:\s+news)?|world news|international)\b", re.IGNORECASE
)
_MAILBOX = re.compile(
    r"\b(?:mailbox|inbox|account)\s*(?:is|=|:)?\s*[\"']?([\w.@+-]+)[\"']?",
    re.IGNORECASE,
)

# Domains commonly aliased for control and sensor lookups.
_ENTITY_DOMAINS = (
    "light|switch|cover|fan|lock|climate|media_player|sensor|binary_sensor|"
    "number|input_number|input_boolean|humidifier|water_heater|camera"
)
_ALIAS = re.compile(
    rf"\b(?:(?:the\s+)?([\w\s-]{{2,40}}?)\s+"
    rf"(?:light|lamp|switch|cover|fan|lock|sensor|entity)"
    rf"|([\w\s-]{{2,40}}?))\s+(?:is|means|=)\s+"
    rf"(?:entity\s+)?({_ENTITY_DOMAINS})\.([\w]+)",
    re.IGNORECASE,
)
_SIMPLE_ALIAS = re.compile(
    rf"\b(?:call|name|map)\s+[\"']?([\w\s-]{{2,40}}?)[\"']?\s+"
    rf"(?:as|to)\s+({_ENTITY_DOMAINS})\.([\w]+)",
    re.IGNORECASE,
)
# "remember this entity is for outdoor air quality"
_THIS_ENTITY_FOR = re.compile(
    r"\b(?:this|that)\s+(?:entity|sensor|one|reading)\s+"
    r"(?:is\s+)?(?:for|as|means)\s+(.+?)\s*$",
    re.IGNORECASE,
)
_CONTROLLED_SUFFIX = re.compile(
    r"\bControlled:\s*([^.]+)\.",
    re.IGNORECASE,
)
_ENTITY_ID_IN_TEXT = re.compile(
    rf"\b((?:{_ENTITY_DOMAINS})\.[a-z0-9_]+)\b",
    re.IGNORECASE,
)


_MEMORY_LEAD_IN = re.compile(
    r"^(?:remember(?:\s+that)?|prefer(?:\s+that)?|from now on|"
    r"always(?:\s+use)?|i prefer|my preference)\s+",
    re.IGNORECASE,
)


def extract_memory_writes(
    user_text: str,
    *,
    fragment: str = "",
    controlled_entity_ids: list[str] | None = None,
    route: str | None = None,
) -> list[ExtractedMemory]:
    """Extract structured memory writes without an LLM when possible."""
    text = (fragment or user_text or "").strip()
    if not text:
        return []
    text = _MEMORY_LEAD_IN.sub("", text).strip() or text

    found: list[ExtractedMemory] = []
    found.extend(_extract_news(text, route=route))
    found.extend(_extract_mailbox(text, route=route))
    found.extend(_extract_aliases(text, controlled_entity_ids or [], route=route))
    return found


def entity_ids_from_history(history: list[dict[str, Any]] | None) -> list[str]:
    """Return recent entity ids from assistant turns (Controlled / turn_meta)."""
    if not history:
        return []
    found: list[str] = []
    for message in reversed(history[-8:]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        batch: list[str] = []
        meta = message.get("turn_meta")
        if isinstance(meta, dict):
            for key in ("referenced_entity_ids", "controlled_entity_ids"):
                raw = meta.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, str) and "." in item and " " not in item:
                            batch.append(item.strip())
        content = str(message.get("content") or "")
        controlled = _CONTROLLED_SUFFIX.search(content)
        if controlled:
            for part in controlled.group(1).split(","):
                eid = part.strip()
                if eid and "." in eid and " " not in eid:
                    batch.append(eid)
        if not batch:
            batch.extend(_ENTITY_ID_IN_TEXT.findall(content))
        if batch:
            for eid in batch:
                if eid not in found:
                    found.append(eid)
            break
    return found


def _extract_news(text: str, *, route: str | None) -> list[ExtractedMemory]:
    if _LOCAL_NEWS.search(text) and re.search(r"\bnews\b", text, re.I):
        return [
            ExtractedMemory(
                key="news.digest_scope",
                value="local",
                route_scope="news",
                notes="Prefer local news briefings",
            )
        ]
    if _NATIONAL_NEWS.search(text):
        return [
            ExtractedMemory(
                key="news.digest_scope",
                value="national",
                route_scope="news",
                notes="Prefer national news briefings",
            )
        ]
    if route == "news" and re.search(r"\b(?:local|nearby)\b", text, re.I):
        return [
            ExtractedMemory(
                key="news.digest_scope",
                value="local",
                route_scope="news",
                notes="Prefer local news briefings",
            )
        ]
    return []


def _extract_mailbox(text: str, *, route: str | None) -> list[ExtractedMemory]:
    match = _MAILBOX.search(text)
    if not match:
        return []
    mailbox = match.group(1).strip()
    return [
        ExtractedMemory(
            key="email.default_mailbox",
            value=mailbox,
            route_scope="email",
            notes=f"Default mailbox {mailbox}",
        )
    ]


def _slugify_alias(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "alias"


def _clean_alias_label(label: str) -> str:
    cleaned = label.strip().strip("\"'").rstrip(".,;:!?")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _alias_write(
    label: str,
    entity_id: str,
    *,
    confidence: float = 1.0,
) -> ExtractedMemory:
    return ExtractedMemory(
        key=f"entity.alias.{_slugify_alias(label)}",
        value=entity_id,
        route_scope="action",
        notes=f"{label} → {entity_id}",
        confidence=confidence,
    )


def _extract_aliases(
    text: str,
    controlled_entity_ids: list[str],
    *,
    route: str | None,
) -> list[ExtractedMemory]:
    del route  # reserved for future route-scoped alias rules
    results: list[ExtractedMemory] = []
    match = _ALIAS.search(text)
    if match:
        label = _clean_alias_label(match.group(1) or match.group(2) or "")
        domain = match.group(3)
        object_id = match.group(4)
        entity_id = f"{domain}.{object_id}"
        if label:
            results.append(_alias_write(label, entity_id))
            return results

    match = _SIMPLE_ALIAS.search(text)
    if match:
        label = _clean_alias_label(match.group(1))
        entity_id = f"{match.group(2)}.{match.group(3)}"
        results.append(_alias_write(label, entity_id))
        return results

    # "remember this entity is for outdoor air quality" after a lookup turn
    if controlled_entity_ids and re.search(
        r"\b(?:remember|prefer|always|default|means|is|for|as)\b", text, re.I
    ):
        entity_id = controlled_entity_ids[-1]
        this_match = _THIS_ENTITY_FOR.search(text)
        if this_match:
            label = _clean_alias_label(this_match.group(1))
            if label:
                results.append(_alias_write(label, entity_id, confidence=0.9))
                return results
        label_match = re.search(
            r"\b(?:the\s+)?([\w\s-]{2,40}?)\s+"
            r"(?:light|lamp|switch|cover|fan|lock|sensor|entity)\b",
            text,
            re.I,
        )
        if label_match:
            label = _clean_alias_label(label_match.group(1))
            if label and label.lower() not in {"this", "that", "the"}:
                results.append(_alias_write(label, entity_id, confidence=0.8))
    return results


# Keys that apply only on matching routes (plus always-global keys).
ROUTE_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "news": ("news.",),
    "email": ("email.",),
    "action": ("entity.alias.", "ha."),
    "chat": (),
}


def keys_for_route(route: str | None) -> tuple[str, ...] | None:
    """Return key prefixes relevant to a route, or None for all keys."""
    if not route:
        return None
    return ROUTE_KEY_PREFIXES.get(route)

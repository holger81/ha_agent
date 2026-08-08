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
_ALIAS = re.compile(
    r"\b(?:(?:the\s+)?([\w\s-]{2,40}?)\s+(?:light|lamp|switch|cover|fan|lock)"
    r"|([\w\s-]{2,40}?))\s+(?:is|means|=)\s+"
    r"(?:entity\s+)?(light|switch|cover|fan|lock|climate|media_player)\.([\w]+)",
    re.IGNORECASE,
)
_SIMPLE_ALIAS = re.compile(
    r"\b(?:call|name|map)\s+[\"']?([\w\s-]{2,40}?)[\"']?\s+"
    r"(?:as|to)\s+(light|switch|cover|fan|lock|climate|media_player)\.([\w]+)",
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

    found: list[ExtractedMemory] = []
    found.extend(_extract_news(text, route=route))
    found.extend(_extract_mailbox(text, route=route))
    found.extend(_extract_aliases(text, controlled_entity_ids or [], route=route))
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


def _extract_aliases(
    text: str,
    controlled_entity_ids: list[str],
    *,
    route: str | None,
) -> list[ExtractedMemory]:
    results: list[ExtractedMemory] = []
    match = _ALIAS.search(text)
    if match:
        label = (match.group(1) or match.group(2) or "").strip()
        domain = match.group(3)
        object_id = match.group(4)
        entity_id = f"{domain}.{object_id}"
        if label:
            results.append(
                ExtractedMemory(
                    key=f"entity.alias.{_slugify_alias(label)}",
                    value=entity_id,
                    route_scope="action",
                    notes=f"{label} → {entity_id}",
                )
            )
            return results

    match = _SIMPLE_ALIAS.search(text)
    if match:
        label = match.group(1).strip()
        entity_id = f"{match.group(2)}.{match.group(3)}"
        results.append(
            ExtractedMemory(
                key=f"entity.alias.{_slugify_alias(label)}",
                value=entity_id,
                route_scope="action",
                notes=f"{label} → {entity_id}",
            )
        )
        return results

    # "remember that dining room light" after a successful control turn
    if controlled_entity_ids and re.search(
        r"\b(?:remember|prefer|always|default|means|is)\b", text, re.I
    ):
        entity_id = controlled_entity_ids[-1]
        label_match = re.search(
            r"\b(?:the\s+)?([\w\s-]{2,40}?)\s+(?:light|lamp|switch|cover|fan|lock)\b",
            text,
            re.I,
        )
        if label_match:
            label = label_match.group(1).strip()
            results.append(
                ExtractedMemory(
                    key=f"entity.alias.{_slugify_alias(label)}",
                    value=entity_id,
                    route_scope="action",
                    notes=f"{label} → {entity_id}",
                    confidence=0.8,
                )
            )
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

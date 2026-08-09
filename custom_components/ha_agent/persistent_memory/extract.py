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

# Alias label → expected entity domains / id markers (readings vs controls).
_SENSOR_DOMAINS = frozenset({"sensor", "binary_sensor", "number", "input_number"})
_CONTROL_DOMAINS = frozenset(
    {
        "light",
        "switch",
        "cover",
        "fan",
        "lock",
        "climate",
        "media_player",
        "humidifier",
        "water_heater",
        "input_boolean",
        "camera",
    }
)
_READING_ALIAS_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aqi", ("aqi", "air quality", "air_quality", "airquality", "pm2.5", "pm25")),
    ("temperature", ("temperature", "temp", "how warm", "how cold")),
    ("humidity", ("humidity",)),
    ("pressure", ("pressure",)),
    ("co2", ("co2", "co₂")),
)
_CONTROL_ALIAS_KINDS: tuple[tuple[str, frozenset[str], tuple[str, ...]], ...] = (
    ("light", frozenset({"light", "switch"}), ("light", "lamp", "lights")),
    ("switch", frozenset({"switch", "light"}), ("switch",)),
    ("cover", frozenset({"cover"}), ("cover", "blind", "shade", "curtain")),
    ("fan", frozenset({"fan"}), ("fan",)),
    ("lock", frozenset({"lock"}), ("lock", "door lock")),
    ("climate", frozenset({"climate"}), ("thermostat", "climate", "hvac")),
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
    """Return recent entity ids from assistant turns (lookups before controls).

    Prefer ``referenced_entity_ids`` (sensor lookups) over ``controlled_entity_ids``
    so “remember this entity…” after a reading does not latch onto an older light.
    """
    if not history:
        return []
    found: list[str] = []
    for message in reversed(history[-8:]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        referenced: list[str] = []
        controlled: list[str] = []
        meta = message.get("turn_meta")
        if isinstance(meta, dict):
            raw_ref = meta.get("referenced_entity_ids")
            if isinstance(raw_ref, list):
                for item in raw_ref:
                    if isinstance(item, str) and "." in item and " " not in item:
                        referenced.append(item.strip())
            raw_ctrl = meta.get("controlled_entity_ids")
            if isinstance(raw_ctrl, list):
                for item in raw_ctrl:
                    if isinstance(item, str) and "." in item and " " not in item:
                        controlled.append(item.strip())
        content = str(message.get("content") or "")
        controlled_match = _CONTROLLED_SUFFIX.search(content)
        if controlled_match:
            for part in controlled_match.group(1).split(","):
                eid = part.strip()
                if eid and "." in eid and " " not in eid:
                    # "Controlled:" lines often list the looked-up sensor too.
                    domain = eid.split(".", 1)[0].lower()
                    if domain in _SENSOR_DOMAINS:
                        referenced.append(eid)
                    else:
                        controlled.append(eid)
        if not referenced and not controlled:
            for eid in _ENTITY_ID_IN_TEXT.findall(content):
                domain = eid.split(".", 1)[0].lower()
                if domain in _SENSOR_DOMAINS:
                    referenced.append(eid)
                else:
                    controlled.append(eid)
        batch = referenced + controlled
        if batch:
            for eid in batch:
                if eid not in found:
                    found.append(eid)
            # Keep scanning older turns so a prior light control does not
            # starve a more recent reading — but stop once we have refs.
            if referenced:
                break
            if len(found) >= 8:
                break
    return found


def _entity_domain(entity_id: str) -> str:
    return entity_id.split(".", 1)[0].lower() if "." in entity_id else ""


def _infer_reading_alias_kind(label: str) -> str | None:
    text = (label or "").strip().lower()
    if not text:
        return None
    for kind, markers in _READING_ALIAS_KINDS:
        for marker in markers:
            if marker in text:
                return kind
    return None


def _infer_control_alias_kind(
    label: str,
) -> tuple[str, frozenset[str]] | None:
    text = (label or "").strip().lower()
    if not text:
        return None
    for kind, domains, markers in _CONTROL_ALIAS_KINDS:
        if any(marker in text for marker in markers):
            return kind, domains
    return None


def _entity_matches_reading_alias(entity_id: str, kind: str) -> bool:
    text = (entity_id or "").strip().lower()
    if not text or _entity_domain(text) not in _SENSOR_DOMAINS:
        return False
    markers = dict(_READING_ALIAS_KINDS).get(kind, (kind,))
    blob = text.replace(".", " ").replace("_", " ")
    return any(marker.replace(" ", "_") in text or marker in blob for marker in markers)


def alias_entity_compatible(label: str, entity_id: str) -> bool:
    """True when the alias label and entity domain/type plausibly match."""
    eid = (entity_id or "").strip()
    if not eid or "." not in eid or " " in eid:
        return False
    domain = _entity_domain(eid)
    reading = _infer_reading_alias_kind(label)
    if reading:
        return _entity_matches_reading_alias(eid, reading)
    control = _infer_control_alias_kind(label)
    if control:
        _kind, domains = control
        return domain in domains
    # Generic label: allow sensors or controls, but never mix a bare
    # "quality"/"outdoor" reading-ish phrase onto a light via history.
    return domain in _SENSOR_DOMAINS or domain in _CONTROL_DOMAINS


def select_entity_for_alias(label: str, entity_ids: list[str] | None) -> str | None:
    """Pick the best history entity for an alias label, or None if none fit."""
    if not entity_ids:
        return None
    reading = _infer_reading_alias_kind(label)
    if reading:
        for eid in entity_ids:
            if _entity_matches_reading_alias(eid, reading):
                return eid
        return None
    control = _infer_control_alias_kind(label)
    if control:
        _kind, domains = control
        for eid in entity_ids:
            if _entity_domain(eid) in domains:
                return eid
        return None
    for eid in entity_ids:
        if alias_entity_compatible(label, eid):
            return eid
    return None


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


def _alias_route_scope(entity_id: str) -> str | None:
    """Sensor aliases are global; control aliases stay action-scoped."""
    if _entity_domain(entity_id) in _SENSOR_DOMAINS:
        return None
    return "action"


def _alias_write(
    label: str,
    entity_id: str,
    *,
    confidence: float = 1.0,
) -> ExtractedMemory | None:
    if not alias_entity_compatible(label, entity_id):
        return None
    return ExtractedMemory(
        key=f"entity.alias.{_slugify_alias(label)}",
        value=entity_id,
        route_scope=_alias_route_scope(entity_id),
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
            write = _alias_write(label, entity_id)
            if write:
                results.append(write)
            return results

    match = _SIMPLE_ALIAS.search(text)
    if match:
        label = _clean_alias_label(match.group(1))
        entity_id = f"{match.group(2)}.{match.group(3)}"
        write = _alias_write(label, entity_id)
        if write:
            results.append(write)
        return results

    # "remember this entity is for outdoor air quality" after a lookup turn
    if controlled_entity_ids and re.search(
        r"\b(?:remember|prefer|always|default|means|is|for|as)\b", text, re.I
    ):
        this_match = _THIS_ENTITY_FOR.search(text)
        if this_match:
            label = _clean_alias_label(this_match.group(1))
            entity_id = select_entity_for_alias(label, controlled_entity_ids)
            if label and entity_id:
                write = _alias_write(label, entity_id, confidence=0.9)
                if write:
                    results.append(write)
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
                entity_id = select_entity_for_alias(label, controlled_entity_ids)
                if entity_id:
                    write = _alias_write(label, entity_id, confidence=0.8)
                    if write:
                        results.append(write)
    return results


# Keys that apply only on matching routes (plus always-global keys).
ROUTE_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "news": ("news.",),
    "email": ("email.",),
    "action": ("entity.alias.", "ha."),
    # Empty → inject keeps all keys (sensor aliases are route_scope=None).
    "chat": (),
}


def keys_for_route(route: str | None) -> tuple[str, ...] | None:
    """Return key prefixes relevant to a route, or None for all keys."""
    if not route:
        return None
    return ROUTE_KEY_PREFIXES.get(route)

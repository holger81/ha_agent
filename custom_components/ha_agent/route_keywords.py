"""Editable route trigger keywords.

Route detection (email / news / action) ships as compiled regexes in
``context.py``. This module makes the trigger keyword list for each built-in
route editable and resettable from the console UI, persisted per config entry
in SQLite. When a route override is disabled, empty, or unchanged from the
shipped default, the deterministic shipped regex is used so behavior never
regresses and a storage problem never breaks a turn.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from homeassistant.core import HomeAssistant

from .const import DATA_KEY, LOGGER

ROUTE_KEYWORDS_STORE_KEY = "route_keyword_stores"

# Ordered, editable built-in routes. Each maps to a router route value.
ROUTE_KEYWORD_ROUTES = ("email", "news", "action")

DEFAULT_ROUTE_KEYWORDS: dict[str, dict[str, object]] = {
    "email": {
        "title": "Email",
        "keywords": ["email", "emails", "e-mail", "mail", "inbox", "unread"],
    },
    "news": {
        "title": "News",
        "keywords": ["news", "headlines", "headline", "briefing", "nachrichten"],
    },
    "action": {
        "title": "Device action",
        "keywords": [
            "open",
            "close",
            "toggle",
            "lock",
            "unlock",
            "switch on",
            "switch off",
            "turn on",
            "turn off",
            "snapshot",
            "photo",
            "picture",
            "capture",
            "camera",
        ],
    },
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_keywords (
    route TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    keywords TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class RouteKeywords:
    """An editable trigger keyword list for a built-in route."""

    route: str
    title: str
    keywords: list[str] = field(default_factory=list)
    enabled: bool = True
    updated_at: float = 0.0
    is_default: bool = True


def default_route_keywords(route: str) -> list[str]:
    """Return the shipped default keyword list for a route."""
    default = DEFAULT_ROUTE_KEYWORDS.get(route)
    if not default:
        return []
    return list(default["keywords"])  # type: ignore[arg-type]


def _normalize_keywords(keywords: list[str] | None) -> list[str]:
    """Strip, drop empties, and de-duplicate keywords preserving order."""
    cleaned: list[str] = []
    for keyword in keywords or []:
        text = str(keyword).strip()
        if text and text.lower() not in {k.lower() for k in cleaned}:
            cleaned.append(text)
    return cleaned


def _is_default(route: str, keywords: list[str]) -> bool:
    default = default_route_keywords(route)
    return [k.lower() for k in keywords] == [k.lower() for k in default]


def _row_to_route_keywords(row: sqlite3.Row) -> RouteKeywords:
    try:
        keywords = json.loads(row["keywords"])
    except (json.JSONDecodeError, TypeError):
        keywords = []
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k) for k in keywords]
    return RouteKeywords(
        route=row["route"],
        title=row["title"],
        keywords=keywords,
        enabled=bool(row["enabled"]),
        updated_at=float(row["updated_at"]),
        is_default=_is_default(row["route"], keywords),
    )


class RouteKeywordStore:
    """Per-config-entry editable route keyword database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the on-disk path for an entry's route keyword database."""
        storage = Path(hass.config.path(".storage"))
        return storage / f"ha_agent_route_keywords_{entry_id}.db"

    def connect(self) -> None:
        """Open the database, ensure schema, and seed missing defaults."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._seed_defaults()
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def _seed_defaults(self) -> None:
        conn = self._conn
        assert conn is not None
        now = time.time()
        for route in ROUTE_KEYWORD_ROUTES:
            default = DEFAULT_ROUTE_KEYWORDS[route]
            conn.execute(
                "INSERT OR IGNORE INTO route_keywords "
                "(route, title, keywords, enabled, updated_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (
                    route,
                    default["title"],
                    json.dumps(default["keywords"]),
                    now,
                ),
            )

    def list_route_keywords(self) -> list[RouteKeywords]:
        """Return built-in route keyword rows in canonical order."""
        rows = self._connection().execute(
            "SELECT * FROM route_keywords",
        ).fetchall()
        by_route = {row["route"]: _row_to_route_keywords(row) for row in rows}
        return [
            by_route[route]
            for route in ROUTE_KEYWORD_ROUTES
            if route in by_route
        ]

    def get_route_keywords(self, route: str) -> RouteKeywords | None:
        """Return one route's keyword row."""
        row = self._connection().execute(
            "SELECT * FROM route_keywords WHERE route = ?",
            (route,),
        ).fetchone()
        return _row_to_route_keywords(row) if row else None

    def update_route_keywords(
        self,
        route: str,
        *,
        keywords: list[str] | None = None,
        enabled: bool | None = None,
    ) -> RouteKeywords | None:
        """Update an existing route's editable fields."""
        current = self.get_route_keywords(route)
        if current is None:
            return None
        if keywords is not None:
            current.keywords = _normalize_keywords(keywords)
        if enabled is not None:
            current.enabled = enabled
        current.updated_at = time.time()
        conn = self._connection()
        conn.execute(
            "UPDATE route_keywords SET keywords = ?, enabled = ?, "
            "updated_at = ? WHERE route = ?",
            (
                json.dumps(current.keywords),
                int(current.enabled),
                current.updated_at,
                current.route,
            ),
        )
        conn.commit()
        current.is_default = _is_default(current.route, current.keywords)
        return current

    def reset_route_keywords(self, route: str) -> RouteKeywords | None:
        """Restore a route's keywords to the shipped default and enable it."""
        if route not in DEFAULT_ROUTE_KEYWORDS:
            return None
        return self.update_route_keywords(
            route,
            keywords=default_route_keywords(route),
            enabled=True,
        )

    def active_keywords(self, route: str) -> list[str] | None:
        """Return the keyword override to apply, or None to use the default.

        Returns ``None`` (use the shipped regex) when the route is disabled,
        has no keywords, or is unchanged from the shipped default. Only a
        customized, enabled, non-empty list overrides the default matcher.
        """
        current = self.get_route_keywords(route)
        if current is None or not current.enabled or not current.keywords:
            return None
        if _is_default(route, current.keywords):
            return None
        return current.keywords

    def active_keyword_map(self) -> dict[str, list[str]]:
        """Return active keyword overrides for all customized routes."""
        result: dict[str, list[str]] = {}
        for route in ROUTE_KEYWORD_ROUTES:
            keywords = self.active_keywords(route)
            if keywords:
                result[route] = keywords
        return result


async def async_route_keyword_map(
    hass: HomeAssistant,
    entry_id: str,
) -> dict[str, list[str]]:
    """Return active route keyword overrides for this turn.

    Falls back to an empty map (shipped regexes) when the store cannot be read
    so a storage problem never breaks a turn.
    """
    try:
        store = get_route_keyword_store(hass, entry_id)
        return await hass.async_add_executor_job(store.active_keyword_map)
    except Exception as err:
        LOGGER.debug("Route keyword store unavailable, using defaults: %s", err)
        return {}


def get_route_keyword_store(
    hass: HomeAssistant,
    entry_id: str,
) -> RouteKeywordStore:
    """Return the route keyword store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, RouteKeywordStore] = domain_data.setdefault(
        ROUTE_KEYWORDS_STORE_KEY, {}
    )
    if entry_id not in stores:
        store = RouteKeywordStore(
            RouteKeywordStore.db_path_for_entry(hass, entry_id)
        )
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_route_keyword_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove a route keyword store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, RouteKeywordStore] = domain_data.get(
        ROUTE_KEYWORDS_STORE_KEY, {}
    )
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

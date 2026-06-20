"""Editable route playbooks.

Playbooks are short, route-pinned workflow recipes injected into the system
prompt for every turn on a given route. Unlike learned skills (which are
matched per turn), a route's playbook is always injected when enabled. The
default text ships with the integration but is fully editable from the console
UI and persisted per config entry in SQLite.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from homeassistant.core import HomeAssistant

from .const import DATA_KEY, LOGGER

PLAYBOOKS_STORE_KEY = "playbook_stores"

# Ordered playbook routes. ``general`` is the fallback used for the chat route
# and any route without a dedicated playbook.
PLAYBOOK_ROUTES = ("email", "news", "action", "general")

DEFAULT_PLAYBOOKS: dict[str, dict[str, str]] = {
    "email": {
        "title": "Email",
        "body": (
            "EMAIL PLAYBOOK:\n"
            "1. Discover tools in domain email if needed.\n"
            "2. Call mailbox_status for unseen count.\n"
            "3. Call search_messages with unread_only=true and a small limit.\n"
            "4. Call get_message only for messages you will cite.\n"
            "5. Answer using tool results only; never invent subjects or counts."
        ),
    },
    "news": {
        "title": "News",
        "body": (
            "NEWS PLAYBOOK:\n"
            "1. Call mcp_news__news_curate with no arguments ({}) for "
            "today's briefing.\n"
            "2. Summarize headlines from that result only.\n"
            "3. Use searchToolsForDomain only if news_curate fails."
        ),
    },
    "action": {
        "title": "Device action",
        "body": (
            "DEVICE PLAYBOOK:\n"
            "1. Prefer an exposed-entity shortcut when one clearly matches.\n"
            "2. If no shortcut fits, discover entities in domain smart-home "
            "with searchToolsForDomain, then callTool.\n"
            "3. Call ha_call_service with domain, service, and entity_id "
            "(e.g. camera.snapshot for photos).\n"
            "4. Read VERIFICATION lines in tool results before telling the user "
            "the action succeeded."
        ),
    },
    "general": {
        "title": "General",
        "body": (
            "GENERAL PLAYBOOK:\n"
            "Gather evidence with tools before answering. Cite tool results. "
            "Exposed entities in context are shortcuts only; discover more in "
            "domain smart-home when needed. If a tool fails, change strategy "
            "using RECOVERY HINTS."
        ),
    },
}

# Map a router route value (TaskRoute.value) to a playbook route key.
_ROUTE_TO_PLAYBOOK = {
    "email": "email",
    "news": "news",
    "action": "action",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS playbooks (
    route TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class Playbook:
    """A route-pinned, editable workflow recipe."""

    route: str
    title: str
    body: str
    enabled: bool = True
    updated_at: float = 0.0
    is_default: bool = True


def playbook_key_for_route(route_value: str) -> str:
    """Return the playbook key for a router route value."""
    return _ROUTE_TO_PLAYBOOK.get(route_value, "general")


def default_playbook_body(route_key: str) -> str:
    """Return the shipped default body for a playbook route."""
    default = DEFAULT_PLAYBOOKS.get(route_key) or DEFAULT_PLAYBOOKS["general"]
    return default["body"]


def _is_default(route: str, body: str) -> bool:
    default = DEFAULT_PLAYBOOKS.get(route)
    return bool(default) and body.strip() == default["body"].strip()


def _row_to_playbook(row: sqlite3.Row) -> Playbook:
    return Playbook(
        route=row["route"],
        title=row["title"],
        body=row["body"],
        enabled=bool(row["enabled"]),
        updated_at=float(row["updated_at"]),
        is_default=_is_default(row["route"], row["body"]),
    )


class PlaybookStore:
    """Per-config-entry editable playbook database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the on-disk path for an entry's playbook database."""
        storage = Path(hass.config.path(".storage"))
        return storage / f"ha_agent_playbooks_{entry_id}.db"

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
        for route in PLAYBOOK_ROUTES:
            default = DEFAULT_PLAYBOOKS[route]
            conn.execute(
                "INSERT OR IGNORE INTO playbooks "
                "(route, title, body, enabled, updated_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (route, default["title"], default["body"], now),
            )

    def list_playbooks(self) -> list[Playbook]:
        """Return all playbooks in canonical route order."""
        rows = self._connection().execute(
            "SELECT * FROM playbooks",
        ).fetchall()
        by_route = {row["route"]: _row_to_playbook(row) for row in rows}
        ordered = [by_route[route] for route in PLAYBOOK_ROUTES if route in by_route]
        extra = [pb for route, pb in by_route.items() if route not in PLAYBOOK_ROUTES]
        return ordered + extra

    def get_playbook(self, route: str) -> Playbook | None:
        """Return one playbook by route key."""
        row = self._connection().execute(
            "SELECT * FROM playbooks WHERE route = ?",
            (route,),
        ).fetchone()
        return _row_to_playbook(row) if row else None

    def update_playbook(
        self,
        route: str,
        *,
        title: str | None = None,
        body: str | None = None,
        enabled: bool | None = None,
    ) -> Playbook | None:
        """Update an existing playbook's editable fields."""
        playbook = self.get_playbook(route)
        if playbook is None:
            return None
        if title is not None:
            playbook.title = title.strip() or playbook.title
        if body is not None:
            playbook.body = body.strip()
        if enabled is not None:
            playbook.enabled = enabled
        playbook.updated_at = time.time()
        conn = self._connection()
        conn.execute(
            "UPDATE playbooks SET title = ?, body = ?, enabled = ?, "
            "updated_at = ? WHERE route = ?",
            (
                playbook.title,
                playbook.body,
                int(playbook.enabled),
                playbook.updated_at,
                playbook.route,
            ),
        )
        conn.commit()
        playbook.is_default = _is_default(playbook.route, playbook.body)
        return playbook

    def reset_playbook(self, route: str) -> Playbook | None:
        """Restore a playbook to its shipped default and enable it."""
        default = DEFAULT_PLAYBOOKS.get(route)
        if default is None:
            return None
        return self.update_playbook(
            route,
            title=default["title"],
            body=default["body"],
            enabled=True,
        )

    def active_body(self, route_key: str) -> str:
        """Return the body to inject, or '' when disabled."""
        playbook = self.get_playbook(route_key)
        if playbook is None:
            return default_playbook_body(route_key)
        if not playbook.enabled:
            return ""
        return playbook.body


async def async_route_playbook(
    hass: HomeAssistant,
    entry_id: str,
    route_value: str,
) -> str:
    """Return the active (UI-editable) playbook text for a route.

    Falls back to the shipped default when the store cannot be read so a
    storage problem never breaks a turn.
    """
    key = playbook_key_for_route(route_value)
    try:
        store = get_playbook_store(hass, entry_id)
        return await hass.async_add_executor_job(store.active_body, key)
    except Exception as err:
        LOGGER.debug("Falling back to default playbook for %s: %s", key, err)
        return default_playbook_body(key)


def get_playbook_store(hass: HomeAssistant, entry_id: str) -> PlaybookStore:
    """Return the playbook store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, PlaybookStore] = domain_data.setdefault(PLAYBOOKS_STORE_KEY, {})
    if entry_id not in stores:
        store = PlaybookStore(PlaybookStore.db_path_for_entry(hass, entry_id))
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_playbook_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove a playbook store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, PlaybookStore] = domain_data.get(PLAYBOOKS_STORE_KEY, {})
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

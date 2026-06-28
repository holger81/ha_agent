"""Editable route playbooks.

Playbooks are short, route-pinned workflow recipes injected into the system
prompt for every turn on a given route. Unlike learned skills (which are
matched per turn), a route's playbook is always injected when enabled. The
default text ships with the integration but is fully editable from the console
UI and persisted per config entry in SQLite.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.core import HomeAssistant

from .const import DATA_KEY, LOGGER

if TYPE_CHECKING:
    from .config_helpers import LlmBackend
    from .llm_client import LlmClient

PLAYBOOKS_STORE_KEY = "playbook_stores"

# Ordered playbook routes. ``general`` is the fallback used for the chat route
# and any route without a dedicated playbook.
PLAYBOOK_ROUTES = ("email", "news", "action", "general")

DEFAULT_PLAYBOOKS: dict[str, dict[str, str]] = {
    "email": {
        "title": "Email",
        "match": "The user asks about email, mail, inbox, or unread messages.",
        "body": (
            "EMAIL PLAYBOOK:\n"
            "1. Discover tools in domain email if needed.\n"
            "2. Call `mail_mcp__imap_mailbox_status` with mailbox INBOX "
            "for unseen count.\n"
            "3. Call `mail_mcp__imap_search_messages` with mailbox INBOX, "
            "unread_only=true, and a small limit (e.g. 10).\n"
            "4. Call `mail_mcp__imap_get_message` only for messages "
            "you will cite (message_id from search results).\n"
            "5. Answer using tool results only; never invent subjects or counts."
        ),
    },
    "news": {
        "title": "News",
        "match": "The user asks for news, headlines, or a briefing.",
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
        "match": (
            "The user asks to control or check a device, such as lights, "
            "switches, covers, locks, climate, or a camera snapshot."
        ),
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
        "match": "Fallback for general requests that still need tools or evidence.",
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
    updated_at REAL NOT NULL,
    match_text TEXT NOT NULL DEFAULT '',
    is_builtin INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the initial release; applied via ALTER on existing DBs.
_ADDED_COLUMNS = {
    "match_text": "TEXT NOT NULL DEFAULT ''",
    "is_builtin": "INTEGER NOT NULL DEFAULT 1",
    "priority": "INTEGER NOT NULL DEFAULT 0",
}


@dataclass(slots=True)
class Playbook:
    """An editable workflow recipe (built-in route or custom rule)."""

    route: str
    title: str
    body: str
    enabled: bool = True
    updated_at: float = 0.0
    is_default: bool = True
    match_text: str = ""
    is_builtin: bool = True
    priority: int = 0


@dataclass(frozen=True, slots=True)
class PlaybookSelection:
    """Playbook chosen for one agent turn."""

    body: str
    key: str
    method: str
    detail: str


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
    keys = row.keys()
    is_builtin = bool(row["is_builtin"]) if "is_builtin" in keys else True
    return Playbook(
        route=row["route"],
        title=row["title"],
        body=row["body"],
        enabled=bool(row["enabled"]),
        updated_at=float(row["updated_at"]),
        is_default=is_builtin and _is_default(row["route"], row["body"]),
        match_text=row["match_text"] if "match_text" in keys else "",
        is_builtin=is_builtin,
        priority=int(row["priority"]) if "priority" in keys else 0,
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
        self._migrate()
        self._seed_defaults()
        self._conn.commit()

    def _migrate(self) -> None:
        conn = self._conn
        assert conn is not None
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(playbooks)").fetchall()
        }
        for column, ddl in _ADDED_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE playbooks ADD COLUMN {column} {ddl}")

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
        for index, route in enumerate(PLAYBOOK_ROUTES):
            default = DEFAULT_PLAYBOOKS[route]
            conn.execute(
                "INSERT OR IGNORE INTO playbooks "
                "(route, title, body, enabled, updated_at, match_text, "
                "is_builtin, priority) "
                "VALUES (?, ?, ?, 1, ?, ?, 1, ?)",
                (
                    route,
                    default["title"],
                    default["body"],
                    now,
                    default["match"],
                    index,
                ),
            )
            # Backfill match_text for rows migrated from the pre-match schema.
            conn.execute(
                "UPDATE playbooks SET match_text = ? "
                "WHERE route = ? AND (match_text IS NULL OR match_text = '')",
                (default["match"], route),
            )

    def list_playbooks(self) -> list[Playbook]:
        """Return built-in playbooks (canonical order) then custom rules."""
        rows = self._connection().execute(
            "SELECT * FROM playbooks",
        ).fetchall()
        by_route = {row["route"]: _row_to_playbook(row) for row in rows}
        ordered = [by_route[route] for route in PLAYBOOK_ROUTES if route in by_route]
        extra = [
            pb
            for route, pb in by_route.items()
            if route not in PLAYBOOK_ROUTES
        ]
        extra.sort(key=lambda pb: (pb.priority, pb.updated_at))
        return ordered + extra

    def list_enabled(self) -> list[Playbook]:
        """Return all enabled playbooks for runtime selection."""
        return [pb for pb in self.list_playbooks() if pb.enabled]

    def custom_count(self) -> int:
        """Return the number of user-added custom playbook rules."""
        row = self._connection().execute(
            "SELECT COUNT(*) AS c FROM playbooks WHERE is_builtin = 0",
        ).fetchone()
        return int(row["c"]) if row else 0

    def get_playbook(self, route: str) -> Playbook | None:
        """Return one playbook by route key."""
        row = self._connection().execute(
            "SELECT * FROM playbooks WHERE route = ?",
            (route,),
        ).fetchone()
        return _row_to_playbook(row) if row else None

    def create_playbook(
        self,
        *,
        title: str,
        body: str,
        match_text: str = "",
        enabled: bool = True,
    ) -> Playbook:
        """Create a custom (non-built-in) playbook rule."""
        now = time.time()
        playbook = Playbook(
            route=f"custom-{uuid.uuid4().hex[:12]}",
            title=title.strip() or "Custom playbook",
            body=body.strip(),
            enabled=enabled,
            updated_at=now,
            is_default=False,
            match_text=match_text.strip(),
            is_builtin=False,
            priority=1000,
        )
        conn = self._connection()
        conn.execute(
            "INSERT INTO playbooks "
            "(route, title, body, enabled, updated_at, match_text, "
            "is_builtin, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (
                playbook.route,
                playbook.title,
                playbook.body,
                int(playbook.enabled),
                playbook.updated_at,
                playbook.match_text,
                playbook.priority,
            ),
        )
        conn.commit()
        return playbook

    def delete_playbook(self, route: str) -> bool:
        """Delete a custom playbook rule. Built-ins cannot be deleted."""
        playbook = self.get_playbook(route)
        if playbook is None or playbook.is_builtin:
            return False
        conn = self._connection()
        conn.execute("DELETE FROM playbooks WHERE route = ?", (route,))
        conn.commit()
        return True

    def update_playbook(
        self,
        route: str,
        *,
        title: str | None = None,
        body: str | None = None,
        match_text: str | None = None,
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
        if match_text is not None:
            playbook.match_text = match_text.strip()
        if enabled is not None:
            playbook.enabled = enabled
        playbook.updated_at = time.time()
        conn = self._connection()
        conn.execute(
            "UPDATE playbooks SET title = ?, body = ?, match_text = ?, "
            "enabled = ?, updated_at = ? WHERE route = ?",
            (
                playbook.title,
                playbook.body,
                playbook.match_text,
                int(playbook.enabled),
                playbook.updated_at,
                playbook.route,
            ),
        )
        conn.commit()
        playbook.is_default = playbook.is_builtin and _is_default(
            playbook.route, playbook.body
        )
        return playbook

    def reset_playbook(self, route: str) -> Playbook | None:
        """Restore a built-in playbook to its shipped default and enable it."""
        default = DEFAULT_PLAYBOOKS.get(route)
        if default is None:
            return None
        return self.update_playbook(
            route,
            title=default["title"],
            body=default["body"],
            match_text=default["match"],
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


_SELECT_PROMPT = (
    "You pick which playbook (if any) best fits the user's latest request.\n"
    'Return ONLY valid JSON: {{"route": "exact-route"}} or {{"route": null}}.\n'
    "Rules:\n"
    "- Choose at most one route from AVAILABLE PLAYBOOKS.\n"
    "- Use the 'when_to_apply' text to decide.\n"
    "- Return null only when none reasonably fit; prefer 'general' as a "
    "catch-all when it is listed."
)


def parse_playbook_selection(content: str, valid_routes: set[str]) -> str | None:
    """Parse the selected route from an LLM selection response."""
    text = (content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    route = data.get("route")
    if isinstance(route, str) and route.strip() in valid_routes:
        return route.strip()
    return None


async def select_playbook_with_llm(
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    history: list[dict[str, str]] | None,
    catalog: list[Playbook],
) -> Playbook | None:
    """Ask the chat model which enabled playbook applies to this turn."""
    if not catalog:
        return None
    entries = [
        {"route": pb.route, "title": pb.title, "when_to_apply": pb.match_text}
        for pb in catalog
    ]
    recent = [
        turn.get("content", "")
        for turn in (history or [])[-4:]
        if turn.get("role") == "user"
    ]
    messages = [
        {"role": "system", "content": _SELECT_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_text": user_text,
                    "recent_user_turns": recent,
                    "available_playbooks": entries,
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        result = await llm.chat(messages, backend, tools=[])
    except Exception as err:
        LOGGER.warning("Playbook selection LLM call failed: %s", err)
        return None
    valid = {pb.route for pb in catalog}
    route = parse_playbook_selection(result.content or "", valid)
    if route is None:
        return None
    return next((pb for pb in catalog if pb.route == route), None)


async def async_select_playbook(
    hass: HomeAssistant,
    entry_id: str,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    user_text: str,
    route_value: str,
    history: list[dict[str, str]] | None = None,
) -> PlaybookSelection:
    """Return the playbook to inject for this turn and how it was chosen.

    When the user has added custom rules, an LLM classifier chooses among all
    enabled playbooks; otherwise (and on any failure) the keyword route's
    built-in playbook is used.
    """
    key = playbook_key_for_route(route_value)
    try:
        store = get_playbook_store(hass, entry_id)
        catalog, custom = await hass.async_add_executor_job(
            _selection_inputs, store
        )
    except Exception as err:
        LOGGER.debug("Playbook store unavailable, using default: %s", err)
        body = default_playbook_body(key)
        return PlaybookSelection(
            body=body,
            key=key,
            method="route",
            detail="default playbook (store unavailable)",
        )

    if custom == 0:
        body = await hass.async_add_executor_job(store.active_body, key)
        return PlaybookSelection(
            body=body,
            key=key,
            method="route",
            detail=f"route playbook ({key})",
        )

    selected = await select_playbook_with_llm(
        llm,
        backend,
        user_text=user_text,
        history=history,
        catalog=catalog,
    )
    if selected is not None:
        return PlaybookSelection(
            body=selected.body,
            key=selected.route,
            method="llm",
            detail=f"LLM picked {selected.route}",
        )
    body = await hass.async_add_executor_job(store.active_body, key)
    return PlaybookSelection(
        body=body,
        key=key,
        method="route",
        detail=f"route fallback ({key})",
    )


def _selection_inputs(store: PlaybookStore) -> tuple[list[Playbook], int]:
    return store.list_enabled(), store.custom_count()


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

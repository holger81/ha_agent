"""SQLite persistence for durable user and household memory."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import DATA_KEY
from .models import MemoryEntry, MemoryScope, MergedMemory

PERSISTENT_MEMORY_STORE_KEY = "persistent_memory_stores"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_memory (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    route_scope TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_memory (
    agent_user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    route_scope TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (agent_user_id, key)
);
CREATE INDEX IF NOT EXISTS idx_user_memory_user
    ON user_memory(agent_user_id);
CREATE INDEX IF NOT EXISTS idx_system_memory_route
    ON system_memory(route_scope);
CREATE INDEX IF NOT EXISTS idx_user_memory_route
    ON user_memory(route_scope);
"""


def _encode_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _decode_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw


def _row_to_system(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        key=str(row["key"]),
        value=_decode_value(str(row["value_json"])),
        scope=MemoryScope.SYSTEM,
        agent_user_id=None,
        route_scope=row["route_scope"],
        updated_at=float(row["updated_at"]),
        created_at=float(row["created_at"]),
        notes=str(row["notes"] or ""),
    )


def _row_to_user(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        key=str(row["key"]),
        value=_decode_value(str(row["value_json"])),
        scope=MemoryScope.USER,
        agent_user_id=str(row["agent_user_id"]),
        route_scope=row["route_scope"],
        updated_at=float(row["updated_at"]),
        created_at=float(row["created_at"]),
        notes=str(row["notes"] or ""),
    )


class PersistentMemoryStore:
    """Per-config-entry durable memory database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the SQLite path for one config entry."""
        return (
            Path(hass.config.path(".storage"))
            / f"ha_agent_persistent_memory_{entry_id}.db"
        )

    def connect(self) -> None:
        """Open the database and ensure schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Persistent memory store is not connected")
        return self._conn

    def list_system(
        self,
        *,
        route_scope: str | None = None,
        key_prefix: str | None = None,
    ) -> list[MemoryEntry]:
        """List household (system) memory entries."""
        sql = "SELECT * FROM system_memory WHERE 1=1"
        params: list[Any] = []
        if route_scope is not None:
            sql += " AND (route_scope IS NULL OR route_scope = ?)"
            params.append(route_scope)
        if key_prefix:
            sql += " AND key LIKE ?"
            params.append(f"{key_prefix}%")
        sql += " ORDER BY key ASC"
        rows = self._db().execute(sql, params).fetchall()
        return [_row_to_system(row) for row in rows]

    def list_user(
        self,
        agent_user_id: str,
        *,
        route_scope: str | None = None,
        key_prefix: str | None = None,
    ) -> list[MemoryEntry]:
        """List user-bound memory entries."""
        sql = "SELECT * FROM user_memory WHERE agent_user_id = ?"
        params: list[Any] = [agent_user_id]
        if route_scope is not None:
            sql += " AND (route_scope IS NULL OR route_scope = ?)"
            params.append(route_scope)
        if key_prefix:
            sql += " AND key LIKE ?"
            params.append(f"{key_prefix}%")
        sql += " ORDER BY key ASC"
        rows = self._db().execute(sql, params).fetchall()
        return [_row_to_user(row) for row in rows]

    def get_system(self, key: str) -> MemoryEntry | None:
        """Return one system memory entry."""
        row = (
            self._db()
            .execute("SELECT * FROM system_memory WHERE key = ?", (key,))
            .fetchone()
        )
        return _row_to_system(row) if row else None

    def get_user(self, agent_user_id: str, key: str) -> MemoryEntry | None:
        """Return one user memory entry."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM user_memory WHERE agent_user_id = ? AND key = ?",
                (agent_user_id, key),
            )
            .fetchone()
        )
        return _row_to_user(row) if row else None

    def set_system(
        self,
        key: str,
        value: Any,
        *,
        route_scope: str | None = None,
        notes: str = "",
    ) -> MemoryEntry:
        """Insert or update a household memory entry."""
        now = time.time()
        existing = self.get_system(key)
        created = existing.created_at if existing else now
        self._db().execute(
            """
            INSERT INTO system_memory(key, value_json, route_scope, notes,
                                      created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                route_scope = excluded.route_scope,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                _normalize_key(key),
                _encode_value(value),
                route_scope,
                notes,
                created,
                now,
            ),
        )
        self._db().commit()
        entry = self.get_system(key)
        assert entry is not None
        return entry

    def set_user(
        self,
        agent_user_id: str,
        key: str,
        value: Any,
        *,
        route_scope: str | None = None,
        notes: str = "",
    ) -> MemoryEntry:
        """Insert or update a user-bound memory entry."""
        now = time.time()
        existing = self.get_user(agent_user_id, key)
        created = existing.created_at if existing else now
        self._db().execute(
            """
            INSERT INTO user_memory(agent_user_id, key, value_json, route_scope,
                                    notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_user_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                route_scope = excluded.route_scope,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                agent_user_id,
                _normalize_key(key),
                _encode_value(value),
                route_scope,
                notes,
                created,
                now,
            ),
        )
        self._db().commit()
        entry = self.get_user(agent_user_id, key)
        assert entry is not None
        return entry

    def delete_system(self, key: str) -> bool:
        """Delete a household memory entry. Return True if removed."""
        cur = self._db().execute("DELETE FROM system_memory WHERE key = ?", (key,))
        self._db().commit()
        return cur.rowcount > 0

    def delete_user(self, agent_user_id: str, key: str) -> bool:
        """Delete a user memory entry. Return True if removed."""
        cur = self._db().execute(
            "DELETE FROM user_memory WHERE agent_user_id = ? AND key = ?",
            (agent_user_id, key),
        )
        self._db().commit()
        return cur.rowcount > 0

    def delete_user_all(self, agent_user_id: str) -> int:
        """Delete all memory for one user. Return count removed."""
        cur = self._db().execute(
            "DELETE FROM user_memory WHERE agent_user_id = ?",
            (agent_user_id,),
        )
        self._db().commit()
        return int(cur.rowcount)

    def merge_for_turn(
        self,
        *,
        agent_user_id: str | None,
        include_user: bool,
        route: str | None = None,
    ) -> MergedMemory:
        """Merge system + optional user memory with user precedence."""
        system_entries = self.list_system(route_scope=route)
        values: dict[str, Any] = {}
        sources: dict[str, MemoryScope] = {}
        ordered: list[MemoryEntry] = []

        for entry in system_entries:
            values[entry.key] = entry.value
            sources[entry.key] = MemoryScope.SYSTEM
            ordered.append(entry)

        if include_user and agent_user_id:
            for entry in self.list_user(agent_user_id, route_scope=route):
                values[entry.key] = entry.value
                sources[entry.key] = MemoryScope.USER
                # Replace any prior system entry with same key in ordered list
                ordered = [e for e in ordered if e.key != entry.key]
                ordered.append(entry)

        return MergedMemory(values=values, sources=sources, entries=ordered)


def _normalize_key(key: str) -> str:
    cleaned = key.strip().lower().replace(" ", "_")
    if not cleaned:
        raise ValueError("memory key must be non-empty")
    return cleaned


def get_persistent_memory_store(
    hass: HomeAssistant, entry_id: str
) -> PersistentMemoryStore:
    """Return the persistent memory store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, PersistentMemoryStore] = domain_data.setdefault(
        PERSISTENT_MEMORY_STORE_KEY, {}
    )
    if entry_id not in stores:
        store = PersistentMemoryStore(
            PersistentMemoryStore.db_path_for_entry(hass, entry_id)
        )
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_persistent_memory_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove a persistent memory store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, PersistentMemoryStore] = domain_data.get(
        PERSISTENT_MEMORY_STORE_KEY, {}
    )
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

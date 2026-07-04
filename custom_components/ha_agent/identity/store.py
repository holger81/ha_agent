"""SQLite persistence for agent users."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import DATA_KEY
from .models import AgentUser, UserKind

IDENTITY_STORE_KEY = "identity_stores"
_REGISTERED_SEED_NAMES = ("Member 1", "Member 2", "Member 3", "Member 4")
_ASSIST_GUEST_NAME = "Voice (unidentified)"
_UNSET = object()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_users (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    display_name TEXT NOT NULL,
    ha_user_id TEXT UNIQUE,
    person_entity_id TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    merged_into TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_seen_at REAL
);
"""


def _row_to_user(row: sqlite3.Row) -> AgentUser:
    return AgentUser(
        id=str(row["id"]),
        kind=UserKind(str(row["kind"])),
        display_name=str(row["display_name"]),
        ha_user_id=row["ha_user_id"],
        person_entity_id=row["person_entity_id"],
        is_default=bool(row["is_default"]),
        merged_into=row["merged_into"],
        notes=str(row["notes"] or ""),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_seen_at=row["last_seen_at"],
    )


class IdentityStore:
    """SQLite store for registered and guest agent users."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the SQLite path for one config entry."""
        return Path(hass.config.path(".storage")) / f"ha_agent_identity_{entry_id}.db"

    def connect(self) -> None:
        """Open the database and ensure schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.ensure_seeded()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            msg = "Identity store is not connected"
            raise RuntimeError(msg)
        return self._conn

    def ensure_seeded(self) -> None:
        """Create four registered slots and one assist guest when empty."""
        count = self._db().execute("SELECT COUNT(*) FROM agent_users").fetchone()[0]
        if count:
            return
        now = time.time()
        users: list[tuple[Any, ...]] = []
        for index, name in enumerate(_REGISTERED_SEED_NAMES):
            users.append(
                (
                    str(uuid.uuid4()),
                    UserKind.REGISTERED.value,
                    name,
                    None,
                    None,
                    1 if index == 0 else 0,
                    None,
                    "",
                    now,
                    now,
                    None,
                )
            )
        users.append(
            (
                str(uuid.uuid4()),
                UserKind.GUEST.value,
                _ASSIST_GUEST_NAME,
                None,
                None,
                0,
                None,
                "Default Assist guest until voice identification is configured.",
                now,
                now,
                None,
            )
        )
        self._db().executemany(
            "INSERT INTO agent_users "
            "(id, kind, display_name, ha_user_id, person_entity_id, is_default, "
            "merged_into, notes, created_at, updated_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            users,
        )
        self._db().commit()

    def list_users(
        self,
        *,
        kind: UserKind | None = None,
        include_merged: bool = False,
    ) -> list[AgentUser]:
        """Return users ordered registered first, then guests by name."""
        clauses = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if not include_merged:
            clauses.append("merged_into IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db().execute(
            f"SELECT * FROM agent_users {where} "
            "ORDER BY CASE kind WHEN 'registered' THEN 0 ELSE 1 END, "
            "display_name COLLATE NOCASE",
            params,
        ).fetchall()
        return [_row_to_user(row) for row in rows]

    def get_user(self, user_id: str) -> AgentUser | None:
        """Return one user by id."""
        row = self._db().execute(
            "SELECT * FROM agent_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return _row_to_user(row) if row else None

    def get_by_ha_user_id(self, ha_user_id: str) -> AgentUser | None:
        """Return a registered user linked to a Home Assistant login."""
        row = self._db().execute(
            "SELECT * FROM agent_users WHERE ha_user_id = ? AND merged_into IS NULL",
            (ha_user_id,),
        ).fetchone()
        return _row_to_user(row) if row else None

    def get_default_registered(self) -> AgentUser | None:
        """Return the default registered user."""
        row = self._db().execute(
            "SELECT * FROM agent_users WHERE kind = ? AND is_default = 1 "
            "AND merged_into IS NULL LIMIT 1",
            (UserKind.REGISTERED.value,),
        ).fetchone()
        if row:
            return _row_to_user(row)
        row = self._db().execute(
            "SELECT * FROM agent_users WHERE kind = ? AND merged_into IS NULL "
            "ORDER BY display_name COLLATE NOCASE LIMIT 1",
            (UserKind.REGISTERED.value,),
        ).fetchone()
        return _row_to_user(row) if row else None

    def get_assist_guest(self) -> AgentUser:
        """Return the singleton Assist guest profile."""
        row = self._db().execute(
            "SELECT * FROM agent_users WHERE kind = ? AND display_name = ? "
            "AND merged_into IS NULL LIMIT 1",
            (UserKind.GUEST.value, _ASSIST_GUEST_NAME),
        ).fetchone()
        if row:
            return _row_to_user(row)
        now = time.time()
        user_id = str(uuid.uuid4())
        self._db().execute(
            "INSERT INTO agent_users "
            "(id, kind, display_name, ha_user_id, person_entity_id, is_default, "
            "merged_into, notes, created_at, updated_at, last_seen_at) "
            "VALUES (?, ?, ?, NULL, NULL, 0, NULL, ?, ?, ?, NULL)",
            (
                user_id,
                UserKind.GUEST.value,
                _ASSIST_GUEST_NAME,
                "Default Assist guest until voice identification is configured.",
                now,
                now,
            ),
        )
        self._db().commit()
        user = self.get_user(user_id)
        assert user is not None
        return user

    def create_guest(self, display_name: str, *, notes: str = "") -> AgentUser:
        """Create a new guest profile."""
        now = time.time()
        user_id = str(uuid.uuid4())
        self._db().execute(
            "INSERT INTO agent_users "
            "(id, kind, display_name, ha_user_id, person_entity_id, is_default, "
            "merged_into, notes, created_at, updated_at, last_seen_at) "
            "VALUES (?, ?, ?, NULL, NULL, 0, NULL, ?, ?, ?, NULL)",
            (user_id, UserKind.GUEST.value, display_name.strip(), notes, now, now),
        )
        self._db().commit()
        user = self.get_user(user_id)
        assert user is not None
        return user

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        ha_user_id: str | None | object = _UNSET,
        person_entity_id: str | None | object = _UNSET,
        is_default: bool | None = None,
        notes: str | None = None,
    ) -> AgentUser | None:
        """Update a user record."""
        user = self.get_user(user_id)
        if user is None or user.merged_into:
            return None
        fields: list[str] = []
        params: list[Any] = []
        if display_name is not None:
            fields.append("display_name = ?")
            params.append(display_name.strip())
        if ha_user_id is not _UNSET:
            fields.append("ha_user_id = ?")
            params.append(ha_user_id)
        if person_entity_id is not _UNSET:
            fields.append("person_entity_id = ?")
            params.append(person_entity_id)
        if notes is not None:
            fields.append("notes = ?")
            params.append(notes)
        if is_default is not None:
            if is_default and user.kind != UserKind.REGISTERED:
                return None
            if is_default:
                self._db().execute(
                    "UPDATE agent_users SET is_default = 0 WHERE kind = ?",
                    (UserKind.REGISTERED.value,),
                )
            fields.append("is_default = ?")
            params.append(int(is_default))
        if not fields:
            return user
        fields.append("updated_at = ?")
        params.append(time.time())
        params.append(user_id)
        self._db().execute(
            f"UPDATE agent_users SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        self._db().commit()
        return self.get_user(user_id)

    def touch_last_seen(self, user_id: str) -> None:
        """Update last_seen_at for a user."""
        now = time.time()
        self._db().execute(
            "UPDATE agent_users SET last_seen_at = ?, updated_at = ? WHERE id = ?",
            (now, now, user_id),
        )
        self._db().commit()


def get_identity_store(hass: HomeAssistant, entry_id: str) -> IdentityStore:
    """Return the identity store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, IdentityStore] = domain_data.setdefault(IDENTITY_STORE_KEY, {})
    if entry_id not in stores:
        store = IdentityStore(IdentityStore.db_path_for_entry(hass, entry_id))
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_identity_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove an identity store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, IdentityStore] = domain_data.get(IDENTITY_STORE_KEY, {})
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

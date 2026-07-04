"""SQLite persistence for agent users."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import DATA_KEY
from .embeddings import (
    merge_centroids,
    pack_embedding,
    unpack_embedding,
    update_centroid,
)
from .models import AgentUser, UserKind, VoiceProfile

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
CREATE TABLE IF NOT EXISTS voice_profiles (
    id TEXT PRIMARY KEY,
    agent_user_id TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    model TEXT,
    sample_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence REAL,
    centroid BLOB,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_seen_at REAL,
    FOREIGN KEY (agent_user_id) REFERENCES agent_users(id)
);
"""


def is_assist_guest_user(user: AgentUser) -> bool:
    """Return True for the singleton Assist fallback guest."""
    return user.kind == UserKind.GUEST and user.display_name == _ASSIST_GUEST_NAME


def _row_to_voice_profile(row: sqlite3.Row) -> VoiceProfile:
    return VoiceProfile(
        id=str(row["id"]),
        agent_user_id=str(row["agent_user_id"]),
        backend=str(row["backend"]),
        model=row["model"],
        sample_count=int(row["sample_count"]),
        avg_confidence=row["avg_confidence"],
        centroid=unpack_embedding(row["centroid"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_seen_at=row["last_seen_at"],
    )


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
        rows = (
            self._db()
            .execute(
                f"SELECT * FROM agent_users {where} "
                "ORDER BY CASE kind WHEN 'registered' THEN 0 ELSE 1 END, "
                "display_name COLLATE NOCASE",
                params,
            )
            .fetchall()
        )
        return [_row_to_user(row) for row in rows]

    def get_user(self, user_id: str) -> AgentUser | None:
        """Return one user by id."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM agent_users WHERE id = ?",
                (user_id,),
            )
            .fetchone()
        )
        return _row_to_user(row) if row else None

    def get_by_ha_user_id(self, ha_user_id: str) -> AgentUser | None:
        """Return a registered user linked to a Home Assistant login."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM agent_users WHERE ha_user_id = ? "
                "AND merged_into IS NULL",
                (ha_user_id,),
            )
            .fetchone()
        )
        return _row_to_user(row) if row else None

    def get_default_registered(self) -> AgentUser | None:
        """Return the default registered user."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM agent_users WHERE kind = ? AND is_default = 1 "
                "AND merged_into IS NULL LIMIT 1",
                (UserKind.REGISTERED.value,),
            )
            .fetchone()
        )
        if row:
            return _row_to_user(row)
        row = (
            self._db()
            .execute(
                "SELECT * FROM agent_users WHERE kind = ? AND merged_into IS NULL "
                "ORDER BY display_name COLLATE NOCASE LIMIT 1",
                (UserKind.REGISTERED.value,),
            )
            .fetchone()
        )
        return _row_to_user(row) if row else None

    def get_assist_guest(self) -> AgentUser:
        """Return the singleton Assist guest profile."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM agent_users WHERE kind = ? AND display_name = ? "
                "AND merged_into IS NULL LIMIT 1",
                (UserKind.GUEST.value, _ASSIST_GUEST_NAME),
            )
            .fetchone()
        )
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

    def next_voice_guest_name(self) -> str:
        """Return the next display name for a voice-clustered guest."""
        rows = (
            self._db()
            .execute(
                "SELECT COUNT(*) FROM agent_users "
                "WHERE kind = ? AND display_name != ? AND merged_into IS NULL",
                (UserKind.GUEST.value, _ASSIST_GUEST_NAME),
            )
            .fetchone()
        )
        count = int(rows[0]) if rows else 0
        return f"Guest {count + 1}"

    def list_voice_profiles(self) -> list[VoiceProfile]:
        """Return voice profiles excluding the Assist fallback guest."""
        rows = (
            self._db()
            .execute(
                "SELECT vp.* FROM voice_profiles vp "
                "JOIN agent_users au ON au.id = vp.agent_user_id "
                "WHERE au.merged_into IS NULL AND au.display_name != ? "
                "ORDER BY vp.updated_at DESC",
                (_ASSIST_GUEST_NAME,),
            )
            .fetchall()
        )
        return [_row_to_voice_profile(row) for row in rows]

    def get_voice_profile_for_user(self, agent_user_id: str) -> VoiceProfile | None:
        """Return the voice profile linked to one agent user."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM voice_profiles WHERE agent_user_id = ?",
                (agent_user_id,),
            )
            .fetchone()
        )
        return _row_to_voice_profile(row) if row else None

    def create_voice_profile(
        self,
        *,
        profile_id: str,
        agent_user_id: str,
        backend: str,
        model: str | None,
        centroid: list[float],
        match_confidence: float | None,
    ) -> VoiceProfile:
        """Create a voice profile with an initial centroid."""
        now = time.time()
        self._db().execute(
            "INSERT INTO voice_profiles "
            "(id, agent_user_id, backend, model, sample_count, avg_confidence, "
            "centroid, created_at, updated_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
            (
                profile_id,
                agent_user_id,
                backend,
                model,
                match_confidence,
                pack_embedding(centroid),
                now,
                now,
                now,
            ),
        )
        self._db().commit()
        profile = self.get_voice_profile_for_user(agent_user_id)
        assert profile is not None
        return profile

    def enroll_voice_sample(
        self,
        agent_user_id: str,
        *,
        embedding: list[float],
        backend: str,
        model: str | None,
        match_confidence: float,
    ) -> VoiceProfile:
        """Add or update a registered member voice profile from enrollment."""
        profile = self.get_voice_profile_for_user(agent_user_id)
        if profile is None:
            return self.create_voice_profile(
                profile_id=str(uuid.uuid4()),
                agent_user_id=agent_user_id,
                backend=backend,
                model=model,
                centroid=embedding,
                match_confidence=match_confidence,
            )
        self.update_voice_profile_centroid(
            profile.id,
            embedding=embedding,
            match_confidence=match_confidence,
            model=model,
        )
        updated = self.get_voice_profile_for_user(agent_user_id)
        assert updated is not None
        return updated

    def update_voice_profile_centroid(
        self,
        profile_id: str,
        *,
        embedding: list[float],
        match_confidence: float,
        model: str | None = None,
    ) -> None:
        """Update a profile centroid with a new embedding sample."""
        row = (
            self._db()
            .execute(
                "SELECT * FROM voice_profiles WHERE id = ?",
                (profile_id,),
            )
            .fetchone()
        )
        if row is None:
            return
        profile = _row_to_voice_profile(row)
        if profile.centroid is None:
            updated = list(embedding)
            sample_count = 1
        else:
            updated = update_centroid(
                profile.centroid,
                profile.sample_count,
                embedding,
            )
            sample_count = profile.sample_count + 1
        if profile.avg_confidence is None:
            avg_confidence = match_confidence
        else:
            avg_confidence = (
                (profile.avg_confidence * profile.sample_count) + match_confidence
            ) / sample_count
        now = time.time()
        self._db().execute(
            "UPDATE voice_profiles SET sample_count = ?, avg_confidence = ?, "
            "centroid = ?, model = COALESCE(?, model), updated_at = ?, "
            "last_seen_at = ? WHERE id = ?",
            (
                sample_count,
                avg_confidence,
                pack_embedding(updated),
                model,
                now,
                now,
                profile_id,
            ),
        )
        self._db().commit()

    def _validate_promotable_guest(self, user_id: str) -> AgentUser:
        user = self.get_user(user_id)
        if user is None or user.merged_into:
            msg = "Guest not found"
            raise ValueError(msg)
        if user.kind != UserKind.GUEST or is_assist_guest_user(user):
            msg = "User is not a promotable guest"
            raise ValueError(msg)
        return user

    def _merge_voice_profiles_into(
        self,
        target: VoiceProfile,
        source: VoiceProfile,
    ) -> None:
        if source.centroid is None:
            return
        if target.centroid is None:
            updated = list(source.centroid)
            sample_count = max(source.sample_count, 1)
            avg_confidence = source.avg_confidence
        else:
            sample_count = target.sample_count + source.sample_count
            updated = merge_centroids(
                target.centroid,
                target.sample_count,
                source.centroid,
                source.sample_count,
            )
            if target.avg_confidence is not None and source.avg_confidence is not None:
                avg_confidence = (
                    (target.avg_confidence * target.sample_count)
                    + (source.avg_confidence * source.sample_count)
                ) / sample_count
            else:
                avg_confidence = target.avg_confidence or source.avg_confidence
        now = time.time()
        self._db().execute(
            "UPDATE voice_profiles SET sample_count = ?, avg_confidence = ?, "
            "centroid = ?, model = COALESCE(?, model), updated_at = ?, "
            "last_seen_at = ? WHERE id = ?",
            (
                sample_count,
                avg_confidence,
                pack_embedding(updated),
                source.model or target.model,
                now,
                now,
                target.id,
            ),
        )

    def promote_guest(
        self,
        guest_id: str,
        registered_id: str,
        *,
        display_name: str | None = None,
    ) -> AgentUser:
        """Attach a guest voice profile to a registered member."""
        self._validate_promotable_guest(guest_id)
        registered = self.get_user(registered_id)
        if (
            registered is None
            or registered.merged_into
            or registered.kind != UserKind.REGISTERED
        ):
            msg = "Registered user not found"
            raise ValueError(msg)

        guest_profile = self.get_voice_profile_for_user(guest_id)
        reg_profile = self.get_voice_profile_for_user(registered_id)
        if guest_profile is not None:
            if reg_profile is not None:
                self._merge_voice_profiles_into(reg_profile, guest_profile)
                self._db().execute(
                    "DELETE FROM voice_profiles WHERE id = ?",
                    (guest_profile.id,),
                )
            else:
                self._db().execute(
                    "UPDATE voice_profiles SET agent_user_id = ?, updated_at = ? "
                    "WHERE id = ?",
                    (registered_id, time.time(), guest_profile.id),
                )

        if display_name:
            self.update_user(registered_id, display_name=display_name.strip())

        now = time.time()
        self._db().execute(
            "UPDATE agent_users SET merged_into = ?, updated_at = ?, notes = ? "
            "WHERE id = ?",
            (
                registered_id,
                now,
                f"Promoted into {registered.display_name}",
                guest_id,
            ),
        )
        self._db().commit()
        updated = self.get_user(registered_id)
        assert updated is not None
        return updated

    def merge_guests(
        self,
        guest_ids: list[str],
        *,
        survivor_id: str | None = None,
    ) -> AgentUser:
        """Merge multiple guest profiles into one survivor guest."""
        unique_ids = list(dict.fromkeys(guest_ids))
        if len(unique_ids) < 2:
            msg = "At least two guests are required"
            raise ValueError(msg)

        users = [self._validate_promotable_guest(guest_id) for guest_id in unique_ids]
        if survivor_id is None:
            survivor = min(users, key=lambda user: user.created_at)
            survivor_id = survivor.id
        elif survivor_id not in unique_ids:
            msg = "Survivor must be included in guest_ids"
            raise ValueError(msg)
        else:
            survivor = self.get_user(survivor_id)
            if survivor is None:
                msg = "Survivor guest not found"
                raise ValueError(msg)

        survivor_profile = self.get_voice_profile_for_user(survivor_id)
        for guest_id in unique_ids:
            if guest_id == survivor_id:
                continue
            source_profile = self.get_voice_profile_for_user(guest_id)
            if source_profile is None:
                continue
            if survivor_profile is None:
                self._db().execute(
                    "UPDATE voice_profiles SET agent_user_id = ?, updated_at = ? "
                    "WHERE id = ?",
                    (survivor_id, time.time(), source_profile.id),
                )
                survivor_profile = self.get_voice_profile_for_user(survivor_id)
                continue
            self._merge_voice_profiles_into(survivor_profile, source_profile)
            self._db().execute(
                "DELETE FROM voice_profiles WHERE id = ?",
                (source_profile.id,),
            )
            survivor_profile = self.get_voice_profile_for_user(survivor_id)

        now = time.time()
        for guest_id in unique_ids:
            if guest_id == survivor_id:
                continue
            self._db().execute(
                "UPDATE agent_users SET merged_into = ?, updated_at = ?, notes = ? "
                "WHERE id = ?",
                (
                    survivor_id,
                    now,
                    f"Merged into {survivor.display_name}",
                    guest_id,
                ),
            )
        self._db().commit()
        merged = self.get_user(survivor_id)
        assert merged is not None
        return merged


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

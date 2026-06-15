"""SQLite + FTS5 persistence for learned skills."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import DATA_KEY, LOGGER
from .models import Skill, SkillIndexRow

SKILLS_STORE_KEY = "skill_stores"
_IMPROVEMENT_COOLDOWN_SECONDS = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    triggers_json TEXT NOT NULL,
    body TEXT NOT NULL,
    tool_steps_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_used_at REAL,
    use_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    last_improved_at REAL,
    last_evaluation_at REAL,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    skill_id UNINDEXED,
    title,
    description,
    triggers,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def _slugify(title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (base[:60] or "skill").strip("-")


def _row_to_skill(row: sqlite3.Row) -> Skill:
    return Skill(
        id=row["id"],
        slug=row["slug"],
        title=row["title"],
        description=row["description"],
        triggers=json.loads(row["triggers_json"]),
        body=row["body"],
        tool_steps=json.loads(row["tool_steps_json"]),
        enabled=bool(row["enabled"]),
        created_at=float(row["created_at"]),
        last_used_at=row["last_used_at"],
        use_count=int(row["use_count"]),
        success_count=int(row["success_count"]),
        last_improved_at=row["last_improved_at"],
        last_evaluation_at=row["last_evaluation_at"],
        version=int(row["version"]),
    )


class SkillStore:
    """Per-config-entry skill database with FTS5 search."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the on-disk path for an entry's skill database."""
        storage = Path(hass.config.path(".storage"))
        return storage / f"ha_agent_skills_{entry_id}.db"

    def connect(self) -> None:
        """Open the database and ensure schema exists."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
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

    def _sync_fts(self, skill: Skill) -> None:
        conn = self._connection()
        conn.execute("DELETE FROM skills_fts WHERE skill_id = ?", (skill.id,))
        conn.execute(
            "INSERT INTO skills_fts(skill_id, title, description, triggers) "
            "VALUES (?, ?, ?, ?)",
            (
                skill.id,
                skill.title,
                skill.description,
                " ".join(skill.triggers),
            ),
        )

    def _unique_slug(self, base_slug: str, *, exclude_id: str | None = None) -> str:
        conn = self._connection()
        slug = base_slug
        counter = 2
        while True:
            row = conn.execute(
                "SELECT id FROM skills WHERE slug = ?",
                (slug,),
            ).fetchone()
            if row is None or (exclude_id and row["id"] == exclude_id):
                return slug
            slug = f"{base_slug[:56]}-{counter}"
            counter += 1

    def insert_skill(
        self,
        *,
        title: str,
        description: str,
        triggers: list[str],
        body: str,
        tool_steps: list[dict[str, Any]],
        enabled: bool = True,
        skill_id: str | None = None,
    ) -> Skill:
        """Insert a new skill and index it in FTS."""
        now = time.time()
        skill = Skill(
            id=skill_id or str(uuid.uuid4()),
            slug=self._unique_slug(_slugify(title)),
            title=title.strip(),
            description=description.strip()[:1024],
            triggers=triggers,
            body=body.strip(),
            tool_steps=tool_steps,
            enabled=enabled,
            created_at=now,
        )
        conn = self._connection()
        conn.execute(
            "INSERT INTO skills "
            "(id, slug, title, description, triggers_json, body, tool_steps_json, "
            "enabled, created_at, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                skill.id,
                skill.slug,
                skill.title,
                skill.description,
                json.dumps(skill.triggers),
                skill.body,
                json.dumps(skill.tool_steps),
                int(skill.enabled),
                skill.created_at,
                skill.version,
            ),
        )
        self._sync_fts(skill)
        conn.commit()
        return skill

    def update_skill(self, skill: Skill) -> Skill:
        """Update an existing skill and refresh FTS."""
        conn = self._connection()
        conn.execute(
            "UPDATE skills SET "
            "slug = ?, title = ?, description = ?, triggers_json = ?, body = ?, "
            "tool_steps_json = ?, enabled = ?, last_used_at = ?, use_count = ?, "
            "success_count = ?, last_improved_at = ?, last_evaluation_at = ?, "
            "version = ? "
            "WHERE id = ?",
            (
                skill.slug,
                skill.title,
                skill.description,
                json.dumps(skill.triggers),
                skill.body,
                json.dumps(skill.tool_steps),
                int(skill.enabled),
                skill.last_used_at,
                skill.use_count,
                skill.success_count,
                skill.last_improved_at,
                skill.last_evaluation_at,
                skill.version,
                skill.id,
            ),
        )
        self._sync_fts(skill)
        conn.commit()
        return skill

    def get_skill(self, skill_id: str) -> Skill | None:
        """Return a skill by id."""
        row = self._connection().execute(
            "SELECT * FROM skills WHERE id = ?",
            (skill_id,),
        ).fetchone()
        return _row_to_skill(row) if row else None

    def get_skill_by_slug(self, slug: str) -> Skill | None:
        """Return a skill by slug."""
        row = self._connection().execute(
            "SELECT * FROM skills WHERE slug = ?",
            (slug,),
        ).fetchone()
        return _row_to_skill(row) if row else None

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill and its FTS row."""
        conn = self._connection()
        conn.execute("DELETE FROM skills_fts WHERE skill_id = ?", (skill_id,))
        cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
        conn.commit()
        return cursor.rowcount > 0

    def set_enabled(self, skill_id: str, enabled: bool) -> Skill | None:
        """Enable or disable a skill."""
        skill = self.get_skill(skill_id)
        if skill is None:
            return None
        skill.enabled = enabled
        return self.update_skill(skill)

    def count_skills(self, *, enabled_only: bool = False) -> int:
        """Return total skill count."""
        if enabled_only:
            row = self._connection().execute(
                "SELECT COUNT(*) AS c FROM skills WHERE enabled = 1",
            ).fetchone()
        else:
            row = self._connection().execute(
                "SELECT COUNT(*) AS c FROM skills",
            ).fetchone()
        return int(row["c"]) if row else 0

    def list_recent(self, *, limit: int = 10) -> list[Skill]:
        """Return recently used skills."""
        rows = self._connection().execute(
            "SELECT * FROM skills ORDER BY "
            "COALESCE(last_used_at, created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_skill(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        enabled_only: bool = True,
    ) -> list[SkillIndexRow]:
        """FTS search over skill metadata."""
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []
        conn = self._connection()
        enabled_clause = "AND s.enabled = 1" if enabled_only else ""
        try:
            rows = conn.execute(
                f"""
                SELECT s.id, s.slug, s.title, s.description, bm25(skills_fts) AS rank
                FROM skills_fts
                JOIN skills s ON s.id = skills_fts.skill_id
                WHERE skills_fts MATCH ? {enabled_clause}
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError as err:
            LOGGER.debug("Skill FTS query failed for %r: %s", fts_query, err)
            return []
        return [
            SkillIndexRow(
                id=row["id"],
                slug=row["slug"],
                title=row["title"],
                description=row["description"],
                rank=float(row["rank"]),
            )
            for row in rows
        ]

    def load_skills_by_ids(self, skill_ids: list[str]) -> list[Skill]:
        """Load full skill records for the given ids."""
        if not skill_ids:
            return []
        placeholders = ",".join("?" for _ in skill_ids)
        rows = self._connection().execute(
            f"SELECT * FROM skills WHERE id IN ({placeholders})",
            skill_ids,
        ).fetchall()
        by_id = {row["id"]: _row_to_skill(row) for row in rows}
        return [by_id[sid] for sid in skill_ids if sid in by_id]

    def record_use(
        self,
        skill_id: str,
        *,
        succeeded: bool,
    ) -> Skill | None:
        """Increment usage counters for a skill."""
        skill = self.get_skill(skill_id)
        if skill is None:
            return None
        skill.use_count += 1
        if succeeded:
            skill.success_count += 1
        skill.last_used_at = time.time()
        return self.update_skill(skill)

    def can_improve(self, skill_id: str) -> bool:
        """Return True when the hourly improvement cooldown has elapsed."""
        skill = self.get_skill(skill_id)
        if skill is None:
            return False
        if skill.last_improved_at is None:
            return True
        return (time.time() - skill.last_improved_at) >= _IMPROVEMENT_COOLDOWN_SECONDS

    def find_duplicate(
        self,
        triggers: list[str],
        *,
        rank_threshold: float = 100.0,
    ) -> Skill | None:
        """Return an existing skill that closely matches trigger phrases."""
        combined = " ".join(triggers)
        matches = self.search(combined, limit=1, enabled_only=False)
        if not matches:
            return None
        if matches[0].rank > rank_threshold:
            return None
        return self.get_skill(matches[0].id)


def _build_fts_query(text: str) -> str:
    """Build an FTS5 OR query from user text."""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    ]
    if not tokens:
        tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]
    if not tokens:
        return ""
    escaped = []
    for token in tokens[:12]:
        safe = token.replace('"', '""')
        escaped.append(f'"{safe}"')
    return " OR ".join(escaped)


_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "your",
        "you",
        "are",
        "was",
        "were",
        "can",
        "please",
        "could",
        "would",
        "about",
        "into",
        "them",
        "they",
        "their",
        "what",
        "when",
        "where",
        "which",
        "how",
        "all",
        "any",
        "some",
    }
)


def get_skill_store(hass: HomeAssistant, entry_id: str) -> SkillStore:
    """Return the skill store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, SkillStore] = domain_data.setdefault(SKILLS_STORE_KEY, {})
    if entry_id not in stores:
        store = SkillStore(SkillStore.db_path_for_entry(hass, entry_id))
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_skill_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove a skill store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, SkillStore] = domain_data.get(SKILLS_STORE_KEY, {})
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

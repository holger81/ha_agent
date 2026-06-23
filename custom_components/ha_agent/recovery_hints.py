"""Editable recovery hints.

Recovery hints are short directives appended to a failed tool result to help
the model change strategy. The default rules ship with the integration in
``loop_policy.enrich_tool_output`` but are fully editable, addable, deletable,
and resettable from the console UI and persisted per config entry in SQLite.

Each rule fires on a failed tool result when its (optional) tool-name substring
and (optional) error-text pattern both match. When the store is unavailable the
sync ``enrich_tool_output`` falls back to its deterministic shipped logic.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from homeassistant.core import HomeAssistant

from .const import DATA_KEY, LOGGER

RECOVERY_HINTS_STORE_KEY = "recovery_hint_stores"

# Built-in recovery hints seeded on first connect. ``error_pattern`` is a
# case-insensitive regex searched in the lowercased tool error; an empty
# pattern matches any error. ``tool_substring`` is matched (lowercased) against
# the tool name; an empty substring matches any tool.
DEFAULT_RECOVERY_HINTS: list[dict[str, object]] = [
    {
        "rule_id": "email_status",
        "title": "Email: prefer mailbox_status + unread search",
        "tool_substring": "mail",
        "error_pattern": "",
        "body": (
            "Prefer mailbox_status for unseen count, then search_messages with "
            "unread_only=true before fetching individual messages."
        ),
        "priority": 0,
    },
    {
        "rule_id": "email_large_inbox",
        "title": "Email: large inbox",
        "tool_substring": "mail",
        "error_pattern": (
            r"\b(too many|too large|very large|large number|limit|timeout|"
            r"overflow)\b"
        ),
        "body": (
            "Search unread messages only with a small limit (e.g. 10) via "
            "mail_mcp_imap_search_messages instead of listing the full inbox."
        ),
        "priority": 1,
    },
    {
        "rule_id": "news_curate",
        "title": "News: call news_curate first",
        "tool_substring": "news",
        "error_pattern": "",
        "body": (
            "For headlines, call mcp_news__news_curate directly with no "
            "arguments ({}) before trying other news tools."
        ),
        "priority": 2,
    },
    {
        "rule_id": "mcp_down",
        "title": "MCP offline",
        "tool_substring": "",
        "error_pattern": (
            r"\b(unreachable|connection refused|timed out|timeout|502|503|504)\b"
        ),
        "body": (
            "MCP may be offline. Tell the user to check MCP proxy connectivity "
            "in HA Agent Settings."
        ),
        "priority": 3,
    },
    {
        "rule_id": "ha_call_service_domain",
        "title": "ha_call_service missing domain",
        "tool_substring": "ha_call_service",
        "error_pattern": "domain",
        "body": (
            "Include domain, service, and entity_id in ha_call_service "
            "arguments. Derive domain from the entity_id prefix "
            "(light.example -> light)."
        ),
        "priority": 4,
    },
    {
        "rule_id": "ha_search_entities_unavailable",
        "title": "ha_search_entities unavailable",
        "tool_substring": "search_entities",
        "error_pattern": r"unknown tool|not found|unavailable",
        "body": (
            "home_assistant__ha_search_entities is unavailable. Skip entity "
            "search. Use an EXPOSED ENTITIES shortcut with "
            "home_assistant__ha_call_service (domain, service, entity_id) "
            "instead."
        ),
        "priority": 5,
    },
]

_DEFAULT_BY_ID = {rule["rule_id"]: rule for rule in DEFAULT_RECOVERY_HINTS}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recovery_hints (
    rule_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    tool_substring TEXT NOT NULL DEFAULT '',
    error_pattern TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    is_builtin INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class RecoveryHint:
    """An editable recovery-hint rule (built-in or custom)."""

    rule_id: str
    title: str
    tool_substring: str = ""
    error_pattern: str = ""
    body: str = ""
    enabled: bool = True
    is_builtin: bool = True
    priority: int = 0
    updated_at: float = 0.0
    is_default: bool = True


def _is_default(rule: RecoveryHint) -> bool:
    default = _DEFAULT_BY_ID.get(rule.rule_id)
    if not default:
        return False
    return (
        rule.tool_substring.strip() == str(default["tool_substring"]).strip()
        and rule.error_pattern.strip() == str(default["error_pattern"]).strip()
        and rule.body.strip() == str(default["body"]).strip()
    )


def _row_to_hint(row: sqlite3.Row) -> RecoveryHint:
    hint = RecoveryHint(
        rule_id=row["rule_id"],
        title=row["title"],
        tool_substring=row["tool_substring"],
        error_pattern=row["error_pattern"],
        body=row["body"],
        enabled=bool(row["enabled"]),
        is_builtin=bool(row["is_builtin"]),
        priority=int(row["priority"]),
        updated_at=float(row["updated_at"]),
    )
    hint.is_default = hint.is_builtin and _is_default(hint)
    return hint


class RecoveryHintStore:
    """Per-config-entry editable recovery-hint database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def db_path_for_entry(hass: HomeAssistant, entry_id: str) -> Path:
        """Return the on-disk path for an entry's recovery-hint database."""
        storage = Path(hass.config.path(".storage"))
        return storage / f"ha_agent_recovery_hints_{entry_id}.db"

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
        for default in DEFAULT_RECOVERY_HINTS:
            conn.execute(
                "INSERT OR IGNORE INTO recovery_hints "
                "(rule_id, title, tool_substring, error_pattern, body, "
                "enabled, is_builtin, priority, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?)",
                (
                    default["rule_id"],
                    default["title"],
                    default["tool_substring"],
                    default["error_pattern"],
                    default["body"],
                    default["priority"],
                    now,
                ),
            )

    def list_hints(self) -> list[RecoveryHint]:
        """Return built-in hints in canonical order then custom rules."""
        rows = self._connection().execute(
            "SELECT * FROM recovery_hints",
        ).fetchall()
        hints = [_row_to_hint(row) for row in rows]
        hints.sort(key=lambda h: (0 if h.is_builtin else 1, h.priority, h.updated_at))
        return hints

    def list_enabled(self) -> list[RecoveryHint]:
        """Return all enabled hints in runtime order."""
        return [hint for hint in self.list_hints() if hint.enabled]

    def custom_count(self) -> int:
        """Return the number of user-added custom hints."""
        row = self._connection().execute(
            "SELECT COUNT(*) AS c FROM recovery_hints WHERE is_builtin = 0",
        ).fetchone()
        return int(row["c"]) if row else 0

    def get_hint(self, rule_id: str) -> RecoveryHint | None:
        """Return one hint by id."""
        row = self._connection().execute(
            "SELECT * FROM recovery_hints WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
        return _row_to_hint(row) if row else None

    def create_hint(
        self,
        *,
        title: str,
        body: str,
        tool_substring: str = "",
        error_pattern: str = "",
        enabled: bool = True,
    ) -> RecoveryHint:
        """Create a custom (non-built-in) recovery-hint rule."""
        now = time.time()
        hint = RecoveryHint(
            rule_id=f"custom-{uuid.uuid4().hex[:12]}",
            title=title.strip() or "Custom recovery hint",
            tool_substring=tool_substring.strip(),
            error_pattern=error_pattern.strip(),
            body=body.strip(),
            enabled=enabled,
            is_builtin=False,
            priority=1000,
            updated_at=now,
            is_default=False,
        )
        conn = self._connection()
        conn.execute(
            "INSERT INTO recovery_hints "
            "(rule_id, title, tool_substring, error_pattern, body, "
            "enabled, is_builtin, priority, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                hint.rule_id,
                hint.title,
                hint.tool_substring,
                hint.error_pattern,
                hint.body,
                int(hint.enabled),
                hint.priority,
                hint.updated_at,
            ),
        )
        conn.commit()
        return hint

    def delete_hint(self, rule_id: str) -> bool:
        """Delete a custom recovery-hint rule. Built-ins cannot be deleted."""
        hint = self.get_hint(rule_id)
        if hint is None or hint.is_builtin:
            return False
        conn = self._connection()
        conn.execute("DELETE FROM recovery_hints WHERE rule_id = ?", (rule_id,))
        conn.commit()
        return True

    def update_hint(
        self,
        rule_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        tool_substring: str | None = None,
        error_pattern: str | None = None,
        enabled: bool | None = None,
    ) -> RecoveryHint | None:
        """Update an existing recovery-hint's editable fields."""
        hint = self.get_hint(rule_id)
        if hint is None:
            return None
        if title is not None:
            hint.title = title.strip() or hint.title
        if body is not None:
            hint.body = body.strip()
        if tool_substring is not None:
            hint.tool_substring = tool_substring.strip()
        if error_pattern is not None:
            hint.error_pattern = error_pattern.strip()
        if enabled is not None:
            hint.enabled = enabled
        hint.updated_at = time.time()
        conn = self._connection()
        conn.execute(
            "UPDATE recovery_hints SET title = ?, body = ?, tool_substring = ?, "
            "error_pattern = ?, enabled = ?, updated_at = ? WHERE rule_id = ?",
            (
                hint.title,
                hint.body,
                hint.tool_substring,
                hint.error_pattern,
                int(hint.enabled),
                hint.updated_at,
                hint.rule_id,
            ),
        )
        conn.commit()
        hint.is_default = hint.is_builtin and _is_default(hint)
        return hint

    def reset_hint(self, rule_id: str) -> RecoveryHint | None:
        """Restore a built-in hint to its shipped default and enable it."""
        default = _DEFAULT_BY_ID.get(rule_id)
        if default is None:
            return None
        return self.update_hint(
            rule_id,
            title=str(default["title"]),
            body=str(default["body"]),
            tool_substring=str(default["tool_substring"]),
            error_pattern=str(default["error_pattern"]),
            enabled=True,
        )


async def async_recovery_hints(
    hass: HomeAssistant,
    entry_id: str,
) -> list[RecoveryHint] | None:
    """Return enabled recovery-hint rules for this turn.

    Returns ``None`` (use the shipped hardcoded logic) when the store cannot be
    read so a storage problem never breaks a turn.
    """
    try:
        store = get_recovery_hint_store(hass, entry_id)
        return await hass.async_add_executor_job(store.list_enabled)
    except Exception as err:
        LOGGER.debug("Recovery hint store unavailable, using defaults: %s", err)
        return None


def get_recovery_hint_store(
    hass: HomeAssistant,
    entry_id: str,
) -> RecoveryHintStore:
    """Return the recovery-hint store for a config entry."""
    domain_data = hass.data.setdefault(DATA_KEY, {})
    stores: dict[str, RecoveryHintStore] = domain_data.setdefault(
        RECOVERY_HINTS_STORE_KEY, {}
    )
    if entry_id not in stores:
        store = RecoveryHintStore(
            RecoveryHintStore.db_path_for_entry(hass, entry_id)
        )
        store.connect()
        stores[entry_id] = store
    return stores[entry_id]


def close_recovery_hint_store(hass: HomeAssistant, entry_id: str) -> None:
    """Close and remove a recovery-hint store on unload."""
    domain_data = hass.data.get(DATA_KEY, {})
    stores: dict[str, RecoveryHintStore] = domain_data.get(
        RECOVERY_HINTS_STORE_KEY, {}
    )
    store = stores.pop(entry_id, None)
    if store is not None:
        store.close()

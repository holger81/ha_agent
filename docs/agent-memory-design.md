# Agent memory — design notes

Durable preferences and household facts, separate from conversation history
(`memory.py`) and workflow skills.

**Status:** implemented (Phase 10) in `custom_components/ha_agent/persistent_memory/`.

**Prerequisite:** [Phase 9 identity](agent-identity-design.md) — user-bound
memory needs a resolved agent user on every turn (voice ID or HA login).

## Not the same as existing `memory.py`

`memory.py` stores **conversation turns** (user/assistant messages per thread). The
feature described here is **durable knowledge**: preferences, defaults, and household
facts that should influence future turns without re-explaining.

## Distinct from skills

| Layer | Purpose | Example |
|-------|---------|---------|
| **Skills** | Repeatable multi-step workflows | Check unread email → search → read |
| **Memory** | Defaults, facts, mappings | Prefer `local: true` on `news_curate`; dining room light = `light.dining_room_lights_ceiling` |

Do **not** force skill save for preference-like requests. Route those to memory.
Manual skill save remains for genuine procedures (optionally with relaxed observer on
explicit user request).

## Two memory scopes

### 1. User-bound memory

Per-user preferences and overrides (`user_memory` table, keyed by `agent_user_id`).

Examples:

- Preferred news style (`news.digest_scope`)
- Default email mailbox (`email.default_mailbox`)
- Personal entity aliases (`entity.alias.*`)

Guests and low-confidence voice matches receive **system memory only**.

### 2. System / household memory

Shared facts (`system_memory` table) that apply to everyone unless a user-bound entry
overrides them.

## Precedence

```text
user-bound memory  >  system memory  >  shipped defaults (context.py, playbooks)
```

## Injection

On each turn after identity + route resolve, merge applicable keys into a compact
`DURABLE MEMORY` block on the system message (`persistent_memory/inject.py`).
Keys are filtered by route prefixes (`news.`, `email.`, `entity.alias.`, …).

Slot defaults from memory fill empty skill bindings (`apply_memory_defaults_to_slots`).

## Intent routing

Rules-first in `persistent_memory/intent.py`:

1. “save … as a skill” → existing skill path
2. remember / prefer / always / forget → memory write/delete short-circuit
3. Ambiguous workflow+remember → clarify question

Deterministic extractors cover local/national news, mailbox, and entity aliases.
Console CRUD: `ha_agent/memory/list|set|delete` (Settings = household, Users = per member).

## Related code

- `persistent_memory/store.py` — SQLite CRUD
- `persistent_memory/runtime.py` — turn short-circuit + slot→memory writes
- `agent.py` — load/inject after identity; skip skill observer for preference turns
- `skills/selection.py` — stronger FTS bar so weak matches do not pin plans

# Agent memory — design notes (future implementation)

Captured from product discussion (2026-07). Use this when implementing persistent
memory beyond today's per-`conversation_id` chat history (`memory.py`).

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

## Two memory scopes (minimum)

### 1. User-bound memory

Per-user preferences and overrides. Different household members can have different
values for the same key.

Examples:

- Preferred news style (local vs national briefing)
- Default email mailbox
- Personal phrasing or habits

Requires a stable **user identity** at turn time (Assist user id, HA person entity,
console profile, etc. — TBD at implementation).

### 2. System / household memory

Shared facts and procedures that apply to everyone unless a user-bound entry
explicitly overrides them.

Examples:

- **Procedures:** “Local news” always means `mcp_news__news_curate` with
  `local: true` / `digest_scope: local` (same workflow for all users)
- **Entity mappings:** “Dining room light” → `light.dining_room_lights_ceiling`
- **Location context:** Home is in the Bay Area (for disambiguation)

## Precedence

```text
user-bound memory  >  system memory  >  shipped defaults (context.py, playbooks)
```

User overrides are explicit (“for me, use …”) or set via a profile UI. System
memory holds the household baseline.

## Injection (sketch)

On each turn, merge applicable memory into LLM context (system or tool context),
scoped by route/tool when possible:

- News route → apply news-related system + user memory
- Action route → entity aliases / area mappings
- Email route → mailbox defaults (user first, then system)

Keep injection compact; prefer structured keys over long prose.

## Intent routing (sketch)

When the user says “remember …”, “I prefer …”, “always use …”, “save this”:

1. **Workflow** (multi-tool procedure) → skill path (existing learning)
2. **Default / fact / single-arg preference** → memory (user or system — ask or infer scope)
3. Ambiguous → clarify: “Save as a reusable workflow, or remember as a default?”

## Open questions (for implementation)

- User identity source in Assist vs console vs automations
- Storage: SQLite table(s) per config entry; separate `user_memory` and `system_memory`
- UI: profile editor, “forget”, export, admin vs self-service
- Structured vs free-text entries (prefer structured for tool args and entity ids)
- How memory interacts with skill `slots` and route playbooks

## Related code today

- `memory.py` — conversation history only
- `context.py` — hardcoded route hints (e.g. `news_curate`)
- `skills/` — workflow learning; observer rejects news summaries as non-workflows
- `playbooks.py` — editable route baselines (closest to system memory for procedures)

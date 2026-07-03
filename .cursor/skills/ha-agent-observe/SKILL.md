---
name: ha-agent-observe
description: Observe HA Agent chats in realtime and analyze turns for issues when the user asks. Use when debugging agent turns, watching live console or Assist chats, evaluating tool errors, verifier failures, or verifying fixes. Activates on observe chat, watch agent, analyze turn, ha_agent_turn_recorded, ha_agent/diagnostics.
---

# HA Agent Chat Observation

Use this skill when the user asks you to **watch**, **observe**, or **analyze** HA Agent chats (console or Assist).

## Prerequisites

- Home Assistant long-lived access token (`HA_TOKEN`)
- HA URL (`HA_URL`, e.g. `http://homeassistant.local:8123`)
- Admin access (all `ha_agent/*` WS commands require admin)

See the **ha-api** skill for WebSocket connection setup.

## Realtime observation

### 1. Subscribe to bus events

After WebSocket auth, subscribe to:

| Event | When |
|-------|------|
| `ha_agent_chat_delta` | Streaming content, thinking, tools (console **and** Assist) |
| `ha_agent_chat_done` | Turn finished (console includes `turn_meta`; Assist includes `error` if failed) |
| `ha_agent_turn_recorded` | Full activity trace committed (best hook for post-turn analysis) |

Filter payloads by `entry_id` and `conversation_id`.

### 2. Poll live buffer (in-progress turns)

```json
{
  "type": "ha_agent/diagnostics/observe",
  "id": 1,
  "entry_id": "<entry_id>",
  "conversation_id": "<optional>"
}
```

Returns:

- `active[]` — in-flight turns with buffered `deltas`
- `console_in_progress` — bool when `conversation_id` provided
- `latest_turn` — most recent completed activity row

### 3. Mid-turn history snapshot

```json
{
  "type": "ha_agent/chat/turn/status",
  "id": 2,
  "entry_id": "<entry_id>",
  "conversation_id": "<conversation_id>"
}
```

Returns `{ in_progress, history }`.

## On-demand analysis (when asked)

### Fetch one turn

```json
{
  "type": "ha_agent/activity/get",
  "id": 3,
  "entry_id": "<entry_id>",
  "timestamp": 1710000000.0
}
```

Or `"latest": true` with optional `"conversation_id"`.

### Analyze issues and fixes

```json
{
  "type": "ha_agent/diagnostics/analyze_turn",
  "id": 4,
  "entry_id": "<entry_id>",
  "latest": true
}
```

Response `analysis` includes:

- `severity` — `ok`, `warning`, or `error`
- `summary` — one-line verdict
- `issues[]` — `{ kind, detail, suggestion }`
- `suggested_actions[]` — e.g. promote eval case, inspect skill repair, open chat history

Interpret the structured output and explain fixes to the user. Re-run after they apply a fix.

## Inject, observe, and evaluate (when asked)

### One-shot inject (recommended)

Send a console message, wait for completion, and get trace + analysis in one call:

```json
{
  "type": "ha_agent/diagnostics/inject_turn",
  "id": 5,
  "entry_id": "<entry_id>",
  "text": "turn on the dining room lights"
}
```

Optional: `conversation_id`, `timeout` (seconds).

Response includes `conversation_id`, `done`, `turn`, `analysis`, and `live` (buffered deltas).

### CLI from this repo

```bash
export HA_URL=http://homeassistant.local:8123
export HA_TOKEN=...
python3 scripts/inject_ha_agent_chat.py "turn on the dining room lights"
```

Watch-only (no inject):

```bash
python3 scripts/watch_ha_agent_chat.py
```

## Typical workflow

1. User asks you to **watch** → subscribe to events or run `watch_ha_agent_chat.py`.
2. User asks you to **try** a command → call `inject_turn` or run `inject_ha_agent_chat.py`.
3. Read `analysis.severity`, `issues`, and `suggested_actions`; propose fixes.
4. User retries → inject again and compare outcomes.
5. Optional: promote good turns via `ha_agent/eval/cases/promote`.

## Entry ID

Call `ha_agent/subscribe` first to get `entry_id` if unknown.

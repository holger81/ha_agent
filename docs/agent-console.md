# HA Agent Console

The **HA Agent Console** is a Home Assistant sidebar panel for text chat, skill management, settings, and activity diagnostics — without relying on Assist voice or the device page alone.

## Access

After installing or updating **HA Agent** (v1.1.0+), restart Home Assistant. The panel appears in the sidebar as **HA Agent** (`mdi:robot-happy`). Admin users only.

No `configuration.yaml` changes are required; the integration registers the panel automatically via `panel_custom`.

## Tabs

| Tab | Purpose |
|-----|---------|
| **Chat** | Streamed text chat with the same `run_agent()` loop as Assist; model thinking and tool progress appear in muted blocks (not spoken). |
| **Skills** | Browse, search, enable/disable, create, edit, delete, export, and import skills. Pending “save as skill” drafts can be confirmed from the chat banner. |
| **Settings** | View and change models, thinking level, routing, skill switches, streaming, history, and memory persistence. |
| **Activity** | Recent turn traces: route, tool calls, matched skills, errors. |

When multiple HA Agent config entries exist, use the header dropdown to switch entries.

## WebSocket API

All commands require an authenticated admin WebSocket connection. Pass `entry_id` for every command except `ha_agent/subscribe` (optional there).

### Handshake

| Type | Description |
|------|-------------|
| `ha_agent/subscribe` | Returns `entries`, `entry_id`, `config` snapshot, and `status`. |
| `ha_agent/status` | Runtime status (last route, active skill, health). |

### Chat

| Type | Description |
|------|-------------|
| `ha_agent/chat/send` | `{ entry_id, conversation_id, text }` — streams deltas via bus events, returns `{ history }` when done. |
| `ha_agent/chat/history/list` | `{ entry_id, conversation_id }` → `{ history }`. |
| `ha_agent/chat/history/clear` | `{ conversation_id }` → `{ success }`. |

**Streaming events** (subscribe via `hass.connection.subscribeEvents`):

- `ha_agent_chat_delta` — `{ entry_id, conversation_id, content?, thinking? }`
- `ha_agent_chat_done` — `{ entry_id, conversation_id, last_route?, active_skill?, error?, cancelled? }`

### Skills

| Type | Description |
|------|-------------|
| `ha_agent/skills/list` | Paginated list (`limit`, `offset`). |
| `ha_agent/skills/search` | FTS search (`query`, `enabled_only`). |
| `ha_agent/skills/get` | Full skill JSON. |
| `ha_agent/skills/set_enabled` | Toggle enabled flag. |
| `ha_agent/skills/delete` | Delete by `skill_id`. |
| `ha_agent/skills/create` | Create from `skill` object. |
| `ha_agent/skills/update` | Update by `skill_id`. |
| `ha_agent/skills/pending_get` | Pending draft for a conversation. |
| `ha_agent/skills/pending_confirm` | Save pending draft as skill. |
| `ha_agent/skills/pending_dismiss` | Discard pending draft. |
| `ha_agent/skills/export` | Export all skills as JSON array. |
| `ha_agent/skills/import` | Import skills from array. |

### Config, activity, threads

| Type | Description |
|------|-------------|
| `ha_agent/config/get` | Config snapshot for settings tab. |
| `ha_agent/config/set` | Partial `updates` dict; may reload integration. |
| `ha_agent/activity/list` | Paginated turn traces. |
| `ha_agent/threads/list` | Conversation thread metadata. |
| `ha_agent/threads/update` | Rename or pin a thread. |

## Memory and threads

- Conversation history is in-process by default (same as Assist).
- Enable **Persist conversation memory** in Settings to store history across restarts (`.storage/ha_agent_memory_{entry_id}.json`).
- Thread titles are updated from the first chat message and stored in `.storage/ha_agent_threads_{entry_id}.json`.

## Development

Frontend sources live in `custom_components/ha_agent/frontend/`. The panel module is served at `/ha_agent_panel/ha-agent-panel.js` with a cache-busting query from the integration version.

```bash
ruff check custom_components tests
pytest tests/
```

## Screenshots

_Screenshots placeholder — add after first HA install._

# Assist pipeline setup

Use **[HA Agent](https://github.com/holger81/ha_agent)** (this integration) for the **conversation** stage. Pair with **[LiquidAI](https://github.com/holger81/ha_liquidai)** for STT and TTS.

## Requirements

- Home Assistant **2025.10** or newer
- OpenAI-compatible LLM (e.g. llama.cpp at `:9292/v1`)
- MCP Proxy with bearer token (default `http://192.168.10.31:2222/mcp`)
- [LiquidAI](https://github.com/holger81/ha_liquidai) for speech I/O

## Install HA Agent

1. Deploy `custom_components/ha_agent/`:

   ```bash
   HA_CONFIG=/path/to/ha/config ./scripts/deploy_to_ha.sh
   ```

2. Restart Home Assistant
3. **Settings → Devices & services → Add integration → HA Agent**
4. Complete the config flow:
   - **Agent prompts** — system prompt and MCP tool instructions
   - **LLM backend** — base URL, **model dropdown** (loaded from `/v1/models`), optional API key
   - **MCP Proxy** — URL, bearer token, health URL
   - **Agent settings** — max iterations, history turns, streaming

## Change models and routing later

On the **HA Agent device** page (like Zigbee device settings):

| Section | Entity | Purpose |
|---------|--------|---------|
| **Configuration** | Chat model | Main conversational model |
| **Configuration** | Action model | Faster model for device commands (when routing enabled) |
| **Configuration** | Action model routing | Toggle separate action model |
| **Diagnostic** | Last route | `chat` or `action` for the last Assist turn |
| **Diagnostic** | MCP tools | Number of session tools loaded |
| **Diagnostic** | LLM server / MCP Proxy | `online` / `offline` health |

### Skills (learned workflows)

HA Agent can learn multi-step workflows and reuse them on similar requests.

| Section | Entity | Purpose |
|---------|--------|---------|
| **Configuration** | Skill learning | Learn skills from successful multi-step turns |
| **Configuration** | Skill auto-save | Save without asking (when learning is on) |
| **Configuration** | Skill auto-use | Inject matching skills into Assist turns |
| **Diagnostic** | Skills total / enabled | Saved skill counts |
| **Diagnostic** | Active skill | Best match for the last turn |
| **Diagnostic** | Last skill improved | Most recent auto-improvement |

**Chat commands** (per skill, no per-skill HA entities):

- “list my skills”
- “disable the dining room lights skill”
- “enable skill …”
- “delete skill …”
- After a successful multi-step task (learning on, auto-save off): “yes” saves the offered skill

**Services:** `ha_agent.enable_skill`, `disable_skill`, `delete_skill`, `list_skills`

Skills are stored in SQLite with FTS search (`.storage/ha_agent_skills_<entry_id>.db`) so discovery stays fast at large scale. Configure **max skills per turn** under **Configure → Skills**.

## Configure the Assist pipeline

| Stage | Provider |
|-------|----------|
| Speech-to-text | **LiquidAI STT** (`stt.ha_liquidai_custom`) |
| Conversation | **HA Agent** (`conversation.ha_agent`) |
| Text-to-speech | **LiquidAI TTS** (`tts.ha_liquidai_custom`) |

Remove any old **Webhook Conversation** entry after [migrating from n8n](migration-from-n8n.md).

## Verify the agent

### Backend smoke test (from dev machine)

```bash
pip install aiohttp
export HA_AGENT_MCP_TOKEN="your-bearer-token"   # if required
python3 scripts/smoke_test_phase4.py
```

Checks LLM `/models`, MCP health, `initialize`, and `tools/list`.

### Assist validation (Phase 4 sign-off)

1. Expose entities in **Settings → Voice assistants → Expose**
2. Wire pipeline: LiquidAI STT → **HA Agent** → LiquidAI TTS
3. Test prompts:
   - “Turn off the dining room lights” (exposed entity)
   - “Open the patio cover” (search + service call)
   - “What's the news?”
   - “How many unread emails do I have?”
4. Ask a follow-up in the same conversation to confirm memory
5. Confirm text appears progressively in Assist debug (streaming enabled)
6. For voice, confirm pipeline debug shows `stream_response: true` and `tts_start_streaming: true` (Home Assistant starts LiquidAI TTS after ~60 characters of assistant text)
7. Check HA logs for MCP tool calls and LLM responses

## Troubleshooting

| Symptom | Check |
|---------|--------|
| LLM connection failed during setup | LLM URL, model name, firewall |
| MCP connection failed | Bearer token, `GET /api/health` on proxy host |
| Tool errors in Assist | Tool instructions in config; MCP proxy logs |
| No streaming text | Enable streaming in agent settings; HA ≥ 2025.10 |
| Text streams but speech waits until the end | Pipeline needs `stream_response: true`; ask a longer question so TTS streaming starts; tune LiquidAI “stream first chunk” |
| Speech issues | [LiquidAI assist setup](https://github.com/holger81/ha_liquidai/blob/main/docs/assist-setup.md) |

## Related

- [Migration from n8n](migration-from-n8n.md)
- [PLAN.md](../PLAN.md)
- [LiquidAI STT/TTS](https://github.com/holger81/ha_liquidai)

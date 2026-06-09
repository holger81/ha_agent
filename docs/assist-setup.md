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

## Change the LLM model later

Three ways (no full reconfigure needed):

1. **Device page** — open the **HA Agent** device → **Configuration** → **LLM model** dropdown (like Zigbee device settings)
2. **Integration options** — **Settings → Devices & services → HA Agent → Configure → Change LLM model**
3. **Reconfigure** — full setup wizard if you also need to change LLM URL or MCP settings

## Configure the Assist pipeline

| Stage | Provider |
|-------|----------|
| Speech-to-text | **LiquidAI STT** (`stt.ha_liquidai_custom`) |
| Conversation | **HA Agent** (`conversation.ha_agent`) |
| Text-to-speech | **LiquidAI TTS** (`tts.ha_liquidai_custom`) |

Remove Webhook Conversation if still wired.

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
6. Check HA logs for MCP tool calls and LLM responses

## Troubleshooting

| Symptom | Check |
|---------|--------|
| LLM connection failed during setup | LLM URL, model name, firewall |
| MCP connection failed | Bearer token, `GET /api/health` on proxy host |
| Tool errors in Assist | Tool instructions in config; MCP proxy logs |
| No streaming text | Enable streaming in agent settings; HA ≥ 2025.10 |
| Speech issues | [LiquidAI assist setup](https://github.com/holger81/ha_liquidai/blob/main/docs/assist-setup.md) |

## Related

- [PLAN.md](../PLAN.md)
- [LiquidAI STT/TTS](https://github.com/holger81/ha_liquidai)

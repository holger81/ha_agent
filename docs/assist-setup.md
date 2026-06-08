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
   - **LLM backend** — base URL, model, optional API key
   - **MCP Proxy** — URL, bearer token, health URL
   - **Agent settings** — max iterations, history turns, streaming

## Configure the Assist pipeline

| Stage | Provider |
|-------|----------|
| Speech-to-text | **LiquidAI STT** (`stt.ha_liquidai_custom`) |
| Conversation | **HA Agent** (`conversation.ha_agent`) |
| Text-to-speech | **LiquidAI TTS** (`tts.ha_liquidai_custom`) |

Remove Webhook Conversation if still wired.

## Verify the agent

1. Expose entities in **Settings → Voice assistants → Expose**
2. Ask Assist: “Turn off the dining room lights” or “What's the news?”
3. Check HA logs for MCP tool calls and LLM responses

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

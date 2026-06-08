# ha_agent

Home Assistant custom integration for the **Assist conversation agent** — LLM + MCP tool loop.

Pair with **[ha_liquidai](https://github.com/holger81/ha_liquidai)** for LiquidAI STT and TTS in the same Assist pipeline.

Replaces n8n Webhook Conversation from [ha_liquidai_n8n](https://github.com/holger81/ha_liquidai_n8n).

## Status

| Feature | Status |
|---------|--------|
| LLM client (OpenAI-compatible) | **Done** |
| MCP Proxy client | **Done** |
| Agent tool loop + memory | **Done** |
| Conversation platform | **Done** |
| Config flow (prompts → LLM → MCP) | **Done** |
| Multi-model router | Phase 5 |
| Full n8n migration | Phase 6 |

See [PLAN.md](PLAN.md) for the roadmap (Phases 4–6).

## Assist pipeline

| Stage | Integration |
|-------|-------------|
| STT | **[ha_liquidai](https://github.com/holger81/ha_liquidai)** → LiquidAI `/v1/asr` |
| Conversation | **This repo** → LLM + MCP tools |
| TTS | **[ha_liquidai](https://github.com/holger81/ha_liquidai)** → LiquidAI `/v1/tts` |

## Requirements

- Home Assistant **2025.10+** (conversation streaming)
- OpenAI-compatible LLM (e.g. llama.cpp)
- MCP Proxy with bearer token

## Install

```bash
HA_CONFIG=/path/to/ha/config ./scripts/deploy_to_ha.sh
```

Or install **HA Agent** from HACS, then add **LiquidAI** from [ha_liquidai](https://github.com/holger81/ha_liquidai).

## Development

```bash
pip install -r requirements.txt
ruff check .
pytest tests/
```

## Docs

- [Assist pipeline setup](docs/assist-setup.md)
- [LiquidAI STT/TTS setup](https://github.com/holger81/ha_liquidai/blob/main/docs/assist-setup.md)

## License

MIT

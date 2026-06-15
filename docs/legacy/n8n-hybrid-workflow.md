# Legacy n8n hybrid workflow (reference only)

The **Webhook Conversation (Hybrid)** workflow lived in [ha_liquidai_n8n](https://github.com/holger81/ha_liquidai_n8n). It is **retired** for production Assist; use **HA Agent** + **ha_liquidai** instead.

## Archived files

| File | Purpose |
|------|---------|
| `simple_n8n_workflow_hybrid.json` | Static workflow export |
| `simple_n8n_workflow_hybrid.sdk.js` | n8n Workflow SDK source |
| `scripts/agent_input_code.js` | Tool-context builder (ported to `context.py`) |

## Webhook URLs (historical)

| Sub-entry | URL |
|-----------|-----|
| Conversation Agent | `http://<n8n-host>:5678/webhook/agent` |
| STT | `http://<n8n-host>:5678/webhook/stt` |
| TTS | `http://<n8n-host>:5678/webhook/tts` |

## Replacement

See [migration-from-n8n.md](../migration-from-n8n.md).

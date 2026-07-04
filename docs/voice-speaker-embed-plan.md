# LiquidAI audio — speaker embedding bridge

**Canonical doc:** [ha_liquidai/docs/voice-speaker-embed-plan.md](https://github.com/holger81/ha_liquidai/blob/main/docs/voice-speaker-embed-plan.md)

This file is a **pointer** in the ha_agent repo so Phase 9b links from `PLAN.md` resolve
on GitHub. Implement STT + voice-cache changes in **ha_liquidai**; implement guest
clustering in **ha_agent** ([agent-voice-inference-plan.md](agent-voice-inference-plan.md)).

## Quick summary

1. **Inference box** (`liquidai-audio-docker` on `.31`): `POST /v1/speaker/embed` → 192-d Sherpa embedding
2. **ha_liquidai STT**: parallel `/v1/asr` + `/v1/speaker/embed`; store in `voice_cache.py`
3. **ha_agent conversation**: pop cache → cluster embedding → stable `agent_user_id`

Full API contract, file checklist, tests, and rollout steps are in the ha_liquidai doc.

# Agent identity — design notes (future implementation)

Captured from product discussion (2026-07). **Prerequisite for Phase 10 memory**
([agent-memory-design.md](agent-memory-design.md)). Do not implement memory until
agent turns can resolve *who* is speaking or typing.

## Problem

Memory has at least two scopes ([agent-memory-design.md](agent-memory-design.md)):

- **User-bound** — you vs your wife prefer different news/email defaults
- **System / household** — local news procedure, dining-room light entity id

User-bound memory requires a stable **agent user id** on every turn. Today ha_agent
has **none**:

- Assist/voice: `conversation_id` only; `ConversationInput.context.user_id` is often
  `null` on voice pipelines ([HA community](https://community.home-assistant.io/t/assist-get-source-when-running-a-script-with-an-llm/861595))
- Console: admin WebSocket auth exists (`require_admin`) but `connection.user.id` is
  not passed into `run_agent`
- `device_id` / `satellite_id` give **location**, not **person**

## Requirements (from product)

| Requirement | Notes |
|-------------|--------|
| ≥ 4 **registered** household users | Named, durable, linkable to HA Person / HA user |
| **Guest** users | Distinguish multiple guests, not one anonymous bucket |
| Guest **promotion** | Guest → registered user (voice samples + memory carry-over) |
| Guest **merge** | Fix misclassification: merge guest A + guest B → one profile |
| **Text / console** identity | HA login user; admins can override active user |
| **Voice** identity | Speaker recognition; independent of HA login |
| **Admin override** | Admins can set/impersonate user for console and correction flows |
| Precedence | User memory overrides system memory (see memory doc) |

## Recommended identity model

Separate **identity** (who) from **memory** (what they prefer).

```text
AgentUser
├── id                  (stable uuid, internal)
├── kind                registered | guest
├── display_name        "Holger", "Guest 3 (Jul 4 visit)"
├── ha_user_id          optional — HA auth user (console)
├── person_entity_id    optional — person.holger
├── voice_profile_ids   optional — links to enrollment(s)
├── created_at, updated_at
├── merged_into         optional — when guest merged away
└── metadata            notes, admin tags

VoiceProfile
├── id
├── agent_user_id       owner (registered or guest)
├── backend             resemblyzer | voicebm | wespeaker | ...
├── embedding_ref       path/id in speaker-ID service
├── sample_count, last_seen, avg_confidence
└── satellite_ids       optional weak hint (not authoritative)
```

**Registered users (4+):** Pre-provisioned in HA Agent settings; map to `person.*`
and/or HA users; voice enrollment attaches `VoiceProfile`(s).

**Guest users:** Created when speaker ID returns *unknown* or *low-confidence
non-match*. Each distinct cluster gets its own `AgentUser(kind=guest)` — not a
single shared “unknown” bucket (contrast VoiceBM’s single virtual `user` id).

**System scope:** Not an `AgentUser`; household defaults live in system memory
(no `agent_user_id`). See memory doc.

## Resolution per channel

### Text / console

```text
WebSocket connection.user.id
    → map to AgentUser (via ha_user_id table)
    → optional admin override (session or per-turn picker in panel)
    → run_agent(..., agent_user=...)
```

- Default: logged-in HA user → registered `AgentUser`
- Admin override: dropdown “Act as: Holger / Sarah / Guest …” stored in console
  session or explicit per message (`ha_agent/chat/send` param)
- Audit: log `resolved_user_id`, `override_by_admin_id` on activity traces

### Voice / Assist

```text
Assist pipeline audio
    → Speaker ID service (see research below)
    → {speaker_key, confidence, embedding?}
    → Identity resolver in ha_agent
         ├─ match enrolled registered user → AgentUser(kind=registered)
         ├─ match existing guest cluster   → AgentUser(kind=guest)
         └─ no match                       → create new guest OR nearest cluster
    → run_agent(..., agent_user=..., source=assist)
```

**Do not rely on** `user_input.context.user_id` for voice — it is usually unset
unless an integration explicitly sets it (e.g. VoIP with configured HA user).

**Weak hints only:** `device_id`, `satellite_id` → room/area for device control,
not for memory user selection.

### Fallback chain

```text
1. Voice speaker match (if assist + audio path available)
2. Explicit admin / console override
3. HA context.user_id (if present)
4. New or generic guest (voice) / console default user (text)
```

## Guest lifecycle

### Create

When speaker ID returns unknown or confidence below `guest_create_threshold`:

1. Compare embedding to existing **guest** centroids (cosine similarity)
2. If within `guest_cluster_threshold` → attach to existing guest
3. Else → create `AgentUser(kind=guest, display_name="Guest N")` + `VoiceProfile`

Store last-seen, utterance count, optional display name from intro (“I'm John”).

### Promote guest → registered

Admin or voice enrollment flow:

1. Select guest + target registered slot (or new registered user)
2. Move guest **user-bound memory** to registered user (merge, don’t duplicate)
3. Add guest voice embedding to registered enrollment; retrain backend profile
4. Mark guest `merged_into` / archive guest row
5. Future utterances match registered user

### Merge guests (misclassification)

Admin selects guest A + guest B → merged guest C:

1. Union voice embeddings (centroid or multi-sample set)
2. Union user-bound memory (conflict: last-write or admin pick)
3. Repoint activity history
4. Archive A, B

### Split (optional, later)

If one guest was wrongly merged, admin splits with manual sample assignment — low
priority; merge is the urgent fix path.

## Voice identification — research summary

Home Assistant **does not** ship speaker recognition in core Assist (2026). Community
requests Wyoming/protocol integration; not available natively.

### Option A — [hass-speaker-recognition](https://github.com/EuleMitKeule/speaker-recognition) (Resemblyzer)

| Pros | Cons |
|------|------|
| HA addon + integration | Unknown speaker = one label, not guest clusters |
| REST `/recognize` + `/train` | Resemblyzer aging; server dep prefers Py &lt;3.10 |
| STT + conversation sub-entries | ha_agent must add guest clustering layer |
| Local, embedding cache | Integration maturity ~39 GitHub stars |

**Fit:** Good MVP if we own guest clustering in ha_agent.

### Option B — [VoiceBM](https://github.com/cybericebyte/VoiceBM) (Sherpa-ONNX / WeSpeaker)

| Pros | Cons |
|------|------|
| Fast, local, MQTT discovery | Identity via MQTT/automation glue |
| Enrollment + blocklist | Unknown → single virtual `user` id (needs extension) |
| HA-oriented docs | Extra moving parts (MQTT topics, automations) |
| Profile attributes per person | Less “drop in” than REST on same host |

**Fit:** Strong engine; align guest model with our multi-guest design (fork or
wrap — don’t use one `user` bucket for everyone).

### Option C — [Wyoming Voice Match](https://github.com/jxlarrea/homeassistant-voice-recipes) (ECAPA-TDNN)

| Pros | Cons |
|------|------|
| Speaker verification + noise rejection | Enrolled speakers only; rejects unknowns |
| Sits in front of STT in pipeline | No guest differentiation |
| GPU option | Another Wyoming hop |

**Fit:** Complement for “only household members can run commands”, not primary
guest/memory identity.

### Option D — Extend **ha_liquidai** STT

Audio already flows through LiquidAI STT on the inference host (`192.168.10.31`).

| Pros | Cons |
|------|------|
| Single place: audio → speaker + transcript | Couples identity to STT integration |
| Can pass identity alongside text to pipeline | New ML dep on inference box |
| Matches existing stack | HA STT entity API may not expose metadata |

**Fit:** **Chosen path** — see [agent-voice-inference-plan.md](agent-voice-inference-plan.md).
Sherpa-ONNX embed on inference box; ha_liquidai voice turn cache; ha_agent guest
clustering. `HA_AGENT_IDENTITY` in `extra_system_prompt` kept as override/debug.

### Option E — Wait for HA core / Wyoming speaker ID

Not viable as sole plan; monitor [architecture discussion](https://github.com/home-assistant/architecture/discussions/1114).

### Recommendation (phased)

| Phase | Approach |
|-------|----------|
| **Identity 1** (text) | HA login user + admin override; no voice yet — **shipped 9a** |
| **Identity 2** (voice MVP) | **Sherpa-ONNX** on inference box + **ha_liquidai** STT bridge + **ha_agent** guest clustering — [plan](agent-voice-inference-plan.md) |
| **Identity 3** (voice polish) | Satellite-aware cache keys, promote/merge UI (9c), optional combined `/v1/assist/transcribe` |
| **Optional** | Voice Match upstream for wake-word false-trigger reduction |

**Guest clustering** should live in **ha_agent** regardless of backend — backends
only return `(embedding | speaker_label, confidence)`.

## Integration contract (ha_agent)

Define an internal protocol; swap backends:

```python
@dataclass
class SpeakerMatch:
    backend: str
    speaker_key: str | None      # enrolled label or None
    confidence: float
    embedding: bytes | None        # optional for clustering
    raw: dict[str, Any]

async def resolve_agent_user(
    hass: HomeAssistant,
    entry_id: str,
    *,
    channel: Literal["assist", "console"],
    ha_user_id: str | None,
    speaker_match: SpeakerMatch | None,
    admin_override_user_id: str | None,
) -> AgentUser
```

**Assist hook points today** (`conversation.py`):

- Read `user_input.extra_system_prompt` for injected identity JSON (automation bridge)
- Read `user_input.context.user_id`, `device_id`, `satellite_id`
- Pass resolved `AgentUser` into `run_agent`

**Console hook** (`websocket_api.py` → `api/chat.py`):

- `connection.user.id` + optional `agent_user_id` override param

**Activity / traces:** add `agent_user_id`, `agent_user_kind`, `speaker_confidence`,
`identity_source` (voice | login | override | guest_new).

## UI (sketch)

**Settings → Users**

- 4+ registered slots: name, HA Person, HA user link, voice enrollment status
- Guest list: last seen, utterance count, merge / promote actions
- Voice enrollment: record N samples per registered user (panel or Assist script)

**Console**

- “Act as” selector (admin only)
- Identity chip on each turn in activity log

**Voice correction**

- “That wasn’t me” → reassign last turn to another user; optional merge guests

## Security & privacy

- Voice embeddings and samples stay **local** (same policy as LLM/MCP)
- Guest memory: consider TTL or manual cleanup for GDPR/hospitality
- Admin override audited
- Low-confidence: do not apply sensitive user memory; prefer system defaults + confirm

## Implementation phases

```text
Phase 9a — Identity registry (SQLite) + text/console resolution + admin override  [shipped]
Phase 9b — Sherpa-ONNX embed on .31 + ha_liquidai cache + ha_agent clustering  [planned]
Phase 9c — Guest promote / merge UI + APIs
Phase 10 — User-bound + system memory (depends on Phase 9)
```

See [agent-voice-inference-plan.md](agent-voice-inference-plan.md) for 9b detail.

## Open questions

- Link registered users to **HA Person** only, or require HA auth user too?
- Per-user **conversation history** vs shared household thread + user tag on turns?
- Guest auto-naming from STT (“I'm Dave”) — trust level?
- Confidence thresholds: tunable per household?
- Satellite-specific enrollment (same person, different mics)?

## Related files (today)

| Path | Relevance |
|------|-----------|
| `conversation.py` | Assist entry; inject identity here |
| `api/chat.py`, `websocket_api.py` | Console user + override |
| `agent.py` | Thread `agent_user` through `run_agent` |
| `memory.py` | Conversation history — separate from identity store |
| `activity.py`, `skills/models.py` | Add identity fields to traces |
| `docs/agent-memory-design.md` | Memory scopes once identity exists |
| `identity/voicebm.py` | `HA_AGENT_IDENTITY` parser (override/debug) |
| `docs/agent-voice-inference-plan.md` | Phase 9b Sherpa + inference box plan |
| `../ha_liquidai/docs/voice-speaker-embed-plan.md` | STT bridge + `/v1/speaker/embed` contract |

## References

- [HA Assist pipelines](https://developers.home-assistant.io/docs/voice/pipelines/)
- [ConversationInput device_id / satellite_id PR](https://github.com/home-assistant/core/pull/164414)
- [hass-speaker-recognition](https://pypi.org/project/hass-speaker-recognition/)
- [Sherpa-ONNX speaker ID](https://github.com/k2-fsa/sherpa-onnx#speaker-identification-speaker-id)
- [agent-voice-inference-plan.md](agent-voice-inference-plan.md)
- [HA community: no voice user in context](https://community.home-assistant.io/t/getting-the-frontend-user-id/919279)

---
title: Look up sensor or entity status
description: Find a Home Assistant entity and report its current reading or status.
slug: look-up-sensor-or-entity-status
triggers:
  - what is the temperature in {{query}}
  - status of {{query}}
  - look up {{query}} sensor
  - what is the {{query}} right now
  - how much {{query}}
route_scope: chat
enabled: true
slots:
  - name: query
    description: Search term for the place, person, device, or sensor
    default: ""
tool_steps:
  - toolName: home_assistant__ha_search
    arguments:
      query: "{{query}}"
---

# Look up status

When the user asks for a reading or status for a place, person, device, or sensor:

1. Set `query` to the place, device, or reading they named. Do not guess an entity_id.
2. Call `home_assistant__ha_search` with `query={{query}}`. Set `domain_filter` only when the ask implies a domain (light, sensor, climate, and so on).
3. Prefer results whose area or friendly name matches the ask. Ignore unrelated substring hits.
4. If the chosen result has no current state, call `home_assistant__ha_get_state` with that `entity_id`.
5. Answer from tool results; do not invent values.

Never use `home_assistant__ha_call_service` for a status/reading question — that tool controls devices, it does not look entities up.

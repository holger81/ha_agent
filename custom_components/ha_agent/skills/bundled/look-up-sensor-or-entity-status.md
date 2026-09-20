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
      domain_filter: sensor
---

# Look up status

When the user asks for a reading or status for a place, person, device, or sensor:

1. Call `home_assistant__ha_search` with a short `query={{query}}`. Prefer `domain_filter` when it narrows results (e.g. `sensor` for production/power/temperature).
2. Pick the entity whose `device_class` / `unit_of_measurement` matches the asked reading (temperature ≠ voltage; power/energy for production).
3. If search results do not include the current state, call `home_assistant__ha_get_state` with the chosen `entity_id`.
4. Answer from tool results; do not invent values.

Never use `home_assistant__ha_call_service` for a status/reading question — that tool controls devices, it does not look entities up.

---
title: News briefing
description: Curate today's headlines and deliver a short news briefing from configured feeds and search.
slug: news-briefing
triggers:
  - news
  - headlines
  - headline
  - briefing
  - today's news
  - news briefing
  - what's in the news
  - nachrichten
route_scope: news
enabled: true
slots:
  - name: digest_scope
    description: Optional focus for the briefing (local, tech, world, etc.)
    default: ""
tool_steps:
  - toolName: mcp_news__news_curate
    arguments:
      query: "{{digest_scope}}"
---

# News briefing

When the user asks for news, headlines, or a briefing:

1. Call `mcp_news__news_curate` (pass `query` from `{{digest_scope}}` when the user named a topic; otherwise omit or use an empty query).
2. Summarize the curated items for the user — titles, sources, and a one-line takeaway each. Do not invent headlines.
3. If curation fails or returns nothing useful, discover news tools with `searchToolsForDomain` / `searchTool` in the news domain and retry with an exact discovered toolName.

Keep answers concise and suitable for voice Assist.

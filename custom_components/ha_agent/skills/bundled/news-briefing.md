---
title: News briefing
description: Deliver the cached news briefing for today, local, or Germany.
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
    description: Optional focus for the briefing (local, germany, global, today)
    default: ""
tool_steps:
  - toolName: mcp_news__news_briefing
    arguments:
      scope: "{{digest_scope}}"
---

# News briefing

When the user asks for news, headlines, or a briefing:

1. Call `mcp_news__news_briefing` once. Pass `scope` from `{{digest_scope}}` when the user named a focus (local, germany, global, today). Omit `scope` for a general briefing.
2. Reply with the `briefing` field already in the tool result. You may add a few item titles underneath. Do not rewrite the HTML `summary` fields and do not invent headlines.
3. Ignore `responseCacheId`, `hasMore`, and truncation. Do not paginate and do not call another news tool unless `news_briefing` returned an error.

Keep the reply concise and suitable for voice Assist.

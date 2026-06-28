---
title: Check and read unread emails
description: Check the inbox for unread messages, report the count, and read unread mail when the user asks.
slug: check-and-read-unread-emails
triggers:
  - unread email
  - unread emails
  - new emails
  - new email
  - check inbox
  - check my email
  - any new mail
  - read my email
route_scope: email
enabled: true
slots:
  - name: mailbox
    description: IMAP mailbox folder (INBOX, Junk, Sent, etc.)
    default: INBOX
tool_steps:
  - toolName: mail_mcp__imap_mailbox_status
    arguments:
      mailbox: "{{mailbox}}"
  - toolName: mail_mcp__imap_search_messages
    arguments:
      mailbox: "{{mailbox}}"
      unread_only: true
      limit: 10
  - toolName: mail_mcp__imap_get_message
    arguments:
      message_id: "{{message_id}}"
---

# Check and read unread email

When the user asks about new or unread mail:

1. Call `mail_mcp__imap_mailbox_status` with mailbox `{{mailbox}}` for unseen count.
2. Call `mail_mcp__imap_search_messages` with mailbox `{{mailbox}}`, `unread_only=true`, and `limit=10`.
3. To read a specific message, call `mail_mcp__imap_get_message` with `message_id` from the search results.
4. Answer from tool results only — report unseen count, then summarize subjects/senders. Never invent mail content.

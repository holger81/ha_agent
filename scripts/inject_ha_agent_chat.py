#!/usr/bin/env python3
"""Inject an HA Agent console message and print observation + analysis."""

from __future__ import annotations

import argparse
import asyncio
import json

from ha_agent_ws import HA_URL, HaAgentWsClient


def _print_result(result: dict) -> None:
    analysis = result.get("analysis") or {}
    turn = result.get("turn") or {}
    done = result.get("done") or {}
    print(f"conversation_id: {result.get('conversation_id')}")
    print(f"severity: {analysis.get('severity')}")
    print(f"summary: {analysis.get('summary')}")
    if done.get("error"):
        print(f"error: {done.get('error')}")
    if turn.get("assistant_text"):
        print(f"assistant: {turn.get('assistant_text')}")
    print(f"route: {turn.get('route')} outcome: {turn.get('outcome')} "
          f"errors: {turn.get('tool_errors')}")
    for issue in analysis.get("issues") or []:
        if issue.get("kind") == "ok":
            continue
        print(f"  [{issue.get('kind')}] {issue.get('detail')}")
        if issue.get("suggestion"):
            print(f"    → {issue.get('suggestion')}")
    for action in analysis.get("suggested_actions") or []:
        print(f"  action: {action.get('action')} — {action.get('detail')}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Inject HA Agent chat and analyze")
    parser.add_argument("text", help="User message to send")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    client = await HaAgentWsClient.connect()
    try:
        print(f"Connected to {HA_URL} entry={client.entry_id}")
        print(f"Injecting: {args.text!r}")
        result = await client.inject_turn(
            args.text,
            conversation_id=args.conversation_id,
            timeout=args.timeout,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _print_result(result)
    finally:
        await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""Reasoning / thinking level helpers for LLM requests."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ThinkingLevel(StrEnum):
    """Supported reasoning effort levels for the chat model."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    INFINITE = "infinite"


THINKING_LEVEL_OPTIONS: list[str] = [level.value for level in ThinkingLevel]
DEFAULT_THINKING_LEVEL = ThinkingLevel.OFF


def normalize_thinking_level(value: str | bool | None) -> str:
    """Coerce stored config values to a valid thinking level."""
    if isinstance(value, bool):
        return ThinkingLevel.MEDIUM if value else ThinkingLevel.OFF
    if isinstance(value, str) and value in THINKING_LEVEL_OPTIONS:
        return value
    return DEFAULT_THINKING_LEVEL


def apply_thinking_to_payload(payload: dict[str, Any], level: str) -> None:
    """Add reasoning parameters for llama.cpp and OpenAI-compatible servers."""
    normalized = normalize_thinking_level(level)
    if normalized == ThinkingLevel.OFF:
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return

    kwargs: dict[str, Any] = {
        "enable_thinking": True,
        "reasoning_effort": normalized,
    }
    if normalized == ThinkingLevel.INFINITE:
        payload["reasoning_effort"] = "high"
        kwargs["reasoning_effort"] = "high"
        kwargs["reasoning_budget"] = -1
    else:
        payload["reasoning_effort"] = normalized

    payload["chat_template_kwargs"] = kwargs
    payload["reasoning_format"] = "deepseek"

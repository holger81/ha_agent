"""JSON-schema helpers for constrained LLM classifier outputs."""

from __future__ import annotations

from typing import Any

ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["chat", "email", "news", "action"],
        },
    },
    "required": ["route"],
    "additionalProperties": False,
}

COMPLEXITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "complexity": {
            "type": "string",
            "enum": ["simple", "single", "complex"],
        },
        "reason": {"type": "string"},
        "routes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["complexity"],
    "additionalProperties": False,
}

PREPASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["chat", "email", "news", "action"],
        },
        "complexity": {
            "type": "string",
            "enum": ["simple", "single", "complex"],
        },
        "reason": {"type": "string"},
        "skill_slug": {"type": "string"},
        "slot_bindings": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["route", "complexity"],
    "additionalProperties": False,
}

SLOT_BINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bindings": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["bindings"],
    "additionalProperties": False,
}

SKILL_SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_slugs": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["skill_slugs"],
    "additionalProperties": False,
}

VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string"},
        "skill_followed": {"type": "boolean"},
        "retry_hint": {"type": "string"},
    },
    "required": ["pass"],
    "additionalProperties": False,
}

PLAN_SUBTASKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "subgoal": {"type": "string"},
                    "route": {
                        "type": "string",
                        "enum": ["chat", "email", "news", "action"],
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["subgoal", "route"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}


def json_schema_format(
    name: str,
    schema: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-compatible response_format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }

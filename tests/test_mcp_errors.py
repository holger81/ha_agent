"""Unit tests for MCP friendly error messages."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _load():
    module_name = "ha_agent.mcp_errors"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    path = COMPONENT / "mcp_errors.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mcp_errors = _load()


def test_friendly_mcp_http_error_auth() -> None:
    """401 responses mention bearer token configuration."""
    message = mcp_errors.friendly_mcp_http_error(
        method="initialize",
        status=401,
        body='{"detail":"Not authenticated"}',
    )
    assert "bearer token" in message.lower()


def test_friendly_mcp_json_error_auth() -> None:
    """JSON-RPC auth errors mention bearer token configuration."""
    message = mcp_errors.friendly_mcp_json_error("Not authenticated")
    assert "bearer token" in message.lower()

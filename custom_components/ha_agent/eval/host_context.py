"""Enrich eval recommendations with server and model runtime context."""

from __future__ import annotations

from typing import Any

from ..llm_server import ServerCapabilities


def build_host_context(caps: ServerCapabilities) -> dict[str, Any]:
    """Summarize runtime signals available without SSH or host agents."""
    loaded_args: list[str] = []
    loaded_presets: list[str] = []
    for detail in caps.model_details:
        if detail.status != "loaded":
            continue
        status = detail.raw.get("status")
        if isinstance(status, dict):
            args = status.get("args")
            if isinstance(args, list):
                loaded_args = [str(item) for item in args]
            preset = status.get("preset")
            if isinstance(preset, str) and preset.strip():
                loaded_presets.append(preset.strip())
        break

    return {
        "router_role": caps.router_role,
        "max_instances": caps.max_instances,
        "models_autoload": caps.models_autoload,
        "loaded_model_count": len(caps.loaded_models),
        "loaded_model_args": loaded_args[:40],
        "loaded_model_presets": loaded_presets[:3],
        "metrics": dict(caps.metrics.parsed if caps.metrics else {}),
        "props_writable": caps.props_writable,
    }

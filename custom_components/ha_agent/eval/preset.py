"""Export llama.cpp preset snippets from eval recommendations."""

from __future__ import annotations

from typing import Any

_PRESET_KEY_ALIASES = {
    "parallel": "parallel",
    "n_parallel": "parallel",
    "slots": "parallel",
    "ctx-size": "ctx-size",
    "ctx_size": "ctx-size",
    "n_ctx": "ctx-size",
    "context": "ctx-size",
    "batch-size": "batch-size",
    "batch_size": "batch-size",
    "n_batch": "batch-size",
    "threads": "threads",
    "n_threads": "threads",
    "n-gpu-layers": "n-gpu-layers",
    "n_gpu_layers": "n-gpu-layers",
    "ngl": "n-gpu-layers",
    "cache-reuse": "cache-reuse",
    "cache_reuse": "cache-reuse",
}


def normalize_setting_key(setting: str) -> str:
    """Map eval setting names to llama.cpp preset keys."""
    cleaned = setting.strip().lower().replace(" ", "-")
    return _PRESET_KEY_ALIASES.get(cleaned, cleaned)


def recommendations_to_preset(
    recommendations: list[dict[str, Any]],
    *,
    title: str = "HA Agent eval",
) -> str:
    """Convert eval setting recommendations to a llama.cpp preset INI block."""
    lines = [f"# {title}", "# Paste into a llama.cpp preset file or server args"]
    seen: set[str] = set()
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        setting = str(item.get("setting") or "").strip()
        value = str(item.get("value") or "").strip()
        if not setting or not value:
            continue
        key = normalize_setting_key(setting)
        if key in seen:
            continue
        seen.add(key)
        reason = str(item.get("reason") or "").strip()
        if reason:
            lines.append(f"# {reason}")
        lines.append(f"{key} = {value}")
    if len(lines) == 2:
        lines.append("# No server settings were recommended.")
    return "\n".join(lines) + "\n"

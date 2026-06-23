"""Apply and verify llama.cpp server settings from eval recommendations."""

from __future__ import annotations

from typing import Any

from ..llm_server import ServerCapabilities
from .preset import normalize_setting_key


def _as_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def server_apply_mode(caps: ServerCapabilities) -> str:
    """Return how server settings can be applied: props or preset."""
    if caps.router_role == "router" or not caps.props_writable:
        return "preset"
    return "props"


def probe_setting_value(caps: ServerCapabilities, setting: str) -> int | None:
    """Read a comparable server value for a recommended setting name."""
    key = normalize_setting_key(setting)
    props = caps.props
    if key == "parallel":
        if props and props.total_slots is not None:
            return props.total_slots
        return caps.max_instances
    if key == "ctx-size":
        return props.n_ctx if props else None
    return None


def verify_settings_applied(
    before: ServerCapabilities,
    after: ServerCapabilities,
    settings: dict[str, str],
) -> dict[str, Any]:
    """Compare probe snapshots after attempting POST /props."""
    checks: list[dict[str, Any]] = []
    for setting, value in settings.items():
        expected = _as_int(value)
        before_val = probe_setting_value(before, setting)
        after_val = probe_setting_value(after, setting)
        verified = (
            expected is not None
            and after_val is not None
            and after_val == expected
        )
        checks.append(
            {
                "setting": setting,
                "expected": value,
                "before": before_val,
                "after": after_val,
                "verified": verified,
            }
        )
    verified_count = sum(1 for item in checks if item["verified"])
    return {
        "checks": checks,
        "verified_count": verified_count,
        "all_verified": verified_count == len(checks) if checks else True,
    }

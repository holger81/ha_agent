"""Config helper validation (IN-1)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_as_int():
    if "ha_agent.config_helpers" in sys.modules and hasattr(
        sys.modules["ha_agent.config_helpers"], "_as_int"
    ):
        return sys.modules["ha_agent.config_helpers"]._as_int

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    for name in ("const", "thinking"):
        mod_name = f"ha_agent.{name}"
        if mod_name in sys.modules:
            continue
        path = COMPONENT / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    if "homeassistant.config_entries" not in sys.modules:
        ce = types.ModuleType("homeassistant.config_entries")
        ce.ConfigEntry = object
        sys.modules["homeassistant.config_entries"] = ce

    path = COMPONENT / "config_helpers.py"
    spec = importlib.util.spec_from_file_location("ha_agent.config_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["ha_agent.config_helpers"] = module
    spec.loader.exec_module(module)
    return module._as_int


_as_int = _load_as_int()


def test_as_int_tolerates_bad_values() -> None:
    assert _as_int("12", 0) == 12
    assert _as_int(None, 7) == 7
    assert _as_int(True, 7) == 7
    assert _as_int("nope", 3) == 3
    assert _as_int(4.8, 0) == 4


def test_set_config_schema_present_in_source() -> None:
    """Guard that api/config validates updates with a voluptuous schema."""
    text = (COMPONENT / "api" / "config.py").read_text(encoding="utf-8")
    assert "_SET_CONFIG_SCHEMA = vol.Schema(" in text
    assert "vol.Range(min=1, max=64)" in text
    assert "extra=vol.PREVENT_EXTRA" in text

"""Unit tests for phase-3 model discovery."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


def _ensure_ha_stubs() -> None:
    if "homeassistant.core" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_core = types.ModuleType("homeassistant.core")

        def callback(func):
            return func

        ha_core.HomeAssistant = object
        ha_core.callback = callback
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.core"] = ha_core


def _load(name: str, path: Path):
    module_name = f"ha_agent.{name.replace('/', '.')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    _ensure_ha_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


llm_server = _load("llm_server", COMPONENT / "llm_server.py")
discover_models = _load(
    "eval.discover_models",
    COMPONENT / "eval" / "discover_models.py",
)
model_download = _load(
    "eval.model_download",
    COMPONENT / "eval" / "model_download.py",
)
discover_runner = _load(
    "eval.discover_runner",
    COMPONENT / "eval" / "discover_runner.py",
)
eval_models = _load("eval.models", COMPONENT / "eval" / "models.py")


def test_hf_download_url() -> None:
    url = model_download.hf_download_url("unsloth/gemma", "model/Q4_K_M.gguf")
    assert "huggingface.co" in url
    assert "gemma" in url
    assert "Q4_K_M.gguf" in url


def test_pick_quant_file_prefers_q4() -> None:
    picked = discover_models._pick_quant_file(
        ["model-f16.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
    )
    assert picked == "model-Q4_K_M.gguf"


def test_router_model_id_with_quant() -> None:
    model_id = discover_models._router_model_id(
        "unsloth/gemma-3-it-GGUF",
        "gemma-3-it-Q4_K_M.gguf",
    )
    assert model_id == "unsloth/gemma-3-it-GGUF:gemma-3-it-Q4_K_M"


def test_pending_approval_download() -> None:
    run = eval_models.DiscoverRun(
        id="1",
        entry_id="entry",
        status="awaiting_approval",
        started_at=0.0,
        progress={"phase": "awaiting_download_approval"},
    )
    state = eval_models.DiscoverRunState(run=run)
    assert discover_runner._pending_approval(state) == "download"


def test_manual_download_hint() -> None:
    hints = model_download.manual_download_hint(
        "unsloth/gemma-3-it-GGUF",
        "gemma-Q4_K_M.gguf",
    )
    assert "huggingface.co" in hints["hf_url"]


def test_discover_run_to_dict_includes_message() -> None:
    run = eval_models.DiscoverRun(
        id="1",
        entry_id="entry",
        status="running",
        started_at=0.0,
        progress={"phase": "discovering", "message": "Searching…"},
    )
    state = eval_models.DiscoverRunState(run=run)
    payload = discover_runner.discover_run_to_dict(state)
    assert payload["progress"]["message"] == "Searching…"

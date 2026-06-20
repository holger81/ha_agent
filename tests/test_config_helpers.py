"""Unit tests for config helpers (classifier backend)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_module(name: str):
    mod_name = f"ha_agent.{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    spec = importlib.util.spec_from_file_location(mod_name, COMPONENT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_config_helpers():
    if "ha_agent.config_helpers" in sys.modules:
        return sys.modules["ha_agent.config_helpers"]
    _load_module("const")
    _load_module("thinking")
    return _load_module("config_helpers")


class _Entry:
    entry_id = "test-entry"

    def __init__(self, data: dict) -> None:
        self.data = data


def test_classifier_backend_none_when_disabled():
    ch = _load_config_helpers()
    entry = _Entry({"llm_model": "chat-model"})
    assert ch.get_classifier_backend(entry) is None


def test_classifier_backend_none_when_no_model():
    ch = _load_config_helpers()
    entry = _Entry({"classifier_model_enabled": True})
    assert ch.get_classifier_backend(entry) is None


def test_classifier_backend_defaults_to_chat_base_url():
    ch = _load_config_helpers()
    entry = _Entry(
        {
            "classifier_model_enabled": True,
            "classifier_llm_model": "fast-classifier",
            "llm_base_url": "http://chat:9292/v1",
            "llm_api_key": "secret",
        }
    )
    backend = ch.get_classifier_backend(entry)
    assert backend is not None
    assert backend.model == "fast-classifier"
    assert backend.base_url == "http://chat:9292/v1"
    assert backend.api_key == "secret"
    assert backend.thinking_level == "off"


def test_classifier_backend_honours_custom_base_url():
    ch = _load_config_helpers()
    entry = _Entry(
        {
            "classifier_model_enabled": True,
            "classifier_llm_model": "fast-classifier",
            "classifier_llm_base_url": "http://classifier:8000/v1/",
        }
    )
    backend = ch.get_classifier_backend(entry)
    assert backend is not None
    assert backend.base_url == "http://classifier:8000/v1"


def test_router_config_includes_classifier_backend():
    ch = _load_config_helpers()
    entry = _Entry(
        {
            "classifier_model_enabled": True,
            "classifier_llm_model": "fast-classifier",
        }
    )
    router = ch.get_router_config(entry)
    assert router.classifier_backend is not None
    assert router.classifier_backend.model == "fast-classifier"

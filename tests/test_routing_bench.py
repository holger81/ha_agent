"""Tests for the route-classifier eval microbench."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _ensure_ha_stubs() -> None:
    if "homeassistant.exceptions" in sys.modules:
        return
    ha_pkg = types.ModuleType("homeassistant")
    ha_exc = types.ModuleType("homeassistant.exceptions")
    ha_core = types.ModuleType("homeassistant.core")
    ha_components = types.ModuleType("homeassistant.components")
    ha_conv = types.ModuleType("homeassistant.components.conversation")

    class HomeAssistantError(Exception):
        pass

    ha_core.HomeAssistant = object
    ha_exc.HomeAssistantError = HomeAssistantError
    sys.modules["homeassistant"] = ha_pkg
    sys.modules["homeassistant.exceptions"] = ha_exc
    sys.modules["homeassistant.core"] = ha_core
    sys.modules["homeassistant.components"] = ha_components
    sys.modules["homeassistant.components.conversation"] = ha_conv


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


# Load deps in dependency order for routing_bench.
_load("const", COMPONENT / "const.py")
config_helpers = _load("config_helpers", COMPONENT / "config_helpers.py")
_load("thinking", COMPONENT / "thinking.py")
_load("context", COMPONENT / "context.py")
_load("structured_output", COMPONENT / "structured_output.py")
_load("llm_client", COMPONENT / "llm_client.py")
router = _load("router", COMPONENT / "router.py")
_load("skills/models", COMPONENT / "skills" / "models.py")
_load("skills/selection", COMPONENT / "skills" / "selection.py")
eval_models = _load("eval/models", COMPONENT / "eval" / "models.py")
routing_bench = _load("eval/routing_bench", COMPONENT / "eval" / "routing_bench.py")


def _backend() -> object:
    return config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="classifier",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )


def _router_config() -> object:
    return config_helpers.RouterConfig(
        action_enabled=True,
        action_backend=_backend(),
    )


@pytest.mark.asyncio
async def test_routing_bench_soft_hint_email_without_llm() -> None:
    """Keyword fallback + soft-domain fill yields chat/email."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content=""))
    case = eval_models.EvalCase(
        id="t",
        task="routing",
        user_text="how many unread emails do I have",
        expected_route="chat",
        expected_domain_hint="email",
    )
    resolution = await routing_bench.run_routing_case(
        llm,
        _backend(),
        case,
        router_config=_router_config(),
        structured_output_enabled=False,
    )
    assert resolution.route == router.TaskRoute.CHAT
    assert resolution.domain_hint == "email"


@pytest.mark.asyncio
async def test_routing_bench_action_after_email_history() -> None:
    """Clear device control after an email thread still routes to action."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content='{"route": "action"}'))
    case = eval_models.EvalCase(
        id="t",
        task="routing",
        user_text="turn off the kitchen lights",
        history=[
            {"role": "user", "content": "how many unread emails do I have"},
            {"role": "assistant", "content": "You have 3 unread emails."},
        ],
        expected_route="action",
    )
    resolution = await routing_bench.run_routing_case(
        llm,
        _backend(),
        case,
        router_config=_router_config(),
    )
    assert resolution.route == router.TaskRoute.HA_ACTION
    assert resolution.domain_hint is None


@pytest.mark.asyncio
async def test_routing_bench_coerces_action_to_chat_for_soft_domain() -> None:
    """LLM action on an email ask is coerced to chat with email hint."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content='{"route": "action"}'))
    case = eval_models.EvalCase(
        id="t",
        task="routing",
        user_text="do I have new emails?",
        expected_route="chat",
        expected_domain_hint="email",
    )
    resolution = await routing_bench.run_routing_case(
        llm,
        _backend(),
        case,
        router_config=_router_config(),
    )
    assert resolution.route == router.TaskRoute.CHAT
    assert resolution.domain_hint == "email"

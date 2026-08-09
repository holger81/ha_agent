"""Unit tests for action/chat routing."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    deps = {
        "router": ["config_helpers", "context"],
        "config_helpers": ["const"],
        "context": [],
    }
    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    if name == "context":
        conv = types.ModuleType("homeassistant.components.conversation")
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.components"] = types.ModuleType(
            "homeassistant.components"
        )
        sys.modules["homeassistant.components.conversation"] = conv

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


router = _load("router")
config_helpers = _load("config_helpers")


def _router_config(*, enabled: bool, model: str | None = "action-model") -> object:
    action_backend = None
    if enabled and model:
        action_backend = config_helpers.LlmBackend(
            base_url="http://example/v1",
            model=model,
            api_key=None,
            max_tokens=256,
            temperature=0.1,
            timeout=30,
            thinking_level="off",
        )
    return config_helpers.RouterConfig(
        action_enabled=enabled,
        action_backend=action_backend,
    )


def test_classify_route_uses_action_for_device_commands() -> None:
    """Device actions route to the action backend when enabled."""
    route = router.classify_route(
        "turn off the dining room lights",
        [{"entity_id": "light.dining", "name": "Dining"}],
        _router_config(enabled=True),
    )
    assert route == router.TaskRoute.HA_ACTION


def test_classify_route_uses_chat_for_news() -> None:
    """News queries stay on chat; skills own the domain, not keyword routes."""
    decision = router.classify_route_with_detail(
        "what's the news?",
        [],
        _router_config(enabled=True),
    )
    assert decision.route == router.TaskRoute.CHAT
    assert decision.domain_hint is None
    assert decision.method == "default"


def test_classify_route_uses_chat_for_mail_queries() -> None:
    """Email queries stay on chat; skills own the domain, not keyword routes."""
    decision = router.classify_route_with_detail(
        "do I have new emails?",
        [],
        _router_config(enabled=True),
    )
    assert decision.route == router.TaskRoute.CHAT
    assert decision.domain_hint is None
    assert decision.method == "default"


def test_classify_route_uses_action_for_camera_snapshot() -> None:
    """Camera snapshot requests route to the action backend when enabled."""
    route = router.classify_route(
        "take a snapshot from my front door cam",
        [],
        _router_config(enabled=True),
    )
    assert route == router.TaskRoute.HA_ACTION


def test_classify_route_disabled_always_chat() -> None:
    """Routing falls back to chat when action routing is disabled."""
    route = router.classify_route(
        "turn off the lights",
        [],
        _router_config(enabled=False),
    )
    assert route == router.TaskRoute.CHAT


def test_classify_route_uses_action_keyword_override() -> None:
    """A custom action keyword selects the action route."""
    decision = router.classify_route_with_detail(
        "dim the lounge",
        [],
        _router_config(enabled=True),
        route_keywords={"action": ["dim"]},
    )
    assert decision.route == router.TaskRoute.HA_ACTION
    assert decision.method == "keyword"


def test_classify_route_ignores_retired_email_keyword_override() -> None:
    """Retired email keyword overrides no longer invent a domain hint."""
    decision = router.classify_route_with_detail(
        "any postbox updates?",
        [],
        _router_config(enabled=True),
        route_keywords={"email": ["postbox"]},
    )
    assert decision.route == router.TaskRoute.CHAT
    assert decision.domain_hint is None


def test_backend_for_route_returns_action_backend() -> None:
    """Action route resolves to the configured action backend."""
    chat = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="chat-model",
        api_key=None,
        max_tokens=512,
        temperature=0.3,
        timeout=30,
        thinking_level="off",
    )
    router_config = _router_config(enabled=True)
    backend = router.backend_for_route(
        router.TaskRoute.HA_ACTION,
        chat_backend=chat,
        router_config=router_config,
        prefer_action=True,
    )
    assert backend.model == "action-model"


def test_classify_route_news_follow_up_stays_default_chat() -> None:
    """Follow-ups no longer invent a news domain route from history alone."""
    history = [
        {"role": "user", "content": "what are todays news"},
        {
            "role": "assistant",
            "content": "California shooting at a library and World Cup headlines.",
        },
    ]
    decision = router.classify_route_with_detail(
        "what is this about the California shooting",
        [],
        _router_config(enabled=True),
        history=history,
    )
    assert decision.route == router.TaskRoute.CHAT
    assert decision.domain_hint is None
    assert decision.method == "default"


def test_classify_route_with_detail_action_keyword() -> None:
    """Action classification includes the matched keyword detail."""
    decision = router.classify_route_with_detail(
        "turn on the lights",
        [],
        _router_config(enabled=True),
    )
    assert decision.route == router.TaskRoute.HA_ACTION
    assert decision.method == "keyword"
    assert "turn on" in decision.detail.lower() or "action" in decision.summary


@pytest.mark.asyncio
async def test_resolve_route_with_classifier_uses_llm() -> None:
    """Route classifier LLM choice wins over keyword hints."""
    from unittest.mock import AsyncMock, MagicMock

    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content='{"route": "chat"}'))
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="classifier",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    resolution = await router.resolve_route_with_classifier(
        llm,
        backend,
        user_text="tell me a joke",
        exposed_entities=[],
        router_config=_router_config(enabled=True),
        history=[
            {"role": "user", "content": "what are todays news"},
            {"role": "assistant", "content": "Headlines..."},
        ],
    )
    assert resolution.route == router.TaskRoute.CHAT
    assert resolution.method == "llm"
    assert resolution.classifier_summary == "LLM → chat"
    llm.chat.assert_awaited_once()


def test_backend_for_route_action_and_chat_only() -> None:
    """backend_for_route only distinguishes action vs chat."""
    chat = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="chat-model",
        api_key=None,
        max_tokens=512,
        temperature=0.3,
        timeout=30,
        thinking_level="off",
    )
    router_config = _router_config(enabled=True)
    assert (
        router.backend_for_route(
            router.TaskRoute.HA_ACTION,
            chat_backend=chat,
            router_config=router_config,
            prefer_action=True,
        ).model
        == "action-model"
    )
    assert (
        router.backend_for_route(
            router.TaskRoute.CHAT,
            chat_backend=chat,
            router_config=router_config,
        ).model
        == "chat-model"
    )


def test_route_schema_rejects_email_news() -> None:
    """Classifier schema only allows chat|action."""
    from ha_agent.structured_output import ROUTE_SCHEMA

    assert ROUTE_SCHEMA["properties"]["route"]["enum"] == ["chat", "action"]

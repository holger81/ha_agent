"""Unit tests for action/chat routing."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = (
    Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"
)


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
    """News queries use the news route."""
    route = router.classify_route(
        "what's the news?",
        [],
        _router_config(enabled=True),
    )
    assert route == router.TaskRoute.NEWS


def test_classify_route_uses_email_for_mail_queries() -> None:
    """Email queries use the email route."""
    route = router.classify_route(
        "do I have new emails?",
        [],
        _router_config(enabled=True),
    )
    assert route == router.TaskRoute.EMAIL


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


def test_classify_route_uses_keyword_override() -> None:
    """A custom email keyword routes to email; defaults no longer apply."""
    route = router.classify_route(
        "any postbox updates?",
        [],
        _router_config(enabled=True),
        route_keywords={"email": ["postbox"]},
    )
    assert route == router.TaskRoute.EMAIL


def test_classify_route_override_does_not_match_default_keyword() -> None:
    """When overridden, the default keyword set is replaced, not merged."""
    route = router.classify_route(
        "do I have new emails?",
        [],
        _router_config(enabled=True),
        route_keywords={"email": ["postbox"]},
    )
    assert route == router.TaskRoute.CHAT


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


def test_classify_route_news_follow_up_after_briefing() -> None:
    """News detail questions stay on the news route."""
    history = [
        {"role": "user", "content": "what are todays news"},
        {
            "role": "assistant",
            "content": "California shooting at a library and World Cup headlines.",
        },
    ]
    route = router.classify_route(
        "what is this about the California shooting",
        [],
        _router_config(enabled=True),
        history=history,
    )
    assert route == router.TaskRoute.NEWS


def test_classify_route_with_detail_news_keyword() -> None:
    """News classification includes the matched keyword."""
    decision = router.classify_route_with_detail(
        "what are todays news",
        [],
        _router_config(enabled=True),
    )
    assert decision.route == router.TaskRoute.NEWS
    assert decision.method == "keyword"
    assert "news" in decision.detail.lower()
    assert "news" in decision.summary


def test_backend_for_route_returns_email_backend() -> None:
    chat = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="chat-model",
        api_key=None,
        max_tokens=512,
        temperature=0.3,
        timeout=30,
        thinking_level="off",
    )
    email = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="email-model",
        api_key=None,
        max_tokens=512,
        temperature=0.3,
        timeout=30,
        thinking_level="off",
    )
    router_config = config_helpers.RouterConfig(
        action_enabled=False,
        action_backend=None,
        email_backend=email,
    )
    backend = router.backend_for_route(
        router.TaskRoute.EMAIL,
        chat_backend=chat,
        router_config=router_config,
    )
    assert backend.model == "email-model"

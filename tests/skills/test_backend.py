"""Tests for skill-aware LLM backend resolution."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


def _load(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    deps = {
        "skills.backend": ["config_helpers", "router", "skills.models"],
        "config_helpers": ["const"],
        "router": ["config_helpers", "context"],
        "const": [],
        "context": [],
        "skills.models": [],
    }
    for dep in deps.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    if name.startswith("skills."):
        path = COMPONENT / "skills" / f"{name.split('.', 1)[1]}.py"
    else:
        path = COMPONENT / f"{name}.py"

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_backend_for_skill_uses_skill_model() -> None:
    backend_mod = _load("skills.backend")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    chat = config_helpers.LlmBackend(
        base_url="http://chat/v1",
        model="chat-model",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    skill = models.Skill(
        id="1",
        slug="news-briefing",
        title="News",
        description="",
        triggers=[],
        body="",
        tool_steps=[],
        route_scope="news",
        llm_model="skill-model",
        llm_base_url="http://skill/v1",
    )
    resolved = backend_mod.backend_for_skill(
        skill,
        route="chat",
        chat_backend=chat,
    )
    assert resolved.model == "skill-model"
    assert resolved.base_url == "http://skill/v1"


def test_backend_for_skill_legacy_email_fallback() -> None:
    backend_mod = _load("skills.backend")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    chat = config_helpers.LlmBackend(
        base_url="http://chat/v1",
        model="chat-model",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    email = config_helpers.LlmBackend(
        base_url="http://email/v1",
        model="email-model",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    skill = models.Skill(
        id="1",
        slug="check-and-read-unread-emails",
        title="Email",
        description="",
        triggers=[],
        body="",
        tool_steps=[],
        route_scope="email",
    )
    router_config = config_helpers.RouterConfig(
        action_enabled=False,
        action_backend=None,
        email_backend=email,
    )
    resolved = backend_mod.backend_for_skill(
        skill,
        route="chat",
        chat_backend=chat,
        router_config=router_config,
    )
    assert resolved.model == "email-model"


def test_backend_for_skill_inherits_chat() -> None:
    backend_mod = _load("skills.backend")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    chat = config_helpers.LlmBackend(
        base_url="http://chat/v1",
        model="chat-model",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    skill = models.Skill(
        id="1",
        slug="generic",
        title="Generic",
        description="",
        triggers=[],
        body="",
        tool_steps=[],
    )
    resolved = backend_mod.backend_for_skill(
        skill,
        route="chat",
        chat_backend=chat,
    )
    assert resolved.model == "chat-model"

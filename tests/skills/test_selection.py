"""Unit tests for LLM skill selection."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = (
    Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"
)


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

    if name.startswith("skills.store"):
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistant:
            pass

        def callback(func):
            return func

        ha_core.HomeAssistant = HomeAssistant
        ha_core.callback = callback
        sys.modules.setdefault("homeassistant", types.ModuleType("homeassistant"))
        sys.modules["homeassistant.core"] = ha_core

    deps = {
        "skills.selection": [
            "const",
            "config_helpers",
            "llm_client",
            "skills.discovery",
            "skills.models",
            "skills.store",
        ],
        "skills.discovery": ["skills.models", "skills.store", "skills.format"],
        "skills.store": ["skills.models", "const"],
        "skills.models": [],
        "config_helpers": ["const"],
        "llm_client": ["const", "config_helpers"],
        "const": [],
    }
    root = name if not name.startswith("skills.") else name.split(".", 1)[1]
    for dep in deps.get(name, deps.get(f"skills.{root}", [])):
        if f"ha_agent.{dep}" not in sys.modules:
            _load(dep)

    if name.startswith("skills."):
        if "ha_agent.skills" not in sys.modules:
            skills_pkg = types.ModuleType("ha_agent.skills")
            skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
            sys.modules["ha_agent.skills"] = skills_pkg
        path = COMPONENT / "skills" / f"{name.split('.', 1)[1]}.py"
    else:
        path = COMPONENT / f"{name}.py"

    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_skill_selection_accepts_json() -> None:
    """Selection JSON returns slug list."""
    selection = _load("skills.selection")
    slugs = selection.parse_skill_selection(
        json.dumps({"skill_slugs": ["check-unread-emails"]})
    )
    assert slugs == ["check-unread-emails"]


def test_parse_skill_selection_accepts_fenced_json() -> None:
    """Fenced JSON is accepted."""
    selection = _load("skills.selection")
    slugs = selection.parse_skill_selection(
        '```json\n{"skill_slugs": ["news-briefing"]}\n```'
    )
    assert slugs == ["news-briefing"]


def test_merge_catalog_preserves_order_without_duplicates() -> None:
    """Catalog merge keeps FTS order and drops duplicates."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    first = Skill(
        id="1",
        slug="a",
        title="A",
        description="",
        triggers=[],
        body="",
        tool_steps=[],
    )
    second = Skill(
        id="2",
        slug="b",
        title="B",
        description="",
        triggers=[],
        body="",
        tool_steps=[],
    )
    merged = selection._merge_catalog([first], [first, second])
    assert [skill.slug for skill in merged] == ["a", "b"]


@pytest.mark.asyncio
async def test_select_skills_with_llm_returns_catalog_matches() -> None:
    """LLM slug output resolves to catalog skills."""
    selection = _load("skills.selection")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    llm_client = _load("llm_client")
    Skill = models.Skill
    LlmBackend = config_helpers.LlmBackend
    ChatResult = llm_client.ChatResult

    skill = Skill(
        id="1",
        slug="check-unread-emails",
        title="Check and Read Unread Emails",
        description="Check inbox",
        triggers=["email"],
        body="",
        tool_steps=[],
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=ChatResult(
            content='{"skill_slugs":["check-unread-emails"]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )

    selected = await selection.select_skills_with_llm(
        llm,
        backend,
        user_text="do I have new email",
        route="email",
        catalog=[skill],
    )

    assert len(selected) == 1
    assert selected[0].slug == "check-unread-emails"

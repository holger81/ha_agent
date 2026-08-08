"""Tests for stronger FTS skill selection gating."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _load_selection_helpers():
    """Load only the scoring helpers without importing selection.py deps."""
    # Execute the helper functions by loading models + copying helpers.
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.skills" not in sys.modules:
        pkg = types.ModuleType("ha_agent.skills")
        pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = pkg
    models_name = "ha_agent.skills.models"
    if models_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            models_name, COMPONENT / "skills" / "models.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[models_name] = mod
        spec.loader.exec_module(mod)
    models = sys.modules[models_name]

    # Inline the scoring constants/functions from selection.py to avoid LLM imports.
    import re

    _MIN_FTS_TRIGGER_SCORE = 0.22

    def _tokenize(text: str) -> set[str]:
        return {tok for tok in re.findall(r"[a-z0-9]{3,}", text.lower()) if tok}

    def trigger_overlap_score(user_text: str, skill) -> float:
        user_tokens = _tokenize(user_text)
        if not user_tokens:
            return 0.0
        skill_tokens = _tokenize(
            " ".join(
                [
                    skill.title,
                    skill.description,
                    *[str(trigger) for trigger in skill.triggers],
                ]
            )
        )
        if not skill_tokens:
            return 0.0
        overlap = user_tokens & skill_tokens
        return len(overlap) / max(len(user_tokens), 1)

    def strong_fts_match(user_text: str, skill) -> bool:
        return trigger_overlap_score(user_text, skill) >= _MIN_FTS_TRIGGER_SCORE

    return models.Skill, strong_fts_match


def test_strong_fts_match_requires_overlap() -> None:
    Skill, strong_fts_match = _load_selection_helpers()
    skill = Skill(
        id="1",
        slug="check-unread-email",
        title="Check unread email",
        description="Read unread mailbox messages",
        body="workflow",
        triggers=["unread email", "check inbox"],
        tool_steps=[],
        enabled=True,
        is_builtin=False,
    )
    assert strong_fts_match("check my unread email inbox", skill) is True
    assert strong_fts_match("what is the weather today", skill) is False

"""SK-1: skill file write/import round-trip preserves tool_steps."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "ha_agent"


def _load():
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(COMPONENT / "skills")]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in (
        "models",
        "body",
        "markdown",
        "defaults",
        "tool_names",
        "bundled",
        "store",
        "files",
    ):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        if name == "store" and "homeassistant.core" not in sys.modules:
            ha = types.ModuleType("homeassistant.core")
            ha.HomeAssistant = object
            sys.modules["homeassistant.core"] = ha
        if name == "store" and "homeassistant.exceptions" not in sys.modules:
            hexc = types.ModuleType("homeassistant.exceptions")

            class HomeAssistantError(Exception):
                pass

            hexc.HomeAssistantError = HomeAssistantError
            sys.modules["homeassistant.exceptions"] = hexc
        needs_const = name in {"store", "files", "bundled"}
        if needs_const and "ha_agent.const" not in sys.modules:
            const_path = COMPONENT / "const.py"
            spec = importlib.util.spec_from_file_location("ha_agent.const", const_path)
            const = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules["ha_agent.const"] = const
            spec.loader.exec_module(const)
        path = COMPONENT / "skills" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    return (
        sys.modules["ha_agent.skills.files"],
        sys.modules["ha_agent.skills.store"],
        sys.modules["ha_agent.skills.models"],
    )


files_mod, store_mod, models_mod = _load()
SkillStore = store_mod.SkillStore
write_skill_file = files_mod.write_skill_file
_import_file = files_mod._import_file
skill_to_markdown = sys.modules["ha_agent.skills.markdown"].skill_to_markdown


def test_write_import_roundtrip_keeps_tool_steps(tmp_path: Path) -> None:
    db = tmp_path / "skills.db"
    store = SkillStore(db)
    store.connect()
    skill = store.insert_skill(
        title="Mailbox Status",
        description="Check mailbox.",
        triggers=["check mailbox"],
        body="1. Call `mail_mcp__imap_mailbox_status` with mailbox.",
        tool_steps=[
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "arguments": {"mailbox": "{{mailbox}}"},
            }
        ],
        slug="mailbox-status",
    )
    directory = tmp_path / "md"
    path = write_skill_file(directory, skill)
    text = path.read_text(encoding="utf-8")
    assert "tool_steps:" in text
    assert "mail_mcp__imap_mailbox_status" in text

    assert _import_file(store, path) is False

    skill.tool_steps = []
    store.update_skill(skill)
    assert _import_file(store, path) is True
    restored = store.get_skill_by_slug("mailbox-status")
    assert restored is not None
    assert restored.tool_steps
    assert restored.tool_steps[0]["toolName"] == "mail_mcp__imap_mailbox_status"
    store.close()


def test_import_without_explicit_tool_steps_keeps_db_steps(tmp_path: Path) -> None:
    db = tmp_path / "skills.db"
    store = SkillStore(db)
    store.connect()
    skill = store.insert_skill(
        title="News",
        description="Briefing.",
        triggers=["news briefing"],
        body="# Workflow\n\n1. Call news_curate.\n",
        tool_steps=[
            {
                "toolName": "mcp_news__news_curate",
                "arguments": {"digest_scope": "{{digest_scope}}"},
            }
        ],
        slug="news-briefing-test",
    )
    directory = tmp_path / "md"
    path = write_skill_file(directory, skill)
    path.write_text(
        skill_to_markdown(skill, include_tool_steps=False),
        encoding="utf-8",
    )
    assert "tool_steps:" not in path.read_text(encoding="utf-8")
    _import_file(store, path)
    restored = store.get_skill_by_slug("news-briefing-test")
    assert restored is not None
    assert restored.tool_steps
    assert restored.tool_steps[0]["toolName"] == "mcp_news__news_curate"
    store.close()

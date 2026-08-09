"""Unit tests for LLM skill selection."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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
            "context",
        ],
        "skills.discovery": ["skills.models", "skills.store", "skills.format"],
        "skills.store": ["skills.models", "const"],
        "skills.models": [],
        "config_helpers": ["const"],
        "llm_client": ["const", "config_helpers"],
        "const": [],
        "context": [],
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

    selected, raw = await selection.select_skills_with_llm(
        llm,
        backend,
        user_text="do I have new email",
        route="email",
        catalog=[skill],
    )

    assert len(selected) == 1
    assert selected[0].slug == "check-unread-emails"
    assert "check-unread-emails" in raw


@pytest.mark.asyncio
async def test_resolve_skips_llm_for_single_fts_match(monkeypatch) -> None:
    """A single FTS match is trusted without an extra LLM selection call."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    skill = Skill(
        id="a",
        slug="a",
        title="check inbox email",
        description="check inbox email",
        triggers=["email", "inbox"],
        body="imap mailbox",
        tool_steps=[],
        route_scope="email",
    )

    store = MagicMock()
    store.search.return_value = [MagicMock(id="a")]
    store.load_skills_by_ids.return_value = [skill]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "check email inbox",
        route="chat",
        domain_hint="email",
    )

    assert [item.slug for item in result.skills] == ["a"]
    assert result.method == "fts_only"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_short_follow_up_does_not_fts_pin_email_check_skill(
    monkeypatch,
) -> None:
    """'check again' must not unsupervised-pin an email skill sharing 'check'."""
    selection = _load("skills.selection")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    llm_client = _load("llm_client")
    Skill = models.Skill
    ChatResult = llm_client.ChatResult

    email_skill = Skill(
        id="email-1",
        slug="check-and-read-unread-emails",
        title="Check and Read Unread Emails",
        description="Check inbox",
        triggers=["check email", "check unread", "check inbox"],
        body="",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages"}],
        route_scope="email",
    )
    status_skill = Skill(
        id="status-1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Status lookup",
        triggers=["temperature", "status"],
        body="",
        tool_steps=[{"toolName": "home_assistant__ha_search"}],
        route_scope="action",
    )

    store = MagicMock()
    store.search.return_value = [MagicMock(id=email_skill.id)]
    store.load_skills_by_ids.return_value = [email_skill]
    store.list_enabled.return_value = [email_skill, status_skill]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    monkeypatch.setattr(
        selection,
        "_load_skill_candidates",
        MagicMock(return_value=([email_skill, status_skill], [email_skill])),
    )

    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=ChatResult(
            content='{"skill_slugs":["look-up-sensor-or-entity-status"]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        backend,
        "check again",
        route="chat",
        history=[
            {"role": "user", "content": "what is the temperature in Jonathans room"},
            {
                "role": "assistant",
                "content": "The temperature in Jonathan's bedroom is 22.9°C.",
            },
        ],
    )

    assert [s.slug for s in result.skills] == ["look-up-sensor-or-entity-status"]
    assert result.method == "llm"
    llm.chat.assert_called_once()
    payload = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert "recent_messages" in payload
    assert payload["user_text"] == "check again"


@pytest.mark.asyncio
async def test_news_domain_hint_does_not_select_email_only_skill(monkeypatch) -> None:
    """A news domain hint on chat must not pick an email-only skill."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="email-1",
        slug="check-unread-emails",
        title="Check and Read Unread Emails",
        description="Check the inbox and read unread email messages",
        triggers=["email", "inbox", "unread"],
        body="",
        tool_steps=[],
        route_scope="email",
    )

    # The store has only an email skill; neither the news query nor the news
    # domain hint matches it, so FTS returns nothing for both searches.
    store = MagicMock()
    store.list_enabled.return_value = [email_skill]
    store.search.return_value = []
    store.load_skills_by_ids.return_value = []

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "what are todays news",
        route="chat",
        domain_hint="news",
    )

    assert result.skills == []
    assert result.method in {"none", "skipped", "fts_only"}
    llm.chat.assert_not_called()


def test_filter_tool_steps_for_route_drops_email_on_news() -> None:
    """Email tool steps are not seeded into a news-route plan."""
    selection = _load("skills.selection")
    steps = selection.filter_tool_steps_for_route(
        [
            {"toolName": "mail_mcp__imap_search_messages", "arguments": {}},
            {"toolName": "mcp_news__news_curate", "arguments": {}},
        ],
        "news",
    )
    assert steps == [{"toolName": "mcp_news__news_curate", "arguments": {}}]


def test_skill_matches_route_rejects_email_on_news() -> None:
    """Email-only skills must not match a news domain hint on chat."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="1",
        slug="advanced-email",
        title="Advanced Email Management",
        description="Check inbox and read unread email messages",
        triggers=["email", "inbox"],
        body="Use imap tools for mailbox status.",
        tool_steps=[],
        route_scope="email",
    )

    assert (
        selection.skill_matches_route(email_skill, "chat", domain_hint="news") is False
    )
    assert (
        selection.skill_matches_route(email_skill, "chat", domain_hint="email") is True
    )


def test_skill_matches_route_rejects_ha_status_on_email_hint() -> None:
    """HA entity-lookup skills must not serve an email soft-domain ask."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    status = Skill(
        id="status-1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Status lookup",
        triggers=["temperature", "status", "look up"],
        body="",
        tool_steps=[{"toolName": "home_assistant__ha_search"}],
        route_scope="chat",
    )
    assert selection.skill_matches_route(status, "chat", domain_hint="email") is False
    assert selection.infer_soft_domain_hint("do I have new emails?") == "email"


@pytest.mark.asyncio
async def test_email_ask_does_not_select_ha_status_skill(monkeypatch) -> None:
    """Email questions must not pin the HA status skill without a domain_hint."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    status = Skill(
        id="status-1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Parameterized status lookup",
        triggers=[
            "what is the temperature in {{query}}",
            "status of {{query}}",
            "look up {{query}} sensor",
        ],
        body="",
        tool_steps=[{"toolName": "home_assistant__ha_search"}],
        route_scope="action",
    )
    email = Skill(
        id="email-1",
        slug="check-and-read-unread-emails",
        title="Check and Read Unread Emails",
        description="Check inbox",
        triggers=["check email", "new emails", "unread"],
        body="",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages"}],
        route_scope="email",
    )

    store = MagicMock()
    store.list_enabled.return_value = [status, email]
    store.search.return_value = [MagicMock(id=status.id), MagicMock(id=email.id)]
    store.load_skills_by_ids.side_effect = lambda ids: [
        skill for skill in (status, email) if skill.id in ids
    ]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "do I have new emails?",
        route="chat",
        domain_hint=None,
    )

    assert all(skill.slug != status.slug for skill in result.skills)
    if result.skills:
        assert result.skills[0].slug == email.slug


def test_skill_matches_route_rejects_email_tools_on_action() -> None:
    """Email tool_steps must not match the action route even without keywords."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="2",
        slug="mark-them-as-read",
        title="mark them as read",
        description="Workflow distilled from a successful override turn",
        triggers=["mark them as read"],
        body="",
        tool_steps=[
            {
                "toolName": "mail_mcp__imap_search_messages",
                "arguments": {"mailbox": "INBOX"},
            }
        ],
        route_scope="email",
    )

    assert selection.skill_matches_route(email_skill, "action") is False
    assert (
        selection.skill_matches_route(email_skill, "chat", domain_hint="email") is True
    )


@pytest.mark.parametrize(
    ("text", "history", "expected"),
    [
        (
            "mark them as read",
            [
                {"role": "user", "content": "do I have new emails"},
                {
                    "role": "assistant",
                    "content": "Yes, you have 2 new unread emails from Netflix.",
                },
            ],
            "email",
        ),
        (
            "please mark all of them as read",
            [
                {"role": "user", "content": "check my inbox"},
                {"role": "assistant", "content": "Three unread messages."},
            ],
            "email",
        ),
        (
            "read the first one",
            [
                {"role": "user", "content": "do I have new emails"},
                {"role": "assistant", "content": "Yes: 1) Netflix 2) Walmart"},
            ],
            "email",
        ),
        (
            "summarize that",
            [
                {"role": "user", "content": "give me the headlines"},
                {"role": "assistant", "content": "Here is today's news briefing."},
            ],
            "news",
        ),
        (
            "again please",
            [
                {"role": "user", "content": "any unread mail"},
                {"role": "assistant", "content": "No new emails."},
            ],
            "email",
        ),
        (
            "what about apple",
            [
                {"role": "user", "content": "how is the stock market today"},
                {"role": "assistant", "content": "Indexes are mixed."},
            ],
            "stock",
        ),
        (
            "compared to yesterday",
            [
                {"role": "user", "content": "how is the stock market today"},
                {"role": "assistant", "content": "Indexes are mixed."},
                {"role": "user", "content": "what about apple"},
                {"role": "assistant", "content": "Apple is up about 1%."},
            ],
            "stock",
        ),
        (
            "summarize that",
            [
                {"role": "user", "content": "give me the headlines"},
                {
                    "role": "assistant",
                    "content": "Top news: stock market mixed; local briefing next.",
                },
            ],
            "news",
        ),
        # New topics — do not inherit.
        (
            "turn them off",
            [
                {"role": "user", "content": "do I have new emails"},
                {"role": "assistant", "content": "Yes, two unread emails."},
            ],
            None,
        ),
        (
            "give me the headlines",
            [
                {"role": "user", "content": "do I have new emails"},
                {"role": "assistant", "content": "Yes, two unread emails."},
            ],
            "news",
        ),
        (
            "check my portfolio",
            [
                {"role": "user", "content": "do I have new emails"},
                {"role": "assistant", "content": "Yes, two unread emails."},
            ],
            "stock",
        ),
        (
            "what is the temperature in the kitchen",
            [
                {"role": "user", "content": "do I have new emails"},
                {"role": "assistant", "content": "Yes, two unread emails."},
            ],
            None,
        ),
        ("mark them as read", [], None),
        ("mark them as read", None, None),
    ],
)
def test_follow_up_inherits_soft_domain_from_history(
    text: str,
    history: list[dict[str, str]] | None,
    expected: str | None,
) -> None:
    """Follow-ups inherit the prior soft domain; new topics do not."""
    selection = _load("skills.selection")
    assert selection.infer_soft_domain_hint(text, history) == expected


def test_email_follow_up_keeps_email_skill_rejects_lights() -> None:
    """History-carried email hint keeps mail workflows and drops lights."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    history = [
        {"role": "user", "content": "do I have new emails"},
        {
            "role": "assistant",
            "content": "Yes, you have 2 new unread emails from Netflix.",
        },
    ]
    assert selection.infer_soft_domain_hint("mark them as read", history) == "email"

    email = models.Skill(
        id="1",
        slug="check-and-read-unread-emails",
        title="Check and read unread emails",
        description="Check inbox",
        triggers=["do I have new emails", "mark them as read"],
        body="# Email",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages", "arguments": {}}],
        route_scope="email",
    )
    lights = models.Skill(
        id="2",
        slug="turn-off-dining-room-lights",
        title="Turn off Dining Room Lights",
        description="Turn off the dining room lights",
        triggers=["turn off dining room lights", "mark them as read"],
        body="# Lights",
        tool_steps=[
            {"toolName": "home_assistant__ha_call_service", "arguments": {}},
        ],
        route_scope="action",
    )
    # History-carried email hint keeps the email skill and drops lights.
    assert (
        selection.skill_matches_route(
            email,
            "chat",
            domain_hint="email",
            user_text="mark them as read",
        )
        is True
    )
    assert (
        selection.skill_matches_route(
            lights,
            "chat",
            domain_hint="email",
            user_text="mark them as read",
        )
        is False
    )


@pytest.mark.asyncio
async def test_resolve_follow_up_keeps_email_catalog_not_lights(
    monkeypatch,
) -> None:
    """After an email turn, mark-them-as-read must not offer a lights skill."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    config_helpers = _load("config_helpers")
    llm_client = _load("llm_client")

    email = models.Skill(
        id="1",
        slug="check-and-read-unread-emails",
        title="Check and read unread emails",
        description="Check inbox and mark messages",
        triggers=["do I have new emails", "mark them as read"],
        body="# Email",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages", "arguments": {}}],
        route_scope="email",
    )
    lights = models.Skill(
        id="2",
        slug="turn-off-dining-room-lights",
        title="Turn off Dining Room Lights",
        description="Turn off dining room lights",
        triggers=["turn off dining room lights", "mark them as read"],
        body="# Lights",
        tool_steps=[
            {"toolName": "home_assistant__ha_call_service", "arguments": {}},
        ],
        route_scope="action",
    )

    store = MagicMock()
    store.search.return_value = [MagicMock(id="1"), MagicMock(id="2")]
    store.load_skills_by_ids.return_value = [email, lights]
    store.list_enabled.return_value = [email, lights]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))
    monkeypatch.setattr(
        selection,
        "_load_skill_candidates",
        MagicMock(return_value=([email, lights], [email, lights])),
    )

    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=llm_client.ChatResult(
            content='{"skill_slugs":["check-and-read-unread-emails"]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )
    history = [
        {"role": "user", "content": "do I have new emails"},
        {
            "role": "assistant",
            "content": "Yes, you have 2 new unread emails.",
        },
    ]

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        backend,
        "mark them as read",
        route="chat",
        history=history,
    )
    assert [skill.slug for skill in result.skills] == ["check-and-read-unread-emails"]
    assert all(skill.slug != "turn-off-dining-room-lights" for skill in result.skills)
    # If the classifier ran, lights must already have been filtered from its catalog.
    if llm.chat.await_args is not None:
        payload = json.loads(llm.chat.await_args.args[0][1]["content"])
        catalog_slugs = {item["slug"] for item in payload["skills"]}
        assert "turn-off-dining-room-lights" not in catalog_slugs


@pytest.mark.asyncio
async def test_single_fts_match_still_respects_route(monkeypatch) -> None:
    """A lone FTS hit is ignored when it conflicts with the active route."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="email-1",
        slug="advanced-email",
        title="Advanced Email Management",
        description="Check inbox and read unread email messages",
        triggers=["email", "inbox"],
        body="",
        tool_steps=[],
    )

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(
        selection,
        "get_skill_store",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        selection,
        "_load_skill_candidates",
        MagicMock(return_value=([email_skill], [email_skill])),
    )

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "what are todays news",
        route="news",
    )

    assert result.skills == []
    assert result.method == "none"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_does_not_auto_select_only_skill(monkeypatch) -> None:
    """A lone catalog skill must not attach to generic greetings."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="email-1",
        slug="advanced-email",
        title="Advanced Email Management",
        description="Check inbox and read unread email messages",
        triggers=["email", "inbox"],
        body="",
        tool_steps=[],
    )

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(
        selection,
        "get_skill_store",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        selection,
        "_load_skill_candidates",
        MagicMock(return_value=([email_skill], [])),
    )

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "hi",
        route="chat",
    )

    assert result.skills == []
    assert result.method == "skipped"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_joke_on_chat_route_skips_learned_skills(monkeypatch) -> None:
    """Casual chat must not attach email workflows via the skill classifier."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill

    email_skill = Skill(
        id="email-1",
        slug="check-and-read-unread-emails",
        title="Email Management",
        description="Check inbox",
        triggers=["email"],
        body="",
        tool_steps=[],
    )

    store = MagicMock()
    store.search.return_value = [MagicMock(id=email_skill.id)]
    store.load_skills_by_ids.return_value = [email_skill]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock()

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        MagicMock(),
        "tell me a joke",
        route="chat",
    )

    assert result.skills == []
    assert result.method == "skipped"
    llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_chat_route_weak_fts_hit_asks_llm_intent(monkeypatch) -> None:
    """Weak chat FTS must not pin unsupervised; fall through to LLM intent."""
    selection = _load("skills.selection")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    llm_client = _load("llm_client")
    Skill = models.Skill
    ChatResult = llm_client.ChatResult

    email_skill = Skill(
        id="email-1",
        slug="email-digest",
        title="Email digest",
        description="Summarize unread mail with a light touch",
        triggers=["email digest", "unread email summary"],
        body="",
        tool_steps=[],
    )

    store = MagicMock()
    store.search.return_value = [MagicMock(id=email_skill.id)]
    store.load_skills_by_ids.return_value = [email_skill]
    store.list_enabled.return_value = [email_skill]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=ChatResult(
            content='{"skill_slugs":[]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        backend,
        "turn off dining room light",
        route="chat",
    )

    assert result.skills == []
    assert result.method == "llm_empty"
    llm.chat.assert_called_once()


def test_dining_lights_skill_does_not_apply_to_temperature_query() -> None:
    """Control skills must not apply to unrelated status questions."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    skill = models.Skill(
        id="1",
        slug="turn-off-dining-room-lights",
        title="Turn off Dining Room Lights",
        description="Turn off the dining room lights",
        triggers=["turn off dining room lights", "dining lights off"],
        body="# Lights",
        tool_steps=[{"toolName": "HassTurnOff", "arguments": {}}],
        route_scope="action",
    )
    assert (
        selection.skill_applies_to_user_text(
            "what is the temperature in Jonathans room",
            skill,
        )
        is False
    )


def test_prefer_overlapping_catalog_filters_zero_overlap() -> None:
    selection = _load("skills.selection")
    models = _load("skills.models")
    Skill = models.Skill
    lights = Skill(
        id="1",
        slug="dining-lights",
        title="Turn off Dining Room Lights",
        description="Dining lights",
        triggers=["dining room lights"],
        body="",
        tool_steps=[],
    )
    temp = Skill(
        id="2",
        slug="room-temperature",
        title="Room temperature",
        description="Read room temperature",
        triggers=["temperature", "how warm"],
        body="",
        tool_steps=[],
    )
    preferred = selection._prefer_overlapping_catalog(
        "what is the temperature in Jonathans room",
        [lights, temp],
    )
    assert [s.slug for s in preferred] == ["room-temperature"]


@pytest.mark.asyncio
async def test_llm_intent_pick_trusted_despite_weak_lexical_overlap(
    monkeypatch,
) -> None:
    """Classifier intent picks apply even when title/triggers barely overlap."""
    selection = _load("skills.selection")
    config_helpers = _load("config_helpers")
    models = _load("skills.models")
    llm_client = _load("llm_client")
    Skill = models.Skill
    ChatResult = llm_client.ChatResult

    status = Skill(
        id="1",
        slug="look-up-sensor-or-entity-status",
        title="Look up sensor or entity status",
        description="Parameterized status lookup",
        triggers=[
            "status of {{query}}",
            "look up {{query}} sensor",
        ],
        body="",
        tool_steps=[{"toolName": "home_assistant__ha_search"}],
        route_scope="action",
    )

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(
        selection,
        "get_skill_store",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        selection,
        "_load_skill_candidates",
        MagicMock(return_value=([status], [status])),
    )

    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=ChatResult(
            content='{"skill_slugs":["look-up-sensor-or-entity-status"]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )

    # Force LLM path: make FTS match weak so it is not pinned early.
    monkeypatch.setattr(selection, "_strong_fts_match", lambda *_a, **_k: False)
    assert (
        selection.skill_applies_to_user_text(
            "what is the temperature in the great room",
            status,
        )
        is False
    )

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        backend,
        "what is the temperature in the great room",
        route="action",
    )
    assert [s.slug for s in result.skills] == ["look-up-sensor-or-entity-status"]
    assert result.method == "llm"
    llm.chat.assert_called_once()


def test_specialized_skill_needs_domain_words_in_the_ask() -> None:
    """An email workflow is not eligible for an unrelated chat ask."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    email = models.Skill(
        id="1",
        slug="check-unread-emails",
        title="Check and read unread emails",
        description="Summarize unread inbox messages",
        triggers=["do I have new emails"],
        body="# Email",
        tool_steps=[{"toolName": "mail_mcp__imap_search_messages", "arguments": {}}],
        route_scope="email",
    )

    assert (
        selection.skill_matches_route(
            email,
            "chat",
            user_text="what is the temperature in Jonathans room",
        )
        is False
    )
    assert (
        selection.skill_matches_route(
            email,
            "chat",
            user_text="did I get any mail today",
        )
        is True
    )


@pytest.mark.asyncio
async def test_state_changing_skill_is_not_pinned_on_a_status_question(
    monkeypatch,
) -> None:
    """Keyword overlap must not run a control workflow for a status ask."""
    selection = _load("skills.selection")
    models = _load("skills.models")
    config_helpers = _load("config_helpers")
    llm_client = _load("llm_client")
    lock = models.Skill(
        id="1",
        slug="lock-doors",
        title="Lock or unlock doors",
        description="Lock and unlock door locks",
        triggers=["lock the front door", "unlock the front door"],
        body="# Locks",
        tool_steps=[
            {"toolName": "home_assistant__ha_call_service", "arguments": {}},
        ],
        route_scope="action",
    )
    assert selection.skill_changes_state(lock) is True

    store = MagicMock()
    store.search.return_value = [MagicMock(id="1")]
    store.load_skills_by_ids.return_value = [lock]
    store.list_enabled.return_value = [lock]

    async def _executor(func):
        return func()

    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    monkeypatch.setattr(selection, "get_skill_store", MagicMock(return_value=store))

    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=llm_client.ChatResult(
            content='{"skill_slugs":[]}',
            tool_calls=[],
            assistant_message={},
        )
    )
    backend = config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.1,
        timeout=30,
        thinking_level="off",
    )

    result = await selection.resolve_skills_for_turn(
        hass,
        "entry",
        llm,
        backend,
        "is the front door locked",
        route="chat",
    )
    assert result.skills == []


def _skill(models, slug: str, *, tools: list[str], scope: str | None = None):
    return models.Skill(
        id="1",
        slug=slug,
        title=slug.replace("-", " "),
        description=slug.replace("-", " "),
        triggers=[slug.replace("-", " ")],
        body=f"# {slug}",
        tool_steps=[{"toolName": name, "arguments": {}} for name in tools],
        route_scope=scope,
    )


@pytest.mark.parametrize(
    ("tool", "changes_state"),
    [
        # Home Assistant intent tools carry the verb in camelCase.
        ("HassTurnOn", True),
        ("HassLightSet", True),
        ("HassMediaPause", True),
        ("GetLiveContext", False),
        # Tool families this repo has never shipped must classify too.
        ("calendar_mcp__calendar_create_event", True),
        ("vacuum_mcp__vacuum_start_cleaning", True),
        ("todo_mcp__todo_add_item", True),
        ("weather_mcp__weather_get_forecast", False),
        ("mail_mcp__imap_search_messages", False),
        ("mcp_news__news_curate", False),
    ],
)
def test_state_change_detection_generalizes_past_ha_tools(
    tool: str, changes_state: bool
) -> None:
    """Effect comes from the tool's verb, not a list of known HA tools."""
    selection = _load("skills.selection")
    models = _load("skills.models")

    skill = _skill(models, "some-workflow", tools=[tool])
    assert selection.skill_changes_state(skill) is changes_state


def test_unknown_domain_skill_needs_no_marker_entry() -> None:
    """A brand-new domain works without adding it to any marker table."""
    selection = _load("skills.selection")
    models = _load("skills.models")

    # "calendar" is not a known soft domain, so the domain gate stays out of the
    # way: a read-only calendar skill is eligible for a calendar question.
    reader = _skill(
        models,
        "read-calendar",
        tools=["calendar_mcp__calendar_list_events"],
        scope="calendar",
    )
    assert (
        selection.skill_matches_route(
            reader, "chat", user_text="whats on my calendar tomorrow"
        )
        is True
    )
    # The read-only-question rule still applies to it, with no new special case.
    writer = _skill(
        models,
        "add-calendar-event",
        tools=["calendar_mcp__calendar_create_event"],
        scope="calendar",
    )
    assert (
        selection.skill_matches_route(
            writer, "chat", user_text="whats on my calendar tomorrow"
        )
        is False
    )
    assert (
        selection.skill_matches_route(
            writer, "chat", user_text="add a dentist appointment on friday"
        )
        is True
    )

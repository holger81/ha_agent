"""Phase 4 exit-criteria scenario tests for the agent loop."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"

MODULE_DEPS: dict[str, list[str]] = {
    "config_helpers": ["const"],
    "llm_client": ["const", "config_helpers"],
    "mcp_client": ["config_helpers"],
    "context": [],
    "tools": ["llm_client", "mcp_client"],
    "memory": ["const"],
    "agent": [
        "const",
        "config_helpers",
        "llm_client",
        "mcp_client",
        "context",
        "tools",
        "memory",
        "router",
        "status",
        "embedded_tools",
        "mcp_session",
        "mcp_errors",
    ],
    "router": ["config_helpers", "context"],
    "status": ["const"],
}


def _ensure_ha_stubs() -> None:
    if "homeassistant.exceptions" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_exc = types.ModuleType("homeassistant.exceptions")
        ha_core = types.ModuleType("homeassistant.core")

        class HomeAssistantError(Exception):
            pass

        def callback(func):
            return func

        ha_core.HomeAssistant = object
        ha_core.callback = callback

        class ServiceCall:
            def __init__(self, data: dict | None = None) -> None:
                self.data = data or {}

        ha_core.ServiceCall = ServiceCall
        ha_exc.HomeAssistantError = HomeAssistantError
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.exceptions"] = ha_exc
        sys.modules["homeassistant.core"] = ha_core


def _load_skills_modules() -> None:
    skills_path = COMPONENT / "skills"
    if "ha_agent.skills" not in sys.modules:
        skills_pkg = types.ModuleType("ha_agent.skills")
        skills_pkg.__path__ = [str(skills_path)]  # type: ignore[attr-defined]
        sys.modules["ha_agent.skills"] = skills_pkg

    for name in (
        "models",
        "store",
        "format",
        "discovery",
        "runtime",
        "creator",
        "evaluator",
        "commands",
        "defaults",
        "repair",
        "files",
    ):
        mod_name = f"ha_agent.skills.{name}"
        if mod_name in sys.modules:
            continue
        path = skills_path / f"{name}.py"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


def _load_module(name: str):
    module_name = f"ha_agent.{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package

    _ensure_ha_stubs()

    for dep in MODULE_DEPS.get(name, []):
        if f"ha_agent.{dep}" not in sys.modules:
            _load_module(dep)

    if name == "agent":
        _load_skills_modules()

    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


agent_mod = _load_module("agent")
config_helpers = _load_module("config_helpers")
llm_client = _load_module("llm_client")


def _backend() -> config_helpers.LlmBackend:
    return config_helpers.LlmBackend(
        base_url="http://example/v1",
        model="test",
        api_key=None,
        max_tokens=128,
        temperature=0.2,
        timeout=30,
        thinking_level="off",
    )


def _content(chunks: list) -> list[str]:
    return [chunk.content for chunk in chunks if chunk.content]


def _agent_config(*, streaming: bool = False) -> config_helpers.AgentConfig:
    return config_helpers.AgentConfig(
        system_prompt="Test agent",
        tool_instructions="Use tools",
        max_iterations=6,
        history_turns=4,
        enable_streaming=streaming,
        show_reasoning_in_chat=False,
    )


def _router_config() -> config_helpers.RouterConfig:
    return config_helpers.RouterConfig(action_enabled=False, action_backend=None)


def _skills_config() -> config_helpers.SkillsConfig:
    return config_helpers.SkillsConfig(
        learning_enabled=False,
        auto_save=False,
        use_enabled=False,
        max_inject=3,
    )


def _hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.async_add_executor_job = AsyncMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )
    hass.async_create_task = MagicMock()
    return hass


def _tool_call(
    name: str,
    arguments: dict,
    *,
    call_id: str = "call_1",
) -> llm_client.ToolCall:
    return llm_client.ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )


def _chat_result(
    *,
    content: str | None = None,
    tool_calls: list[llm_client.ToolCall] | None = None,
) -> llm_client.ChatResult:
    tool_calls = tool_calls or []
    assistant_message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ]
    return llm_client.ChatResult(
        content=content,
        tool_calls=tool_calls,
        assistant_message=assistant_message,
    )


def _route_classifier_chat(route: str = "chat") -> llm_client.ChatResult:
    return _chat_result(content=json.dumps({"route": route}))


def _chat_side_effect_with_route(route: str, *results: llm_client.ChatResult):
    return [_route_classifier_chat(route), *results]


@pytest.mark.asyncio
async def test_phase4_light_off_with_exposed_entity() -> None:
    """Exposed light can be turned off with one MCP service call."""
    service_call = _tool_call(
        "callTool",
        {
            "toolName": "home_assistant__ha_call_service",
            "arguments": {
                "domain": "light",
                "service": "turn_off",
                "entity_id": "light.dining",
            },
        },
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=_chat_side_effect_with_route(
            "chat",
            _chat_result(tool_calls=[service_call]),
            _chat_result(content="The dining room lights are off."),
        )
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"success": true}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            skills_config=_skills_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-light",
            user_text="turn off the dining room lights",
            exposed_entities=[
                {
                    "entity_id": "light.dining",
                    "name": "Dining",
                    "state": "on",
                    "area_name": "Dining room",
                }
            ],
        )
    ]

    assert _content(chunks) == ["The dining room lights are off."]
    mock_mcp.call_tool.assert_awaited_once()
    call_args = mock_mcp.call_tool.await_args.args
    assert call_args[0] == "callTool"
    assert call_args[1]["toolName"] == "home_assistant__ha_call_service"


@pytest.mark.asyncio
async def test_phase4_cover_open_without_exposed_entity() -> None:
    """Cover actions can search smart-home tools then call open_cover."""
    search_call = _tool_call(
        "searchToolsForDomain",
        {"domain": "smart-home", "query": "open cover"},
        call_id="call_search",
    )
    open_call = _tool_call(
        "callTool",
        {
            "toolName": "home_assistant__ha_call_service",
            "arguments": {
                "domain": "cover",
                "service": "open_cover",
                "entity_id": "cover.patio",
            },
        },
        call_id="call_open",
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=_chat_side_effect_with_route(
            "chat",
            _chat_result(tool_calls=[search_call]),
            _chat_result(tool_calls=[open_call]),
            _chat_result(content="I opened the patio cover."),
        )
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(
        side_effect=[
            '{"tools":[{"toolName":"home_assistant__ha_call_service"}]}',
            '{"success": true}',
        ]
    )
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            skills_config=_skills_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-cover",
            user_text="open the patio cover",
            exposed_entities=[],
        )
    ]

    assert _content(chunks) == ["I opened the patio cover."]
    assert mock_mcp.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_phase4_news_query_uses_mcp_tool() -> None:
    """News questions execute MCP news tools before answering."""
    news_call = _tool_call(
        "callTool",
        {"toolName": "mcp_news__news_curate", "arguments": {"limit": 5}},
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=_chat_side_effect_with_route(
            "news",
            _chat_result(tool_calls=[news_call]),
            _chat_result(content="Here are today's headlines."),
        )
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"headlines":["Example headline"]}')
    mock_mcp.get_session_prompt = AsyncMock(
        return_value="MCP SERVER INSTRUCTIONS:\nUse domain news."
    )
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            skills_config=_skills_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-news",
            user_text="What's the news?",
            exposed_entities=[],
        )
    ]

    assert _content(chunks) == ["Here are today's headlines."]
    mock_mcp.call_tool.assert_awaited_once()
    assert mock_mcp.call_tool.await_args.args[1]["toolName"] == "mcp_news__news_curate"


@pytest.mark.asyncio
async def test_phase4_email_unread_count_uses_mcp_tool() -> None:
    """Email questions execute MCP mail tools before answering."""
    mail_call = _tool_call(
        "callTool",
        {
            "toolName": "mail_mcp__imap_search_messages",
            "arguments": {"mailbox": "INBOX", "unread_only": True},
        },
    )
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=_chat_side_effect_with_route(
            "email",
            _chat_result(tool_calls=[mail_call]),
            _chat_result(content="You have 3 unread emails."),
        )
    )
    mock_mcp = MagicMock()
    mock_mcp.call_tool = AsyncMock(return_value='{"count": 3}')
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    chunks = [
        chunk
        async for chunk in agent_mod.run_agent(
            _hass(),
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=_backend(),
            agent_config=_agent_config(),
            router_config=_router_config(),
            skills_config=_skills_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-email",
            user_text="how many unread emails do I have",
            exposed_entities=[],
        )
    ]

    assert _content(chunks) == ["You have 3 unread emails."]
    mock_mcp.call_tool.assert_awaited_once()
    assert (
        mock_mcp.call_tool.await_args.args[1]["toolName"]
        == "mail_mcp__imap_search_messages"
    )


@pytest.mark.asyncio
async def test_phase4_conversation_memory_across_turns() -> None:
    """Second turn includes prior user and assistant messages."""
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(
        side_effect=[
            _route_classifier_chat("chat"),
            _chat_result(content="Sunny today."),
            _route_classifier_chat("chat"),
            _chat_result(content="Still sunny."),
        ]
    )
    mock_mcp = MagicMock()
    mock_mcp.get_session_prompt = AsyncMock(return_value="")
    mock_mcp.get_llm_tools = AsyncMock(return_value=[])

    hass = _hass()
    backend = _backend()
    agent_config = _agent_config()

    async def _run(user_text: str) -> None:
        async for _chunk in agent_mod.run_agent(
            hass,
            llm=mock_llm,
            mcp_client=mock_mcp,
            backend=backend,
            agent_config=agent_config,
            router_config=_router_config(),
            skills_config=_skills_config(),
            entry_id="phase4-entry",
            conversation_id="phase4-memory",
            user_text=user_text,
            exposed_entities=[],
        ):
            pass

    await _run("what is the weather")
    await _run("and tomorrow")

    second_messages = mock_llm.chat.await_args_list[3].args[0]
    roles = [message["role"] for message in second_messages]
    contents = [message.get("content", "") for message in second_messages]

    assert roles.count("user") == 2
    assert roles.count("assistant") == 1
    assert "what is the weather" in contents
    assert "Sunny today." in contents
    assert second_messages[-1]["content"] == "and tomorrow"


def _skill_store(tmp_path: Path):
    store_mod = sys.modules["ha_agent.skills.store"]
    store = store_mod.SkillStore(tmp_path / "skills.db")
    store.connect()
    return store


@pytest.mark.asyncio
async def test_phase4_email_param_error_repairs_skill(tmp_path) -> None:
    """Mailbox param errors during email skill use trigger auto-repair."""
    from unittest.mock import patch

    _load_skills_modules()
    store = _skill_store(tmp_path)
    models_mod = sys.modules["ha_agent.skills.models"]
    repair_mod = sys.modules["ha_agent.skills.repair"]
    repair_mod._last_repair_at.clear()

    skill = store.insert_skill(
        title="Email Management",
        description="Check email inbox.",
        triggers=["check email", "any new emails"],
        body="Check mailbox status.",
        tool_steps=[{"toolName": "mail_mcp__imap_mailbox_status"}],
        route_scope="email",
    )
    hass = _hass()
    hass.data = {"ha_agent": {"skill_stores": {"phase4-entry": store}}}

    trace = models_mod.TurnTrace(
        user_text="any new emails?",
        history_len=0,
        route="email",
        tool_calls=[
            {
                "toolName": "mail_mcp__imap_mailbox_status",
                "arguments": {},
                "succeeded": False,
                "error_kind": "param",
                "missing_fields": ["mailbox"],
                "error": "Tool error: missing field 'mailbox'",
            }
        ],
        assistant_text="You have 2 unread emails.",
        iterations=2,
    )

    with patch.object(repair_mod, "mirror_skill_to_file"):
        suffix, meta_patch = await agent_mod._post_turn_skills(
            hass,
            entry_id="phase4-entry",
            llm=MagicMock(),
            backend=_backend(),
            observer_backend=_backend(),
            skills_config=config_helpers.SkillsConfig(
                learning_enabled=False,
                auto_save=False,
                use_enabled=True,
                max_inject=3,
            ),
            trace=trace,
            history=[],
            matched_skills=[skill],
        )

    assert meta_patch is not None
    assert meta_patch["skill_update"]["title"] == "Email Management"
    assert meta_patch["skill_repair"]["issue_kind"] == "missing_param"
    assert "mailbox" in meta_patch["skill_repair"]["fields"]
    assert "Updated skill" in suffix

    updated = store.get_skill(skill.id)
    assert updated is not None
    assert updated.version > 1
    assert any(
        (step.get("arguments") or {}).get("mailbox") == "{{mailbox}}"
        for step in updated.tool_steps
    )


@pytest.mark.asyncio
async def test_override_learning_respects_auto_save_false(tmp_path) -> None:
    """Override learning queues pending when auto_save is off."""
    from unittest.mock import patch

    _load_skills_modules()
    store = _skill_store(tmp_path)
    models_mod = sys.modules["ha_agent.skills.models"]
    observer_mod = sys.modules.get("ha_agent.skills.observer")
    if observer_mod is None:
        path = COMPONENT / "skills" / "observer.py"
        spec = importlib.util.spec_from_file_location("ha_agent.skills.observer", path)
        assert spec is not None and spec.loader is not None
        observer_mod = importlib.util.module_from_spec(spec)
        sys.modules["ha_agent.skills.observer"] = observer_mod
        spec.loader.exec_module(observer_mod)

    parent = store.insert_skill(
        title="Turn off lights",
        description="Turn off lights via MCP.",
        triggers=["turn off lights"],
        body="Use `home_assistant__ha_call_service`.",
        tool_steps=[
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "{{entity_id}}",
                },
            }
        ],
        route_scope="action",
    )
    draft = models_mod.SkillDraft(
        title="Turn off dining room light",
        description="Turn off dining room lights via MCP.",
        triggers=["turn off dining room light"],
        body="Use `home_assistant__ha_call_service`.",
        tool_steps=[
            {
                "toolName": "home_assistant__ha_call_service",
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.dining_room",
                },
            }
        ],
        route_scope="action",
    )
    observed = observer_mod.SkillObserverResult(
        learn=True,
        reason="New dining-room workflow",
        draft=draft,
        update_parent=True,
    )
    trace = models_mod.TurnTrace(
        user_text="turn off dining room light",
        history_len=0,
        route="action",
        conversation_id="conv-override",
        skill_plan_override=True,
        outcome="success",
        tool_calls=[
            {
                "toolName": "home_assistant__ha_call_service",
                "succeeded": True,
                "arguments": {
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.dining_room",
                },
            }
        ],
        controlled_entity_ids=["light.dining_room"],
        assistant_text="Done.",
        iterations=1,
    )
    hass = _hass()
    hass.data = {"ha_agent": {"skill_stores": {"phase4-entry": store}}}
    queued: list[dict] = []

    def _queue(*args, **kwargs):
        queued.append({"args": args, "kwargs": kwargs})

    with (
        patch.object(
            agent_mod,
            "observe_skill_override",
            AsyncMock(return_value=observed),
        ),
        patch.object(agent_mod, "queue_pending_save", side_effect=_queue),
        patch.object(agent_mod, "persist_skill_draft", AsyncMock()) as persist,
        patch.object(
            agent_mod,
            "auto_repair_skill",
            MagicMock(return_value=None),
        ),
        patch.object(agent_mod, "detect_repairable_issues", MagicMock(return_value=[])),
    ):
        suffix, _meta = await agent_mod._post_turn_skills(
            hass,
            entry_id="phase4-entry",
            llm=MagicMock(),
            backend=_backend(),
            observer_backend=_backend(),
            skills_config=config_helpers.SkillsConfig(
                learning_enabled=True,
                auto_save=False,
                use_enabled=True,
                max_inject=3,
            ),
            trace=trace,
            history=[],
            matched_skills=[parent],
        )

    persist.assert_not_called()
    assert queued
    assert queued[0]["kwargs"].get("update_skill_id") == parent.id
    assert "Reply yes" in suffix


def test_merge_learned_skills_for_post_turn_deduplicates() -> None:
    """Orchestrated and top-level learned skills merge without duplicates."""
    models_mod = sys.modules["ha_agent.skills.models"]
    Skill = models_mod.Skill

    def _skill(skill_id: str, *, builtin: bool = False) -> Skill:
        return Skill(
            id=skill_id,
            slug=skill_id,
            title=skill_id,
            description="desc",
            triggers=["t"],
            body="body",
            tool_steps=[],
            is_builtin=builtin,
        )

    merged = agent_mod._merge_learned_skills_for_post_turn(
        [_skill("builtin", builtin=True), _skill("learned-a")],
        [_skill("learned-a"), _skill("learned-b")],
    )
    assert [item.id for item in merged] == ["learned-a", "learned-b"]

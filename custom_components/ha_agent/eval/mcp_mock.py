"""Deterministic MCP client for eval benchmark runs."""

from __future__ import annotations

from typing import Any

from ..mcp_session import FALLBACK_MCP_TOOLS, mcp_tools_to_openai_schemas


class EvalMcpClient:
    """Scripted MCP responses so eval scores depend on the LLM, not live HA."""

    def __init__(
        self,
        *,
        session_prompt: str = "",
        responses: list[str] | None = None,
    ) -> None:
        self._session_prompt = session_prompt
        self._responses = list(responses or [])
        self._call_index = 0

    async def get_session_prompt(self) -> str:
        return self._session_prompt

    async def get_llm_tools(self) -> list[dict[str, Any]]:
        return mcp_tools_to_openai_schemas(FALLBACK_MCP_TOOLS)

    async def call_tool(self, _tool_name: str, _tool_args: dict[str, Any]) -> str:
        if self._call_index >= len(self._responses):
            return '{"success": true}'
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

"""Per-conversation history for multi-turn agent sessions."""

from __future__ import annotations


class ConversationMemory:
    """In-memory store for conversation turns."""

    def __init__(self) -> None:
        """Initialize an empty memory store."""
        self._store: dict[str, list[dict[str, str]]] = {}

    def get_history(
        self,
        conversation_id: str | None,
        *,
        max_turns: int,
    ) -> list[dict[str, str]]:
        """Return stored history for a conversation."""
        if not conversation_id or max_turns <= 0:
            return []
        history = self._store.get(conversation_id, [])
        max_messages = max_turns * 2
        if len(history) > max_messages:
            return history[-max_messages:]
        return list(history)

    def append_turn(
        self,
        conversation_id: str | None,
        user_text: str,
        assistant_text: str,
        *,
        max_turns: int,
    ) -> None:
        """Append a user/assistant turn to memory."""
        if not conversation_id or max_turns <= 0:
            return
        if not user_text.strip() and not assistant_text.strip():
            return

        history = self._store.setdefault(conversation_id, [])
        if user_text.strip():
            history.append({"role": "user", "content": user_text.strip()})
        if assistant_text.strip():
            history.append({"role": "assistant", "content": assistant_text.strip()})

        max_messages = max_turns * 2
        if len(history) > max_messages:
            self._store[conversation_id] = history[-max_messages:]

    def clear_conversation(self, conversation_id: str | None) -> None:
        """Clear stored history for a conversation."""
        if not conversation_id:
            return
        self._store.pop(conversation_id, None)

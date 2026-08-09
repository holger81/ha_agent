"""Apply durable memory writes during or after an agent turn."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from ..identity.models import ResolvedIdentity, UserKind
from .extract import ExtractedMemory, extract_memory_writes
from .inject import should_include_user_memory
from .intent import MemoryIntent, MemoryIntentKind, detect_memory_intent
from .store import get_persistent_memory_store


def apply_memory_defaults_to_slots(
    slot_bindings: dict[str, str],
    memory_values: dict[str, Any],
) -> dict[str, str]:
    """Fill empty slots from durable memory defaults."""
    updated = dict(slot_bindings)
    mailbox = memory_values.get("email.default_mailbox")
    if mailbox and not updated.get("mailbox"):
        updated["mailbox"] = str(mailbox)
    digest = memory_values.get("news.digest_scope")
    if digest and not updated.get("digest_scope"):
        updated["digest_scope"] = str(digest)
    return updated


async def async_handle_memory_intent(
    hass: HomeAssistant,
    entry_id: str,
    *,
    user_text: str,
    identity: ResolvedIdentity,
    controlled_entity_ids: list[str] | None = None,
    route: str | None = None,
    force_system: bool = False,
) -> str | None:
    """Handle remember/prefer/forget short-circuit turns.

    Returns an assistant reply when the turn is fully handled, else None.
    """
    intent = detect_memory_intent(user_text)
    if intent.kind == MemoryIntentKind.NONE:
        return None
    if intent.kind == MemoryIntentKind.CLARIFY:
        return intent.clarify_message

    store = get_persistent_memory_store(hass, entry_id)
    use_user = should_include_user_memory(identity) and not force_system

    if intent.kind == MemoryIntentKind.FORGET:
        return await hass.async_add_executor_job(
            _forget,
            store,
            identity,
            intent,
            use_user,
        )

    writes = extract_memory_writes(
        user_text,
        fragment=intent.fragment,
        controlled_entity_ids=controlled_entity_ids,
        route=route,
    )
    if not writes:
        # Ambiguous remember without extractable structure
        return (
            "I can remember preferences like local news, a default mailbox, "
            "or entity aliases (for example: remember dining light is "
            "light.dining_room, or after a lookup: remember this entity is "
            "for outdoor air quality). What should I store?"
        )

    return await hass.async_add_executor_job(
        _write_all,
        store,
        identity,
        writes,
        use_user,
    )


def _forget(
    store,
    identity: ResolvedIdentity,
    intent: MemoryIntent,
    use_user: bool,
) -> str:
    fragment = (intent.fragment or "").lower()
    deleted = 0
    entries = store.list_user(identity.user.id) if use_user else store.list_system()

    targets = []
    for entry in entries:
        hay = f"{entry.key} {entry.notes} {entry.value}".lower()
        if not fragment or any(tok in hay for tok in fragment.split() if len(tok) > 2):
            targets.append(entry)

    if not targets and fragment:
        # Fallback: delete keys that look related
        tokens = [tok for tok in fragment.replace(" ", "_").split("_") if tok]
        for entry in entries:
            if any(tok in entry.key for tok in tokens):
                targets.append(entry)

    for entry in targets:
        if entry.scope.value == "user" and entry.agent_user_id:
            if store.delete_user(entry.agent_user_id, entry.key):
                deleted += 1
        elif store.delete_system(entry.key):
            deleted += 1

    if deleted == 0:
        return "I could not find a matching memory to forget."
    return f"Forgot {deleted} memor{'y' if deleted == 1 else 'ies'}."


def _write_all(
    store,
    identity: ResolvedIdentity,
    writes: list[ExtractedMemory],
    use_user: bool,
) -> str:
    lines: list[str] = []
    for item in writes:
        if use_user and identity.user.kind == UserKind.REGISTERED:
            store.set_user(
                identity.user.id,
                item.key,
                item.value,
                route_scope=item.route_scope,
                notes=item.notes,
            )
            scope_label = "for you"
        else:
            store.set_system(
                item.key,
                item.value,
                route_scope=item.route_scope,
                notes=item.notes,
            )
            scope_label = "for the household"
        lines.append(f"- {item.key} = {item.value!r} ({scope_label})")
    return "Got it. I'll remember:\n" + "\n".join(lines)

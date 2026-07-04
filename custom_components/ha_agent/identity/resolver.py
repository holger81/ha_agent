"""Resolve the active agent user for a turn."""

from __future__ import annotations

from typing import Literal

from homeassistant.core import HomeAssistant

from .clustering import enroll_speaker_embedding, resolve_speaker_embedding
from .config import IDENTITY_VOICE_CONFIG, IdentityVoiceConfig
from .models import IdentitySource, ResolvedIdentity, SpeakerMatch, UserKind
from .runtime import (
    clear_enrollment_target,
    get_enrollment_session,
    get_identity_override,
    record_enrollment_sample,
)
from .store import IdentityStore, get_identity_store
from .voicebm import parse_voice_identity


async def resolve_agent_user(
    hass: HomeAssistant,
    entry_id: str,
    *,
    channel: Literal["console", "assist"],
    ha_user_id: str | None = None,
    conversation_id: str | None = None,
    admin_override_user_id: str | None = None,
    override_by_ha_user_id: str | None = None,
    extra_system_prompt: str | None = None,
    context_user_id: str | None = None,
    speaker_match: SpeakerMatch | None = None,
    voice_config: IdentityVoiceConfig | None = None,
    user_text: str | None = None,
) -> ResolvedIdentity:
    """Resolve who is acting for this turn."""
    store = get_identity_store(hass, entry_id)
    voice_cfg = voice_config or IDENTITY_VOICE_CONFIG

    def _resolve() -> ResolvedIdentity:
        if admin_override_user_id:
            user = store.get_user(admin_override_user_id)
            if user and not user.merged_into:
                return ResolvedIdentity(
                    user=user,
                    source=IdentitySource.OVERRIDE,
                    ha_user_id=ha_user_id,
                    override_by_ha_user_id=override_by_ha_user_id,
                )

        if channel == "console":
            override_id = get_identity_override(
                hass,
                entry_id,
                conversation_id=conversation_id,
            )
            if override_id:
                user = store.get_user(override_id)
                if user and not user.merged_into:
                    return ResolvedIdentity(
                        user=user,
                        source=IdentitySource.OVERRIDE,
                        ha_user_id=ha_user_id,
                        override_by_ha_user_id=override_by_ha_user_id,
                    )

        voice_payload = parse_voice_identity(extra_system_prompt)
        if voice_payload:
            resolved = _resolve_voice_payload(store, voice_payload)
            if resolved is not None:
                return resolved

        if channel == "assist" and speaker_match is not None:
            enrollment = get_enrollment_session(hass, entry_id)
            if enrollment is not None:
                enrolled = enroll_speaker_embedding(
                    store,
                    enrollment.agent_user_id,
                    speaker_match,
                    config=voice_cfg,
                )
                if enrolled is not None:
                    session = record_enrollment_sample(hass, entry_id)
                    if (
                        session is not None
                        and session.samples_collected
                        >= voice_cfg.enrollment_samples_target
                    ):
                        clear_enrollment_target(hass, entry_id)
                    return enrolled

            resolved = resolve_speaker_embedding(
                store,
                speaker_match,
                config=voice_cfg,
                user_text=user_text,
            )
            if resolved is not None:
                return resolved

        login_id = ha_user_id or context_user_id
        if login_id:
            user = store.get_by_ha_user_id(login_id)
            if user:
                return ResolvedIdentity(
                    user=user,
                    source=IdentitySource.LOGIN,
                    ha_user_id=login_id,
                )

        if channel == "assist":
            return ResolvedIdentity(
                user=store.get_assist_guest(),
                source=IdentitySource.ASSIST_GUEST,
                ha_user_id=login_id,
            )

        default_user = store.get_default_registered()
        if default_user:
            return ResolvedIdentity(
                user=default_user,
                source=IdentitySource.FALLBACK,
                ha_user_id=login_id,
            )

        users = store.list_users(kind=UserKind.REGISTERED)
        if users:
            return ResolvedIdentity(
                user=users[0],
                source=IdentitySource.FALLBACK,
                ha_user_id=login_id,
            )
        guest = store.get_assist_guest()
        return ResolvedIdentity(
            user=guest,
            source=IdentitySource.FALLBACK,
            ha_user_id=login_id,
        )

    identity = await hass.async_add_executor_job(_resolve)

    def _touch() -> None:
        store.touch_last_seen(identity.user.id)

    await hass.async_add_executor_job(_touch)
    return identity


def _resolve_voice_payload(
    store: IdentityStore,
    payload: dict,
) -> ResolvedIdentity | None:
    """Resolve VoiceBM-style identity payloads (Phase 9b hook)."""
    speaker_id = str(
        payload.get("speaker_id") or payload.get("agent_user_id") or ""
    ).strip()
    display_name = str(
        payload.get("display_name") or payload.get("speaker") or ""
    ).strip()
    confidence_raw = payload.get("confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else None

    if speaker_id:
        user = store.get_user(speaker_id)
        if user and not user.merged_into:
            return ResolvedIdentity(
                user=user,
                source=IdentitySource.VOICE,
                speaker_confidence=confidence,
            )

    if display_name:
        for user in store.list_users():
            if user.display_name.lower() == display_name.lower():
                return ResolvedIdentity(
                    user=user,
                    source=IdentitySource.VOICE,
                    speaker_confidence=confidence,
                )

    return None


def apply_identity_to_trace(trace, identity: ResolvedIdentity) -> None:
    """Copy resolved identity onto a turn trace."""
    trace.agent_user_id = identity.user.id
    trace.agent_user_display_name = identity.user.display_name
    trace.agent_user_kind = identity.user.kind.value
    trace.identity_source = identity.source.value
    trace.identity_ha_user_id = identity.ha_user_id
    trace.identity_override_by_ha_user_id = identity.override_by_ha_user_id
    trace.identity_speaker_confidence = identity.speaker_confidence

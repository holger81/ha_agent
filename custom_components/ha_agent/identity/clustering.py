"""Guest clustering from speaker embeddings."""

from __future__ import annotations

import uuid

from .config import SKIP_EMBED_QUALITIES, IdentityVoiceConfig
from .embeddings import cosine_similarity
from .models import IdentitySource, ResolvedIdentity, SpeakerMatch
from .store import IdentityStore


def resolve_speaker_embedding(
    store: IdentityStore,
    speaker_match: SpeakerMatch,
    *,
    config: IdentityVoiceConfig,
) -> ResolvedIdentity | None:
    """Match or create an agent user from a speaker embedding."""
    if not config.enabled:
        return None

    quality = speaker_match.quality or "ok"
    if quality in SKIP_EMBED_QUALITIES:
        return None

    embedding = speaker_match.embedding
    if not embedding:
        return None

    if (
        speaker_match.duration_ms is not None
        and speaker_match.duration_ms < config.min_utterance_ms
    ):
        return None

    profiles = store.list_voice_profiles()
    best_profile = None
    best_score = -1.0

    for profile in profiles:
        if profile.centroid is None:
            continue
        score = cosine_similarity(embedding, profile.centroid)
        if score > best_score:
            best_score = score
            best_profile = profile

    if best_profile is not None and best_score >= config.guest_create_threshold:
        user = store.get_user(best_profile.agent_user_id)
        if user and not user.merged_into:
            store.update_voice_profile_centroid(
                best_profile.id,
                embedding=embedding,
                match_confidence=best_score,
                model=speaker_match.model,
            )
            return ResolvedIdentity(
                user=user,
                source=IdentitySource.VOICE,
                speaker_confidence=best_score,
            )

    guest = store.create_guest(store.next_voice_guest_name())
    store.create_voice_profile(
        profile_id=str(uuid.uuid4()),
        agent_user_id=guest.id,
        backend=speaker_match.backend,
        model=speaker_match.model,
        centroid=embedding,
        match_confidence=best_score if best_score >= 0 else None,
    )
    return ResolvedIdentity(
        user=guest,
        source=IdentitySource.VOICE,
        speaker_confidence=best_score if best_score >= 0 else None,
    )

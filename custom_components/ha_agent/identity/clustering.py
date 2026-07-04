"""Guest clustering from speaker embeddings."""

from __future__ import annotations

import uuid

from ..const import LOGGER
from .config import SKIP_EMBED_QUALITIES, IdentityVoiceConfig
from .embeddings import cosine_similarity
from .models import IdentitySource, ResolvedIdentity, SpeakerMatch, VoiceProfile
from .store import IdentityStore


def _pick_best_profile(
    profiles: list[VoiceProfile],
    embedding: list[float],
    *,
    tie_margin: float,
) -> tuple[VoiceProfile | None, float]:
    """Return the best profile for an embedding, with recency tie-break."""
    scored: list[tuple[VoiceProfile, float]] = []
    for profile in profiles:
        if profile.centroid is None:
            continue
        score = cosine_similarity(embedding, profile.centroid)
        scored.append((profile, score))

    if not scored:
        return None, -1.0

    scored.sort(
        key=lambda item: (
            item[1],
            item[0].last_seen_at or 0.0,
            item[0].sample_count,
        ),
        reverse=True,
    )
    best_profile, best_score = scored[0]
    if len(scored) == 1:
        return best_profile, best_score

    runner_up_score = scored[1][1]
    if best_score - runner_up_score > tie_margin:
        return best_profile, best_score

    close = [
        item
        for item in scored
        if best_score - item[1] <= tie_margin
    ]
    close.sort(
        key=lambda item: (
            item[0].last_seen_at or 0.0,
            item[0].sample_count,
            item[1],
        ),
        reverse=True,
    )
    return close[0][0], close[0][1]


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
    best_profile, best_score = _pick_best_profile(
        profiles,
        embedding,
        tie_margin=config.guest_tie_margin,
    )

    if best_profile is not None and best_score >= config.guest_create_threshold:
        user = store.get_user(best_profile.agent_user_id)
        if user and not user.merged_into:
            store.update_voice_profile_centroid(
                best_profile.id,
                embedding=embedding,
                match_confidence=best_score,
                model=speaker_match.model,
            )
            LOGGER.debug(
                "Voice identity matched %s (score=%.3f, samples=%d)",
                user.display_name,
                best_score,
                best_profile.sample_count,
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
    LOGGER.debug(
        "Voice identity created %s (best_score=%.3f, profiles=%d)",
        guest.display_name,
        best_score,
        len(profiles),
    )
    return ResolvedIdentity(
        user=guest,
        source=IdentitySource.VOICE,
        speaker_confidence=best_score if best_score >= 0 else None,
    )

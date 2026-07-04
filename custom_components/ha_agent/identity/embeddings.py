"""Embedding vector helpers."""

from __future__ import annotations

import math
import struct


def pack_embedding(values: list[float]) -> bytes:
    """Serialize an embedding vector to bytes."""
    return struct.pack(f"{len(values)}f", *values)


def unpack_embedding(blob: bytes | None) -> list[float] | None:
    """Deserialize an embedding vector from bytes."""
    if not blob:
        return None
    count = len(blob) // 4
    if count == 0:
        return None
    return list(struct.unpack(f"{count}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def update_centroid(
    current: list[float] | None,
    sample_count: int,
    new_sample: list[float],
) -> list[float]:
    """Return a running-average centroid after one new sample."""
    if current is None or sample_count <= 0:
        return list(new_sample)
    total = sample_count + 1
    return [
        ((current[index] * sample_count) + new_sample[index]) / total
        for index in range(len(new_sample))
    ]

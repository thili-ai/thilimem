"""Entity resolution: map name variants to one canonical entity, and merge duplicate memories."""
from __future__ import annotations

from .embeddings import Embedder, cosine
from .types import Memory


def resolve_entity(name: str, known: list[str], embedder: Embedder, threshold: float = 0.82) -> str:
    """Return the canonical name for `name` given already-known entities.

    Embedding-similarity match: if the closest known entity is similar enough, reuse it; otherwise
    `name` is a new entity. (For higher precision, add an LLM yes/no judge on the top candidate.)
    """
    if not known:
        return name
    nv = embedder.embed(name)
    best, score = name, -1.0
    for k in known:
        s = cosine(nv, embedder.embed(k))
        if s > score:
            best, score = k, s
    return best if score >= threshold else name


def merge_memories(a: Memory, b: Memory) -> Memory:
    """Merge two memories about the same fact: keep the newer text, the higher importance, union entities."""
    newer = a if a.created_at >= b.created_at else b
    newer.importance = max(a.importance, b.importance)
    newer.entities = sorted(set(a.entities) | set(b.entities))
    return newer

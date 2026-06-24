"""Hybrid retrieval: rank by semantic similarity + recency + importance, not similarity alone."""
from __future__ import annotations

from datetime import datetime, timezone

from .store import MemoryStore
from .types import Memory


def recency_decay(created_at: datetime, half_life_hours: float = 72.0) -> float:
    """1.0 for a brand-new memory, decaying by half every `half_life_hours`."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - created_at).total_seconds()
    return 0.5 ** (age_s / (half_life_hours * 3600.0))


def retrieve(
    store: MemoryStore,
    query: str,
    k: int = 5,
    weights: tuple[float, float, float] = (0.6, 0.25, 0.15),
    candidate_k: int = 20,
    half_life_hours: float = 72.0,
) -> list[Memory]:
    """Return the k most useful memories for `query`.

    score = w_sim*similarity + w_rec*recency_decay + w_imp*importance
    Pure vector search (w_sim=1) misses memories that are most *recent* or most *important* rather
    than most lexically similar — that's the gap this closes.
    """
    candidates = store.search(query, k=candidate_k)
    w_sim, w_rec, w_imp = weights
    scored = [
        (m, w_sim * sim + w_rec * recency_decay(m.created_at, half_life_hours) + w_imp * m.importance)
        for m, sim in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:k]]

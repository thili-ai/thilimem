"""The knowledge pipeline: merge near-duplicates, decay the stale, prune the irrelevant.

Run this periodically (a background job), not on the hot retrieval path.
"""
from __future__ import annotations

from .embeddings import cosine
from .entities import merge_memories
from .retrieve import recency_decay
from .store import MemoryStore


def consolidate(
    store: MemoryStore,
    sim_threshold: float = 0.95,
    prune_below: float = 0.05,
    protect_importance: float = 0.6,
) -> dict:
    """Merge near-duplicate memories, then decay+prune low-value ones. Returns stats.

    Durable facts are protected: a memory whose importance is >= `protect_importance` is never pruned
    by decay, however old (a peanut allergy must not be forgotten just for being old). For the rest,
    value = importance * recency_decay(last_used_at or created_at), so memories that keep getting
    *recalled* stay alive even if they were first learned long ago.
    """
    ids = [m.id for m in store.all()]
    by_id = {m.id: m for m in store.all()}
    removed: set[str] = set()
    merged = 0

    for i in range(len(ids)):
        if ids[i] in removed:
            continue
        vi = store._vecs.get(ids[i])
        if vi is None:
            continue
        for j in range(i + 1, len(ids)):
            if ids[j] in removed:
                continue
            vj = store._vecs.get(ids[j])
            if vj is None:
                continue
            if cosine(vi, vj) >= sim_threshold:
                survivor = merge_memories(by_id[ids[i]], by_id[ids[j]])
                store.add(survivor)            # update the survivor row
                store.delete(ids[j])
                removed.add(ids[j])
                merged += 1

    # Decay + prune, but never drop a durable (high-importance) memory.
    pruned = 0
    for m in store.all():
        if m.importance >= protect_importance:
            continue
        ref = m.last_used_at or m.created_at  # a recalled memory stays fresh
        if m.importance * recency_decay(ref) < prune_below:
            store.delete(m.id)
            pruned += 1

    return {"merged": merged, "pruned": pruned, "remaining": len(store)}

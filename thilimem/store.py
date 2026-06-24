"""Persistent memory store: SQLite for rows, an in-memory vector index for semantic search."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np

from .embeddings import Embedder, HashingEmbedder, cosine
from .types import Memory


class MemoryStore:
    """Stores memories in SQLite (with their embeddings) and serves semantic search from memory.

    Vectors are normalized, so similarity is a dot product. Embeddings are cached in `_vecs` and
    persisted as blobs so a file-backed store survives restarts.
    """

    def __init__(self, path: str = ":memory:", embedder: Optional[Embedder] = None):
        self.embedder = embedder or HashingEmbedder()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self._init_schema()
        self._vecs: dict[str, np.ndarray] = {}
        self._load_vectors()

    def _init_schema(self) -> None:
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, type TEXT NOT NULL,
                entities TEXT NOT NULL DEFAULT '[]', importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL, last_used_at TEXT, source TEXT, embedding BLOB)"""
        )
        self.db.commit()

    def _load_vectors(self) -> None:
        for row in self.db.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"):
            self._vecs[row["id"]] = np.frombuffer(row["embedding"], dtype="float32")

    def add(self, m: Memory) -> Memory:
        vec = self.embedder.embed(m.text).astype("float32")
        self.db.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
            (m.id, m.text, m.type, json.dumps(m.entities), m.importance,
             m.created_at.isoformat(), m.last_used_at.isoformat() if m.last_used_at else None,
             m.source, vec.tobytes()),
        )
        self.db.commit()
        self._vecs[m.id] = vec
        return m

    def _to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"], text=row["text"], type=row["type"],
            entities=json.loads(row["entities"] or "[]"), importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_used_at=datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None,
            source=row["source"],
        )

    def get(self, mid: str) -> Optional[Memory]:
        row = self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        return self._to_memory(row) if row else None

    def all(self) -> list[Memory]:
        return [self._to_memory(r) for r in self.db.execute("SELECT * FROM memories")]

    def delete(self, mid: str) -> None:
        self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
        self.db.commit()
        self._vecs.pop(mid, None)

    def search(self, query: str, k: int = 5) -> list[tuple[Memory, float]]:
        """Top-k memories by cosine similarity to the query."""
        if not self._vecs:
            return []
        qv = self.embedder.embed(query)
        ranked = sorted(((mid, cosine(qv, v)) for mid, v in self._vecs.items()),
                        key=lambda x: x[1], reverse=True)
        out: list[tuple[Memory, float]] = []
        for mid, score in ranked[:k]:
            m = self.get(mid)
            if m:
                out.append((m, score))
        return out

    def entities(self) -> list[str]:
        names: set[str] = set()
        for m in self.all():
            names.update(m.entities)
        return sorted(names)

    def __len__(self) -> int:
        return len(self._vecs)

"""Embedders. Dependency-free hashing embedder by default; sentence-transformers if installed.

All embedders return L2-normalized float32 vectors, so `cosine` is just a dot product.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words hashed into a fixed-dim vector.

    Crude (no semantics beyond shared tokens) but works offline with zero deps — fine for demos,
    tests, and the course. For real quality install `thilimem[st]` and use SentenceTransformerEmbedder.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep

        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        v = self._m.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")


def get_embedder(name: str = "hashing", **kwargs) -> Embedder:
    if name in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(**kwargs)
    return HashingEmbedder(**kwargs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors (== dot product)."""
    return float(np.dot(a, b))

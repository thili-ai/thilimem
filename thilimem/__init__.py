"""thilimem — a small, hackable memory engine for AI agents.

Extract → resolve → store → retrieve → consolidate. Inspired by Maximem's Synap; built for learning.
"""
from .agent import MemoryAgent
from .consolidate import consolidate
from .embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine,
    get_embedder,
)
from .entities import merge_memories, resolve_entity
from .eval import run_eval
from .extract import extract_memories
from .llm import LLM, json_complete, openai_compatible
from .retrieve import recency_decay, retrieve
from .store import MemoryStore
from .types import Memory, MemoryType

__version__ = "0.1.0"

__all__ = [
    "Memory", "MemoryType",
    "Embedder", "HashingEmbedder", "SentenceTransformerEmbedder", "get_embedder", "cosine",
    "LLM", "json_complete", "openai_compatible",
    "extract_memories", "resolve_entity", "merge_memories",
    "MemoryStore", "retrieve", "recency_decay", "consolidate",
    "MemoryAgent", "run_eval",
    "__version__",
]

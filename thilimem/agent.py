"""A chat agent with long-term memory: read relevant memories before answering, write after."""
from __future__ import annotations

from .extract import extract_memories
from .llm import LLM
from .retrieve import retrieve
from .store import MemoryStore

_SYSTEM = (
    "You are a helpful assistant with long-term memory. "
    "Use the relevant memories below when they help; ignore them when they don't."
)


class MemoryAgent:
    """Wraps an LLM with a thilimem store. read-before / write-after each turn."""

    def __init__(self, store: MemoryStore, llm: LLM, k: int = 5):
        self.store = store
        self.llm = llm
        self.k = k

    def build_prompt(self, user_msg: str) -> tuple[str, list]:
        mems = retrieve(self.store, user_msg, k=self.k)
        block = "\n".join(f"- {m.text}" for m in mems) or "(no relevant memories yet)"
        prompt = f"{_SYSTEM}\n\nRelevant memories:\n{block}\n\nUser: {user_msg}\nAssistant:"
        return prompt, mems

    def respond(self, user_msg: str, session: str = "default") -> str:
        prompt, _ = self.build_prompt(user_msg)          # READ (anticipatory)
        reply = self.llm(prompt)
        for m in extract_memories([{"role": "user", "content": user_msg}], llm=self.llm, source=session):
            self.store.add(m)                            # WRITE
        return reply

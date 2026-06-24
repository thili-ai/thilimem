"""Turn raw conversation turns into atomic, structured memories."""
from __future__ import annotations

from typing import Optional

from .llm import LLM, json_complete
from .types import Memory

_PROMPT = """You extract durable memories from a conversation.
Return JSON: {"memories": [{"text": str, "type": str, "entities": [str], "importance": number}]}
- text: ONE atomic statement worth remembering long-term
- type: one of fact | preference | event | entity
- entities: canonical names mentioned (people, projects, orgs)
- importance: 0..1
Only include things worth remembering; skip greetings and chit-chat.

Conversation:
__CONV__
"""

_VALID = {"fact", "preference", "event", "entity"}


def extract_memories(turns: list[dict], llm: Optional[LLM] = None, source: str | None = None) -> list[Memory]:
    """Extract memories from recent turns. With no `llm`, falls back to a simple rule-based heuristic."""
    if not turns:
        return []
    if llm is None:
        return _naive_extract(turns, source)

    conv = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns)
    data = json_complete(llm, _PROMPT.replace("__CONV__", conv))
    items = data.get("memories", []) if isinstance(data, dict) else []
    out: list[Memory] = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        typ = it.get("type") if it.get("type") in _VALID else "fact"
        try:
            importance = float(it.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        out.append(Memory(
            text=text, type=typ, entities=list(it.get("entities") or []),
            importance=max(0.0, min(1.0, importance)), source=source,
        ))
    return out


def _naive_extract(turns: list[dict], source: str | None) -> list[Memory]:
    """No-LLM fallback: keep declarative user statements. Crude, but lets thilimem run offline."""
    out = []
    for t in turns:
        if t.get("role") == "user":
            text = (t.get("content") or "").strip()
            if text and not text.endswith("?"):
                out.append(Memory(text=text, type="fact", source=source))
    return out

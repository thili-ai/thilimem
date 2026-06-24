# thilimem

**A small, hackable memory engine for AI agents.** Your agent is only as smart as what it remembers —
`thilimem` gives it long-term memory: it **extracts** durable facts from conversations, **resolves**
entities, **stores** them with embeddings, **retrieves** the right ones (semantic + recency +
importance), and **consolidates** them over time.

It's intentionally tiny and readable — a teaching-grade companion to production memory stacks like
[Maximem's Synap](https://www.maximem.ai/), [Mem0](https://github.com/mem0ai/mem0), and
[Zep](https://www.getzep.com/). Built for the [Learn Fast](https://learnfast.thili.ai) course
*"Build an Agent Memory Engine."*

> ⚠️ Educational reference implementation, not production infra. Inspired by Synap; not affiliated with Maximem.

## Install
```bash
pip install thilimem            # core (dependency-free embeddings)
pip install "thilimem[st]"      # + sentence-transformers for real semantic embeddings
```

## Quickstart
```python
from thilimem import MemoryStore, MemoryAgent, openai_compatible

# 1) any OpenAI-compatible LLM (OpenAI, Gemini's OpenAI endpoint, vLLM, ...)
llm = openai_compatible("https://api.openai.com/v1", api_key="sk-...", model="gpt-4o-mini")

store = MemoryStore("memories.db")          # SQLite-backed, persists across runs
agent = MemoryAgent(store, llm)

# Session 1
agent.respond("Hi, I'm Maya and I'm allergic to peanuts.", session="s1")
# Session 2 — a brand new conversation; the agent still remembers
print(agent.respond("What should I avoid at dinner?", session="s2"))
```

Use the pieces directly, too:
```python
from thilimem import MemoryStore, retrieve, extract_memories, consolidate

store = MemoryStore()                                  # in-memory
for m in extract_memories([{"role": "user", "content": "I prefer morning meetings."}], llm=llm):
    store.add(m)

hits = retrieve(store, "when should we schedule?", k=3) # semantic + recency + importance
stats = consolidate(store)                              # merge dups, decay, prune
```

No LLM handy? The core (store / retrieve / consolidate) works without one, and `extract_memories`
falls back to a simple rule-based extractor — so you can explore offline.

## What's inside
| Module | Role |
|---|---|
| `types` | the `Memory` schema |
| `embeddings` | hashing embedder (no deps) or sentence-transformers |
| `extract` | conversation turns → atomic memories (LLM or rule-based) |
| `entities` | entity resolution + memory merging |
| `store` | SQLite rows + vector search |
| `retrieve` | hybrid ranking (similarity + recency + importance) |
| `consolidate` | merge / decay / prune |
| `agent` | read-before / write-after chat wrapper |
| `eval` | tiny recall+latency benchmark harness |

## License
MIT © thili.ai

# thilimem

**A small, hackable memory engine for AI agents.** Your agent is only as smart as what it remembers —
`thilimem` gives it long-term memory: it **extracts** durable facts from conversations, **resolves**
entities, **stores** them with embeddings, **retrieves** the right ones (semantic + recency +
importance), and **consolidates** them over time.

It's intentionally tiny and readable — a teaching-grade companion to production memory stacks like
[Maximem's Synap](https://www.maximem.ai/), [Mem0](https://github.com/mem0ai/mem0), and
[Zep](https://www.getzep.com/).

> ⚠️ Educational reference implementation, not production infrastructure. Inspired by Synap; not affiliated with Maximem.

## 📚 Built lesson-by-lesson in a Learn Fast course

`thilimem` is the reference implementation for the **[Learn Fast](https://learnfast.thili.ai) course
"Build an Agent Memory Engine."** You don't just *use* this library — in the course you **build it
yourself**, one module per lesson, starting from a chatbot that forgets and ending with one that
remembers across separate sessions.

This repo is here so you can **check your work**, unblock yourself, or take a shortcut. Each lesson
mirrors one (or two) modules:

| Lesson | What you build | Module(s) |
|---|---|---|
| 1 · Why agents need memory | a forgetful-bot demo + the memory taxonomy | *(motivation)* |
| 2 · Capturing & extracting memories | conversation turns → atomic, typed memories | `types.py`, `llm.py`, `extract.py` |
| 3 · Entity resolution | link "Maya" / "Maya Sharma" into one entity | `entities.py` |
| 4 · Storage & embeddings | the `MemoryStore` (vectors + SQLite) | `embeddings.py`, `store.py` |
| 5 · Retrieval beyond vector search | hybrid recall (semantic + recency + importance) | `retrieve.py` |
| 6 · Consolidation & forgetting | the knowledge pipeline (merge / decay / prune) | `consolidate.py` |
| 7 · Wire it into an agent | the read-before / write-after loop | `agent.py` |
| 8 · Evaluate it | a mini LongMemEval/LoCoMo-style benchmark | `eval.py` |

> New here? **Start with the course**, then come back to read the polished source. The whole engine is
> a few hundred lines, so you can hold it in your head.

## Install

Not on PyPI yet — install straight from GitHub:

```bash
pip install "git+https://github.com/thili-ai/thilimem.git"                  # core (dependency-free embeddings)
pip install "thilimem[st] @ git+https://github.com/thili-ai/thilimem.git"   # + sentence-transformers for real semantic embeddings
```

## Quickstart

```python
from thilimem import MemoryStore, MemoryAgent, openai_compatible

# 1) any OpenAI-compatible LLM (OpenAI, Groq, Gemini's OpenAI endpoint, vLLM, ...)
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

store = MemoryStore()                                   # in-memory
for m in extract_memories([{"role": "user", "content": "I prefer morning meetings."}], llm=llm):
    store.add(m)

hits = retrieve(store, "when should we schedule?", k=3)  # semantic + recency + importance
stats = consolidate(store)                               # merge dups, decay, prune
```

No LLM handy? The core (store / retrieve / consolidate) works without one, and `extract_memories`
falls back to a simple rule-based extractor — so you can explore offline.

## What's inside

| Module | Role |
|---|---|
| `types` | the `Memory` schema |
| `llm` | provider-agnostic LLM callable (`prompt -> str`) + robust JSON parsing |
| `embeddings` | hashing embedder (no deps) or sentence-transformers |
| `extract` | conversation turns → atomic memories (LLM or rule-based) |
| `entities` | entity resolution + memory merging |
| `store` | SQLite rows + vector search |
| `retrieve` | hybrid ranking (similarity + recency + importance) |
| `consolidate` | merge / decay / prune |
| `agent` | read-before / write-after chat wrapper |
| `eval` | tiny recall + latency benchmark harness |

## Development

```bash
pip install -e ".[dev,st]"
pytest -q
```

## License

MIT © thili.ai

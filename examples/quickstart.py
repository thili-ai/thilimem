"""Run thilimem end-to-end with NO API key (rule-based extraction + hashing embeddings).

    python examples/quickstart.py
"""
from thilimem import MemoryStore, extract_memories, retrieve, consolidate

store = MemoryStore()  # in-memory

# Ingest a couple of "sessions" (no LLM -> rule-based extraction)
session1 = [
    {"role": "user", "content": "I'm Maya and I'm allergic to peanuts."},
    {"role": "user", "content": "I prefer morning meetings over afternoons."},
]
for turn in session1:
    for m in extract_memories([turn], llm=None, source="s1"):
        store.add(m)

print(f"stored {len(store)} memories\n")

# Retrieve for a later, differently-worded query
print("Q: what should Maya avoid eating?")
for m in retrieve(store, "what foods are unsafe for Maya?", k=3):
    print("  •", m.text)

print("\nQ: when to schedule?")
for m in retrieve(store, "best time for a meeting", k=3):
    print("  •", m.text)

print("\nconsolidate ->", consolidate(store))

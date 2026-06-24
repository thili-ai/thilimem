"""Offline tests — no network. Uses the hashing embedder and a stub LLM."""
import json

from thilimem import (
    Memory,
    MemoryAgent,
    MemoryStore,
    consolidate,
    extract_memories,
    merge_memories,
    resolve_entity,
    retrieve,
    run_eval,
)
from thilimem.embeddings import HashingEmbedder


def test_store_add_search_persist(tmp_path):
    db = str(tmp_path / "m.db")
    s = MemoryStore(db)
    s.add(Memory(text="Maya is allergic to peanuts", entities=["Maya"]))
    s.add(Memory(text="The sky is blue"))
    assert len(s) == 2
    hits = s.search("what is Maya allergic to", k=1)
    assert hits and "peanut" in hits[0][0].text.lower()
    # reopen -> vectors reload from disk
    s2 = MemoryStore(db)
    assert len(s2) == 2


def test_retrieve_returns_memories():
    s = MemoryStore()
    s.add(Memory(text="Maya is allergic to peanuts"))
    s.add(Memory(text="Project Apollo ships in March"))
    out = retrieve(s, "peanuts", k=1)
    assert out and isinstance(out[0], Memory)


def test_naive_extract_skips_questions():
    mems = extract_memories(
        [{"role": "user", "content": "I love hiking."},
         {"role": "user", "content": "What's the weather?"}],
        llm=None,
    )
    assert len(mems) == 1 and mems[0].text == "I love hiking."


def test_extract_with_stub_llm():
    def stub(_prompt):
        return json.dumps({"memories": [
            {"text": "Maya is allergic to peanuts", "type": "fact",
             "entities": ["Maya"], "importance": 0.9}
        ]})
    mems = extract_memories([{"role": "user", "content": "Maya can't eat peanuts"}], llm=stub)
    assert len(mems) == 1 and mems[0].importance == 0.9 and mems[0].entities == ["Maya"]


def test_resolve_entity_new_vs_known():
    emb = HashingEmbedder()
    assert resolve_entity("Apollo", [], emb) == "Apollo"
    assert resolve_entity("Apollo", ["Apollo"], emb) == "Apollo"  # exact -> canonical


def test_merge_keeps_max_importance():
    a = Memory(text="x", importance=0.3, entities=["A"])
    b = Memory(text="x", importance=0.8, entities=["B"])
    m = merge_memories(a, b)
    assert m.importance == 0.8 and set(m.entities) == {"A", "B"}


def test_consolidate_merges_duplicates():
    s = MemoryStore()
    s.add(Memory(text="Maya is allergic to peanuts"))
    s.add(Memory(text="Maya is allergic to peanuts"))  # exact dup
    stats = consolidate(s)
    assert stats["merged"] >= 1 and stats["remaining"] < 2


def test_agent_writes_after_turn():
    def stub(_prompt):
        return "noted"
    s = MemoryStore()
    agent = MemoryAgent(s, llm=lambda p: json.dumps({"memories": [
        {"text": "User is Maya", "type": "entity", "entities": ["Maya"], "importance": 0.7}]})
        if "extract" in p.lower() else "noted")
    # simplest: use naive path via a stub that returns memories regardless
    reply = agent.respond("I'm Maya", session="s1")
    assert isinstance(reply, str)
    assert len(s) >= 1


def test_run_eval_accuracy():
    cases = [{"expect": "peanuts"}, {"expect": "blue"}]
    res = run_eval(cases, answer_fn=lambda c: f"the answer is {c['expect']}")
    assert res["accuracy"] == 1.0 and res["n"] == 2

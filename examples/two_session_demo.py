"""The forgetting test: tell an agent something in session 1, then ask in a brand-new session 2.

    python examples/two_session_demo.py                      # no API key: a tiny stub model, so you can see the memory plumbing
    OPENAI_API_KEY=sk-... python examples/two_session_demo.py   # a real model (any OpenAI-compatible endpoint; see below)

Session 2 opens the SQLite file again with a fresh agent, so nothing is carried over except what was stored.
Optional env: OPENAI_BASE_URL (default https://api.openai.com/v1), OPENAI_MODEL (default gpt-4o-mini).
"""
import json, os, re, tempfile
from pathlib import Path

from thilimem import MemoryAgent, MemoryStore, openai_compatible


def stub_llm(prompt: str) -> str:
    """Offline stand-in for a model. It only shows how memories flow in and out; it is not smart."""
    if prompt.startswith("You extract durable memories"):
        conv = prompt.split("Conversation:", 1)[1]
        facts = [l.split(":", 1)[1].strip() for l in conv.strip().splitlines() if l.startswith("user:") and not l.strip().endswith("?")]
        return json.dumps({"memories": [{"text": f, "type": "fact", "entities": [], "importance": 0.8} for f in facts]})
    block = prompt.split("Relevant memories:\n", 1)[1].split("\n\nUser:", 1)[0]
    return f"[stub model] I remember:\n{block}"


if os.environ.get("OPENAI_API_KEY"):
    llm = openai_compatible(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                            api_key=os.environ["OPENAI_API_KEY"], model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    print("Using a real model.\n")
else:
    llm = stub_llm
    print("No OPENAI_API_KEY set: using a stub model (shows the memory flow only).\n")

db = Path(tempfile.mkdtemp()) / "memories.db"

print("--- Session 1 ---")
agent = MemoryAgent(MemoryStore(str(db)), llm)
print("You:  Hi, I'm Maya and I'm allergic to peanuts.")
agent.respond("Hi, I'm Maya and I'm allergic to peanuts.", session="s1")
print(f"(stored {len(agent.store)} memory)\n")

print("--- Session 2: a new agent, a new conversation, same database file ---")
agent2 = MemoryAgent(MemoryStore(str(db)), llm)
question = "What should I avoid at dinner?"
print("You: ", question)
print("Bot: ", agent2.respond(question, session="s2"))

print("\n--- The same question with no memory (empty store) ---")
print("Bot: ", MemoryAgent(MemoryStore(), llm).respond(question, session="s0"))

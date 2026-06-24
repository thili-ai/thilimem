"""LLM plumbing.

An `LLM` is just a callable `(prompt: str) -> str`. That keeps thilimem provider-agnostic — wire in
OpenAI, Gemini, a local vLLM, or a stub for tests. `json_complete` adds robust JSON parsing with a
retry, and `openai_compatible` builds an LLM for any OpenAI-style /chat/completions endpoint with no
extra dependencies (stdlib urllib).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Union

LLM = Callable[[str], str]


def json_complete(llm: LLM, prompt: str, retries: int = 2) -> Union[dict, list]:
    """Call the LLM and parse JSON, retrying with a stronger instruction on failure."""
    last_err: Exception | None = None
    for _ in range(retries):
        out = llm(prompt) or ""
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            last_err = e
            m = re.search(r"(\{.*\}|\[.*\])", out, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError as e2:
                    last_err = e2
        prompt = prompt + "\n\nReturn ONLY valid JSON, no prose, no code fences."
    raise ValueError(f"LLM did not return valid JSON: {last_err}")


def openai_compatible(base_url: str, api_key: str, model: str, temperature: float = 0.2) -> LLM:
    """Build an LLM callable for any OpenAI-compatible endpoint (OpenAI, Gemini's OpenAI endpoint,
    vLLM, etc). Uses stdlib urllib so thilimem stays dependency-free.

    Example:
        llm = openai_compatible(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=os.environ["GEMINI_API_KEY"], model="gemini-2.5-flash")
    """
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"

    def _llm(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]

    return _llm

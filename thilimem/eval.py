"""A tiny evaluation harness — score recall accuracy + latency, in the spirit of LongMemEval/LoCoMo."""
from __future__ import annotations

import time
from typing import Callable, Optional

Judge = Callable[[str, str], bool]


def _contains_judge(answer: str, expected: str) -> bool:
    return expected.lower() in (answer or "").lower()


def run_eval(cases: list[dict], answer_fn: Callable[[dict], str], judge: Optional[Judge] = None) -> dict:
    """Run `answer_fn` over cases and report accuracy + p50 latency.

    Each case is a dict with at least 'expect'. `answer_fn(case)` returns the system's answer.
    Default judge = case-insensitive substring match; pass an LLM judge for fuzzy grading.
    """
    judge = judge or _contains_judge
    correct = 0
    latencies: list[float] = []
    for case in cases:
        t0 = time.perf_counter()
        answer = answer_fn(case)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if judge(answer, case.get("expect", "")):
            correct += 1
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    return {
        "accuracy": round(correct / len(cases), 3) if cases else 0.0,
        "p50_latency_ms": round(p50, 2),
        "n": len(cases),
    }

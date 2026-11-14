"""Held-out question/citation evaluation for the local assistant."""
from __future__ import annotations

import json
from pathlib import Path


def score_case(answer: str, sources: list, must_contain, source_domain: str):
    lower = answer.lower()
    answer_ok = any(str(x).lower() in lower for x in (must_contain or []))
    citation_ok = any(source_domain in str(s.get("source", ""))
                      for s in (sources or []))
    return {
        "answer_ok": bool(answer_ok),
        "citation_ok": bool(citation_ok),
        "strict_ok": bool(answer_ok and citation_ok),
    }


def load_cases(path: str | Path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(p.read_text(encoding="utf-8"))


def run_eval(assistant, cases, max_new: int = 160, temperature: float = 0.0):
    results = []
    for case in cases:
        res = assistant.answer(case["question"], use_knowledge=True,
                               max_new_tokens=max_new,
                               temperature=temperature)
        score = score_case(res.get("answer", ""), res.get("sources", []),
                           case.get("must_contain", []),
                           case.get("source_domain", ""))
        results.append({"question": case["question"],
                        "answer": res.get("answer", ""),
                        "sources": res.get("sources", []),
                        **score})
    n = len(results)
    return {
        "n": n,
        "answer_accuracy": sum(r["answer_ok"] for r in results) / max(n, 1),
        "citation_rate": sum(r["citation_ok"] for r in results) / max(n, 1),
        "strict_accuracy": sum(r["strict_ok"] for r in results) / max(n, 1),
        "results": results,
    }

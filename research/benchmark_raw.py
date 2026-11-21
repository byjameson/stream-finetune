"""Raw-model knowledge probe: no RAG, no tools, no agent runtime.

    python research/benchmark_raw.py \
        --model checkpoints/transformer_behavior_50m_long \
        --out research/results/raw_knowledge_base.json

This is deliberately a small, simple probe. It shows what the 50M language
model knows on its own; the product answer path uses DocMemory and tools.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from streamft.transformer import generate

try:
    from research.chat_human import load_model
except ImportError:
    from chat_human import load_model

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Turkey?", ["ankara"]),
    ("What is 2 + 2?", ["4", "four"]),
    ("What is 7 * 8?", ["56", "fifty-six"]),
    ("What is the largest planet?", ["jupiter"]),
    ("Who wrote Hamlet?", ["shakespeare"]),
    ("What is the boiling point of water?", ["100", "212"]),
    ("What does HTTP stand for?", ["hypertext"]),
    ("Name a primary color.", ["red", "blue", "yellow"]),
    ("What is the speed of light?", ["299", "300,000", "186,000"]),
]


def _alpha(text: str) -> float:
    return sum(ch.isalpha() for ch in text) / max(len(text), 1)


def run(model_path: str, tokenizer: str, adapter: str = "") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(model_path, device, "transformer", tokenizer,
                            adapter)
    uid = tok.token_to_id("<|user|>")
    aid = tok.token_to_id("<|assistant|>")
    rows = []
    for question, expected in CASES:
        text, stop = generate(
            model, tok, f"<|user|> {question} <|assistant|>",
            max_new=64, temperature=1e-5, top_k=1,
            repetition_penalty=1.0, banned_ids=(uid, aid), device=device)
        lower = text.lower()
        hit = any(value.lower() in lower for value in expected)
        rows.append({
            "question": question,
            "expected": expected,
            "answer": text,
            "correct": bool(hit),
            "alpha": round(_alpha(text), 3),
            "stop": stop,
            "tag_leak": "<|" in text,
        })
    passed = sum(row["correct"] for row in rows)
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_path,
        "adapter": adapter,
        "score": passed / max(len(rows), 1),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
        "note": ("Raw-model probe without DocMemory or tools. A low score is "
                 "expected at 50M and is why facts belong in RAG."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="checkpoints/transformer_behavior_50m_long")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", default="research/results/raw_knowledge.json")
    args = ap.parse_args(argv)
    payload = run(args.model, args.tokenizer, args.adapter)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for row in payload["rows"]:
        print(("PASS " if row["correct"] else "FAIL ") + row["question"] +
              " -> " + row["answer"].replace("\n", " ")[:100])
    print(f"raw knowledge: {payload['passed']}/{payload['total']} "
          f"({payload['score'] * 100:.1f}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

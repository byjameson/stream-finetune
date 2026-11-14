"""Build a held-out narrative-coherence benchmark from TinyStories validation.

For each case: two context sentences from a held-out story, the true next
sentence, and three next sentences from other stories. The task is to rank the
true continuation highest by length-normalized log probability.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOK = Tokenizer.from_file(str(ROOT / "data/ts_bpe8k.json"))
OUT = ROOT / "research/semantic_cases.json"


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def ntok(s):
    return len(TOK.encode(s).ids)


def main():
    ids = torch.load(ROOT / "data/ts_valid.pt", weights_only=True)[:300_000]
    text = TOK.decode(ids.tolist()).replace("\n\n", " ")
    stories = [s.strip() for s in re.split(r"(?=Once upon a time)", text)
               if s.strip()]
    rng = random.Random(7)
    rows = []
    for st in stories:
        ss = sentences(st)
        if (len(ss) >= 3 and 6 <= ntok(ss[2]) <= 40
                and "Once upon a time" not in ss[2]):
            rows.append((ss[0] + " " + ss[1], ss[2]))
    rng.shuffle(rows)
    cases = []
    for i, (ctx, pos) in enumerate(rows[:120]):
        n = ntok(pos)
        pool = [p for j, (c, p) in enumerate(rows) if j != i
                and abs(ntok(p) - n) <= 4 and p != pos]
        if len(pool) < 3:
            continue
        negs = rng.sample(pool, 3)
        cases.append({"id": len(cases), "context": ctx, "positive": pos,
                      "negatives": negs})
        if len(cases) >= 100:
            break
    OUT.write_text(json.dumps({"n": len(cases), "chance": 0.25,
                               "cases": cases}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

"""Small qualitative/quantitative probe for a spiking checkpoint.

    python scripts/spiking_quality.py --model checkpoints/spiking_4m_full

Generates a fixed prompt set and reports distinct-1/2 and repeated 4-gram
fraction. It is a smoke metric, not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamft.spiking import generate, load_checkpoint  # noqa: E402

PROMPTS = [
    "Once upon a time",
    "The little dog",
    "One day, a girl named Lily",
    "Tom and his friend",
]


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model, tok, meta = load_checkpoint(args.model)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = []
    for prompt in PROMPTS:
        samples.append(generate(model, tok, prompt, args.max_new, device,
                                temperature=args.temperature, top_k=args.top_k,
                                repetition_penalty=1.1))
    toks = [re.findall(r"\w+|[^\w\s]", s.lower()) for s in samples]
    all_toks = [t for row in toks for t in row]
    uni = {t for t in all_toks}
    bi = {g for row in toks for g in ngrams(row, 2)}
    four = [g for row in toks for g in ngrams(row, 4)]
    repeat = 0.0
    if four:
        counts = Counter(four)
        repeat = sum(c - 1 for c in counts.values() if c > 1) / len(four)
    print(json.dumps({
        "model": args.model,
        "meta": meta,
        "samples": samples,
        "distinct_1": len(uni) / max(len(all_toks), 1),
        "distinct_2": len(bi) / max(sum(max(len(r) - 1, 0) for r in toks), 1),
        "repeat_4gram": repeat,
        "chars_mean": sum(len(s) for s in samples) / max(len(samples), 1),
    }, indent=2))


if __name__ == "__main__":
    main()

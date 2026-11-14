"""Held-out narrative-coherence benchmark for spiking checkpoints.

Each case has two context sentences, the true next sentence, and three
distractors from other held-out stories. The model picks the candidate with the
highest length-normalized log probability. Chance is 25%.

    python research/spiking_semantic_eval.py --models checkpoints/spiking_4m_10k \
      checkpoints/spiking_21m checkpoints/spiking_50m_run
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from streamft.spiking import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def candidate_score(model, tok, context, candidate, device, max_len=256):
    text = context.rstrip() + " " + candidate.strip()
    ctx_ids = tok.encode(context.rstrip()).ids
    ids = tok.encode(text).ids[:max_len]
    if len(ids) <= len(ctx_ids) + 1:
        return -1e9
    inp = torch.tensor([ids], device=device)
    logits = model(inp)
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    target = torch.tensor(ids[1:], device=device)
    vals = [float(logp[j - 1, target[j - 1]])
            for j in range(len(ctx_ids), len(ids))]
    if not vals:
        return -1e9
    return sum(vals) / len(vals)


@torch.no_grad()
def evaluate(model, tok, cases, device):
    correct = 0
    margins = []
    for case in cases:
        cands = [case["positive"]] + case["negatives"]
        scores = [candidate_score(model, tok, case["context"], c, device)
                  for c in cands]
        pred = max(range(4), key=lambda i: scores[i])
        correct += int(pred == 0)
        margins.append(scores[0] - max(scores[1:]))
    return {
        "n": len(cases),
        "accuracy": correct / max(len(cases), 1),
        "chance": 0.25,
        "mean_margin": sum(margins) / max(len(margins), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--cases", default=str(ROOT / "research/semantic_cases.json"))
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_semantic.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = payload["cases"][:args.limit] if args.limit else payload["cases"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for path in args.models:
        model, tok, meta = load_checkpoint(path)
        if tok is None:
            tok = Tokenizer.from_file(str(ROOT / "data/ts_bpe8k.json"))
        model.to(device)
        model.eval()
        result = evaluate(model, tok, cases, device)
        result.update({"model": path, "params": sum(p.numel() for p in model.parameters()),
                       "step": meta.get("step")})
        results.append(result)
        print(json.dumps(result), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": payload["n"], "results": results},
                              indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

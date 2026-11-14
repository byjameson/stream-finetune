"""Longer phase-B test: replay and replay+EWC on the 21M model.

Compared with the 300-step run, this uses 600 phase-B steps and also measures
greedy/sampled generation quality for each arm.

    python research/spiking_long_replay.py --steps-b 600
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from research.spiking_replay import RandomWindows
from research.spiking_replay_ewc import train_b
from research.spiking_shift import (
    clone_state,
    eval_both,
    fixed_batches,
    load_state,
    load_tokens,
)
from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    fisher_importance,
    generate,
)
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Once upon a time", "The little dog",
           "One day, a girl named Lily", "Tom and his friend"]


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def quality(samples):
    toks = [_tokens(s) for s in samples]
    flat = [t for row in toks for t in row]
    uni = {t for t in flat}
    bi = {g for row in toks for g in ngrams(row, 2)}
    four = [g for row in toks for g in ngrams(row, 4)]
    repeat = 0.0
    if four:
        counts = Counter(four)
        repeat = sum(c - 1 for c in counts.values() if c > 1) / len(four)
    return {
        "chars_mean": sum(len(s) for s in samples) / max(len(samples), 1),
        "distinct_1": len(uni) / max(len(flat), 1),
        "distinct_2": len(bi) / max(sum(max(len(r) - 1, 0) for r in toks), 1),
        "repeat_4gram": repeat,
        "samples": samples,
    }


def samples_for(model, tok, device, max_new=48):
    out = []
    for p in PROMPTS:
        out.append(generate(model, tok, p, max_new, device,
                            temperature=0.8, top_k=50,
                            repetition_penalty=1.1))
    return quality(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=300)
    ap.add_argument("--steps-b", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr-a", type=float, default=5e-4)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_long_replay.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tok = Tokenizer.from_file(str(ROOT / "data/ts_bpe8k.json"))
    a_train = load_tokens(str(ROOT / "data/ts_train.pt"), args.a_tokens)
    a_valid = load_tokens(str(ROOT / "data/ts_valid.pt"))
    b_train = load_tokens(str(ROOT / "data/ts_wt2_train.pt"))
    b_valid = load_tokens(str(ROOT / "data/ts_wt2_valid.pt"))
    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 100, device)
    new_batches = fixed_batches(b_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 200, device)

    model = SpikingLM(SpikingConfig(vocab=tok.get_vocab_size(), dim=args.dim,
                                    layers=args.layers, adapt=args.adapt)).to(device)
    t0 = time.perf_counter()
    _losses_a, _ = train_b(
        model, RandomWindows(a_train, args.seq, args.batch, args.seed, device),
        args.steps_a, args.lr_a, device, args.amp)
    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_both(model, old_batches, new_batches, tok)
    a_quality = samples_for(model, tok, device)
    importance = fisher_importance(model, old_batches[:args.fisher_batches])
    flat = torch.cat([v.float().flatten() for v in importance.values()])
    scale = float(flat.mean().clamp_min(1e-12))
    fisher = {k: (v / scale).to(device) for k, v in importance.items()}
    print(json.dumps({"event": "after_a", "params": sum(p.numel() for p in model.parameters()),
                      "sec": round(time.perf_counter() - t0, 1), "before": before,
                      "quality": {k: v for k, v in a_quality.items() if k != "samples"}}),
          flush=True)

    arms = [("naive", 0.0, 0.0), ("replay_20", args.replay, 0.0),
            ("replay_20_ewc", args.replay, args.ewc_lambda)]
    rows = []
    for name, replay, lam in arms:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 replay, args.seed + 1000, device)
        losses, sec = train_b(model, sampler, args.steps_b, args.lr_b,
                              device, args.amp,
                              anchor=anchor if lam else None,
                              fisher=fisher if lam else None,
                              ewc_lambda=lam)
        after = eval_both(model, old_batches, new_batches, tok)
        q = samples_for(model, tok, device)
        row = {"arm": name, "replay": replay, "ewc_lambda": lam,
               "train_loss": losses[-1], "sec": round(sec, 1),
               "before": before, "after": after,
               "quality": {k: v for k, v in q.items() if k != "samples"},
               "samples": q["samples"]}
        rows.append(row)
        print(json.dumps({"event": "arm", "arm": name,
                          "old": after["old"]["bpb"], "new": after["new"]["bpb"],
                          "quality": row["quality"], "sec": row["sec"]}), flush=True)

    payload = {
        "question": "Does replay-first hold with a longer phase B, and does soft EWC help?",
        "config": vars(args),
        "phase_a_quality": a_quality,
        "after_a": before,
        "arms": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

"""Does sleep-like replay protect an old task in the spiking LM?

Phase A trains on TinyStories. Phase B continues on WikiText with a mixture
sampler: a fraction of every batch comes from old TinyStories windows. The
control is the same run with replay=0. This is the brain-like "sleep/replay"
counterpart to post-hoc shadow rollback.

    python research/spiking_replay.py --steps-a 300 --steps-b 300
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from research.spiking_shift import (
    clone_state,
    eval_both,
    fixed_batches,
    load_state,
    load_tokens,
)
from streamft.spiking import SpikingConfig, SpikingLM
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]


class RandomWindows:
    def __init__(self, tokens, seq, batch, seed, device):
        self.tokens = tokens
        self.seq = seq
        self.batch = batch
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def next(self):
        hi = len(self.tokens) - self.seq - 1
        starts = torch.randint(0, hi, (self.batch,), generator=self.gen).tolist()
        xs = torch.stack([self.tokens[s:s + self.seq] for s in starts])
        ys = torch.stack([self.tokens[s + 1:s + self.seq + 1] for s in starts])
        return xs.to(self.device), ys.to(self.device)


def train_phase(model, sampler, steps, lr, device, amp=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        x, y = sampler.next()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=300)
    ap.add_argument("--steps-b", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=2e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--replays", default="0,0.2,0.5")
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_replay.json"))
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
    print(json.dumps({"event": "phase_a", "params": sum(p.numel() for p in model.parameters())}), flush=True)
    losses_a, sec_a = train_phase(
        model, RandomWindows(a_train, args.seq, args.batch, args.seed, device),
        args.steps_a, args.lr_a, device, args.amp)
    state_a = clone_state(model)
    before = eval_both(model, old_batches, new_batches, tok)
    print(json.dumps({"event": "after_a", "loss": losses_a[-1],
                      "sec": round(sec_a, 1), **before}), flush=True)

    results = []
    for frac in [float(x) for x in args.replays.split(",") if x.strip()]:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 frac, args.seed + 1000, device)
        losses_b, sec_b = train_phase(model, sampler, args.steps_b,
                                      args.lr_b, device, args.amp)
        after = eval_both(model, old_batches, new_batches, tok)
        row = {"replay": frac, "train_loss": losses_b[-1],
               "sec": round(sec_b, 1), "before": before, "after": after}
        results.append(row)
        print(json.dumps({"event": "arm", **row}), flush=True)

    payload = {
        "question": "Does sleep-like replay reduce old-task forgetting in the spiking LM?",
        "config": {
            "steps_a": args.steps_a, "steps_b": args.steps_b,
            "batch": args.batch, "seq": args.seq, "dim": args.dim,
            "layers": args.layers, "lr_a": args.lr_a, "lr_b": args.lr_b,
            "adapt": args.adapt, "a_tokens": args.a_tokens,
            "seed": args.seed, "amp": args.amp,
        },
        "after_a": before,
        "arms": results,
        "note": "Phase A TinyStories; phase B WikiText with old/new replay mix.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

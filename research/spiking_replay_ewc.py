"""21M controlled test: replay, replay+shadow rollback, replay+EWC.

Phase A is trained once per seed; every phase-B arm starts from the same
phase-A state. Arms:

  naive
  replay_20
  replay_20_rollback_10
  replay_20_ewc

    python research/spiking_replay_ewc.py --seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from research.spiking_replay import RandomWindows
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
    selective_rollback,
)
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]


def train_b(model, sampler, steps, lr, device, amp=False, anchor=None,
            fisher=None, ewc_lambda=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = sampler.next()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   y.reshape(-1))
        if ewc_lambda > 0 and anchor is not None and fisher is not None:
            pen = 0.0
            for name, p in model.named_parameters():
                pen = pen + (fisher[name] * (p.float() - anchor[name]) ** 2).sum()
            loss = loss + 0.5 * ewc_lambda * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def run_seed(seed: int, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tok = Tokenizer.from_file(str(ROOT / "data/ts_bpe8k.json"))
    a_train = load_tokens(str(ROOT / "data/ts_train.pt"), args.a_tokens)
    a_valid = load_tokens(str(ROOT / "data/ts_valid.pt"))
    b_train = load_tokens(str(ROOT / "data/ts_wt2_train.pt"))
    b_valid = load_tokens(str(ROOT / "data/ts_wt2_valid.pt"))
    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, seed + 100, device)
    new_batches = fixed_batches(b_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, seed + 200, device)

    model = SpikingLM(SpikingConfig(vocab=tok.get_vocab_size(), dim=args.dim,
                                    layers=args.layers, adapt=args.adapt)).to(device)
    losses_a, sec_a = train_b(
        model, RandomWindows(a_train, args.seq, args.batch, seed, device),
        args.steps_a, args.lr_a, device, args.amp)
    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_both(model, old_batches, new_batches, tok)
    importance = fisher_importance(model, old_batches[:args.fisher_batches])
    flat = torch.cat([v.float().flatten() for v in importance.values()])
    scale = float(flat.mean().clamp_min(1e-12))
    fisher = {k: (v / scale).to(device) for k, v in importance.items()}
    print(json.dumps({"event": "after_a", "seed": seed,
                      "params": sum(p.numel() for p in model.parameters()),
                      "sec": round(sec_a, 1), **before}), flush=True)

    arms = [
        ("naive", 0.0, 0.0, 0.0),
        ("replay_20", args.replay, 0.0, 0.0),
        ("replay_20_rollback_10", args.replay, args.rollback, 0.0),
        ("replay_20_ewc", args.replay, 0.0, args.ewc_lambda),
    ]
    rows = []
    for name, replay, rollback, lam in arms:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 replay, seed + 1000, device)
        losses_b, sec_b = train_b(model, sampler, args.steps_b, args.lr_b,
                                  device, args.amp,
                                  anchor=anchor if lam else None,
                                  fisher=fisher if lam else None,
                                  ewc_lambda=lam)
        state_b = clone_state(model)
        if rollback > 0:
            state_b = selective_rollback(state_a, state_b, importance,
                                         model, rollback)
            load_state(model, state_b)
        after = eval_both(model, old_batches, new_batches, tok)
        row = {"arm": name, "replay": replay, "rollback": rollback,
               "ewc_lambda": lam, "train_loss": losses_b[-1],
               "sec": round(sec_b, 1), "before": before, "after": after}
        rows.append(row)
        print(json.dumps({"event": "arm", "seed": seed, **row}), flush=True)
    return {"seed": seed, "after_a": before, "arms": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps-a", type=int, default=300)
    ap.add_argument("--steps-b", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr-a", type=float, default=5e-4)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--rollback", type=float, default=0.10)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_replay_ewc_21m.json"))
    args = ap.parse_args()

    results = [run_seed(int(s), args) for s in args.seeds.split(",") if s.strip()]
    payload = {
        "question": "Does replay-first hold at 21M, and does EWC beat hard rollback?",
        "config": vars(args),
        "seeds": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

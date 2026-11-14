"""Tune replay+EWC lambda on the 50M checkpoint.

Phase A is the existing 50M checkpoint; phase B is 600 WikiText steps with 20%
replay. Only the EWC lambda changes. Each arm is scored on old/new bpb and the
held-out narrative-coherence benchmark.

    python research/spiking_ewc_sweep_50m.py --lambdas 0.5,2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.spiking_replay_ewc import train_b
from research.spiking_shift import (
    clone_state,
    eval_both,
    fixed_batches,
    load_state,
    load_tokens,
)
from research.spiking_semantic_eval import evaluate as semantic_eval
from streamft.spiking import fisher_importance, load_checkpoint
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", default=str(ROOT / "checkpoints/spiking_50m_run"))
    ap.add_argument("--cases", default=str(ROOT / "research/semantic_cases.json"))
    ap.add_argument("--lambdas", default="0.5,2.0")
    ap.add_argument("--steps-b", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr-b", type=float, default=5e-5)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_ewc_sweep_50m.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, tok, meta = load_checkpoint(args.init_model)
    model.to(device)
    model.eval()
    a_train = load_tokens(str(ROOT / "data/ts_train.pt"), args.a_tokens)
    a_valid = load_tokens(str(ROOT / "data/ts_valid.pt"))
    b_train = load_tokens(str(ROOT / "data/ts_wt2_train.pt"))
    b_valid = load_tokens(str(ROOT / "data/ts_wt2_valid.pt"))
    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 100, device)
    new_batches = fixed_batches(b_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 200, device)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]

    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_both(model, old_batches, new_batches, tok)
    sem_a = semantic_eval(model, tok, cases, device)
    importance = fisher_importance(model, old_batches[:args.fisher_batches])
    flat = torch.cat([v.float().flatten() for v in importance.values()])
    scale = float(flat.mean().clamp_min(1e-12))
    fisher = {k: (v / scale).to(device) for k, v in importance.items()}
    print(json.dumps({"event": "after_a", "meta": meta, "before": before,
                      "semantic": sem_a}), flush=True)

    rows = []
    for lam in [float(x) for x in args.lambdas.split(",") if x.strip()]:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 args.replay, args.seed + 1000, device)
        _losses, sec = train_b(model, sampler, args.steps_b, args.lr_b,
                               device, args.amp, anchor=anchor, fisher=fisher,
                               ewc_lambda=lam)
        after = eval_both(model, old_batches, new_batches, tok)
        sem = semantic_eval(model, tok, cases, device)
        row = {"lambda": lam, "sec": round(sec, 1), "after": after,
               "semantic": sem}
        rows.append(row)
        print(json.dumps({"event": "arm", **row}), flush=True)

    payload = {"init_model": args.init_model, "before": before,
               "semantic_a": sem_a, "arms": rows,
               "config": vars(args)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

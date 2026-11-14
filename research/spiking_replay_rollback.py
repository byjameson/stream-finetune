"""Replay + Fisher shadow-rollback, controlled on one phase-A state.

Arms:
  naive                       phase B with replay=0, no rollback
  replay_20                   phase B with 20% old-data replay
  rollback_10                 phase B naive + post-hoc 10% Fisher rollback
  replay_20_rollback_10       phase B replay + post-hoc 10% Fisher rollback

    python research/spiking_replay_rollback.py --seed 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.spiking_replay import RandomWindows, train_phase
from research.spiking_shift import (
    clone_state,
    eval_both,
    fixed_batches,
    load_state,
    load_tokens,
)
from streamft.spiking import SpikingConfig, SpikingLM, fisher_importance, selective_rollback
from streamft.spiking_ingest import MixtureWindows

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--rollback", type=float, default=0.10)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_replay_rollback.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    from tokenizers import Tokenizer
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
    losses_a, sec_a = train_phase(
        model, RandomWindows(a_train, args.seq, args.batch, args.seed, device),
        args.steps_a, args.lr_a, device, args.amp)
    state_a = clone_state(model)
    before = eval_both(model, old_batches, new_batches, tok)
    importance = fisher_importance(model, old_batches[:args.fisher_batches])
    print(json.dumps({"event": "after_a", "params": sum(p.numel() for p in model.parameters()),
                      "loss": losses_a[-1], "sec": round(sec_a, 1), **before}), flush=True)

    arms = [
        ("naive", 0.0, 0.0),
        (f"replay_{int(args.replay*100)}", args.replay, 0.0),
        (f"rollback_{int(args.rollback*100)}", 0.0, args.rollback),
        (f"replay_{int(args.replay*100)}_rollback_{int(args.rollback*100)}",
         args.replay, args.rollback),
    ]
    results = []
    for name, replay, rollback in arms:
        load_state(model, state_a)
        sampler = MixtureWindows(a_train, b_train, args.seq, args.batch,
                                 replay, args.seed + 1000, device)
        losses_b, sec_b = train_phase(model, sampler, args.steps_b,
                                      args.lr_b, device, args.amp)
        state_b = clone_state(model)
        if rollback > 0:
            state_b = selective_rollback(state_a, state_b, importance,
                                         model, rollback)
            load_state(model, state_b)
        after = eval_both(model, old_batches, new_batches, tok)
        row = {"arm": name, "replay": replay, "rollback": rollback,
               "train_loss": losses_b[-1], "sec": round(sec_b, 1),
               "before": before, "after": after}
        results.append(row)
        print(json.dumps({"event": "arm", **row}), flush=True)

    payload = {
        "question": "Does 20% replay plus Fisher rollback beat either mechanism alone?",
        "config": {
            "steps_a": args.steps_a, "steps_b": args.steps_b,
            "batch": args.batch, "seq": args.seq, "dim": args.dim,
            "layers": args.layers, "lr_a": args.lr_a, "lr_b": args.lr_b,
            "adapt": args.adapt, "seed": args.seed, "amp": args.amp,
            "a_tokens": args.a_tokens,
        },
        "after_a": before,
        "arms": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

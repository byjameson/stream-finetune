"""Continual binding: learn new triples without forgetting old ones.

Phase A trains the statement-memory spiking model on 32 (person, color, object)
triples. Phase B trains on 16 disjoint triples. A separate 16-triple split is
never trained. Arms compare the retention mechanisms from the TinyStories ->
WikiText work on this new data:

    naive
    replay_20
    rollback_10
    replay_20_rollback_10
    replay_20_ewc

    python research/spiking_binding_continual.py --steps-a 3000 --steps-b 3000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from research.binding_baselines import binding_aux_loss, evaluate
from research.spiking_binding import COLORS, LEN, NAMES, OBJECTS, WORDS, make_split
from research.spiking_binding_conj import make_data
from research.spiking_shift import clone_state, load_state
from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    fisher_importance,
    selective_rollback,
)

ROOT = Path(__file__).resolve().parents[1]


def train_phase(model, new_data, old_data, steps, batch, lr, device, amp,
                replay=0.0, ewc_lambda=0.0, anchor=None, fisher=None,
                aux_weight=0.5, aux_attn=0.5, aux_pos=19, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    n_old = int(round(batch * replay))
    n_new = batch - n_old
    for step in range(1, steps + 1):
        idx_new = torch.randint(0, len(new_data), (n_new,))
        parts = [new_data[idx_new].to(device)]
        if n_old:
            idx_old = torch.randint(0, len(old_data), (n_old,))
            parts.append(old_data[idx_old].to(device))
        x = torch.cat(parts, dim=0)
        x = x[torch.randperm(len(x), device=device)]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(WORDS)),
                                   x[:, 1:].reshape(-1))
            loss = loss + binding_aux_loss(model, x, aux_pos,
                                           copy_weight=aux_weight,
                                           attn_weight=aux_attn)
            if ewc_lambda and anchor is not None and fisher is not None:
                pen = torch.zeros((), device=device)
                for name, p in model.named_parameters():
                    pen = pen + (fisher[name] * (p.float() - anchor[name]) ** 2).sum()
                loss = loss + 0.5 * ewc_lambda * pen
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def eval_sets(model, sets, device):
    out = {}
    for name, data in sets.items():
        ev = evaluate(model, data, device, LEN)
        ev.pop("pred", None)
        out[name] = ev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=3000)
    ap.add_argument("--steps-b", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--aux-attn-weight", type=float, default=0.5)
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--rollback", type=float, default=0.10)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--fisher-batch-size", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-split", choices=["none", "colors"], default="none",
                    help="none = disjoint triples; colors = old colors -> new color")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out",
                    default=str(ROOT / "research/results/spiking_binding_continual.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.hard_split == "colors":
        old_triples = [(n, c, o) for n in NAMES for c in COLORS[:2]
                       for o in OBJECTS]
        new_triples = [(n, c, o) for n in NAMES for c in COLORS[2:]
                       for o in OBJECTS[:2]]
        held_triples = [(n, c, o) for n in NAMES for c in COLORS[2:]
                        for o in OBJECTS[2:]]
    else:
        train_triples, held_triples = make_split(args.seed)
        old_triples = train_triples[:32]
        new_triples = train_triples[32:]
    old_train = make_data(args.n_train, old_triples, args.seed + 11)
    new_train = make_data(args.n_train, new_triples, args.seed + 12)
    sets = {
        "old": make_data(args.n_test, old_triples, args.seed + 13),
        "new": make_data(args.n_test, new_triples, args.seed + 14),
        "held": make_data(args.n_test, held_triples, args.seed + 15),
    }
    print(json.dumps({"hard_split": args.hard_split,
                      "old_triples": len(old_triples),
                      "new_triples": len(new_triples),
                      "held_triples": len(held_triples),
                      "train": [len(old_train), len(new_train)],
                      "test": {k: len(v) for k, v in sets.items()}}), flush=True)

    model = SpikingLM(SpikingConfig(
        vocab=len(WORDS), dim=args.dim, layers=args.layers, mode="spike",
        adapt=args.adapt, memory=True, mem_read_pos=19)).to(device)

    losses_a, sec_a = train_phase(
        model, old_train, old_train, args.steps_a, args.batch, args.lr_a,
        device, args.amp, aux_weight=args.aux_weight,
        aux_attn=args.aux_attn_weight, tag="A")
    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_sets(model, sets, device)
    print(json.dumps({"event": "after_a", "sec": round(sec_a, 1),
                      "train_loss": losses_a[-1], "eval": before}), flush=True)

    fisher_batches = []
    for _ in range(args.fisher_batches):
        idx = torch.randint(0, len(old_train), (args.fisher_batch_size,))
        xb = old_train[idx].to(device)
        fisher_batches.append((xb, xb))
    importance = fisher_importance(model, fisher_batches)
    flat = torch.cat([v.float().flatten() for v in importance.values()])
    scale = float(flat.mean().clamp_min(1e-12))
    fisher = {k: (v / scale).to(device) for k, v in importance.items()}

    arms = [
        ("naive", 0.0, 0.0, 0.0),
        ("replay_20", args.replay, 0.0, 0.0),
        ("rollback_10", 0.0, args.rollback, 0.0),
        ("replay_20_rollback_10", args.replay, args.rollback, 0.0),
        ("replay_20_ewc", args.replay, 0.0, args.ewc_lambda),
    ]
    rows = []
    for name, replay, rollback, ewc in arms:
        load_state(model, state_a)
        losses_b, sec_b = train_phase(
            model, new_train, old_train, args.steps_b, args.batch, args.lr_b,
            device, args.amp, replay=replay, ewc_lambda=ewc,
            anchor=anchor if ewc else None, fisher=fisher if ewc else None,
            aux_weight=args.aux_weight, aux_attn=args.aux_attn_weight,
            tag=name)
        state_b = clone_state(model)
        if rollback:
            state_b = selective_rollback(state_a, state_b, importance,
                                         model, rollback)
            load_state(model, state_b)
        after = eval_sets(model, sets, device)
        row = {"arm": name, "replay": replay, "rollback": rollback,
               "ewc_lambda": ewc, "sec": round(sec_b, 1),
               "train_loss": losses_b[-1], "after": after}
        rows.append(row)
        print(json.dumps({"event": "arm", **row}), flush=True)

    payload = {
        "question": "Do replay/shadow rollback/EWC retain old binding after new triples?",
        "config": vars(args),
        "split": {"hard_split": args.hard_split,
                  "old_triples": len(old_triples),
                  "new_triples": len(new_triples),
                  "held_triples": len(held_triples)},
        "before": before,
        "arms": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

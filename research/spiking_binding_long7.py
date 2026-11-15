"""Four-statement binding: longer stories, color split, replay/rollback/EWC.

The 3-statement task is solved by the statement memory + matching attention.
This script asks whether the same machinery scales to four statements and a
no-overlap color shift, and whether replay still retains old/new binding.

    python research/spiking_binding_long.py --n-statements 4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from research.binding_baselines import binding_aux_loss
from research.spiking_binding import COLORS, NAMES, OBJECTS, W2I, WORDS

NAMES7 = NAMES + ["zoe", "max", "ben"]
WORDS7 = WORDS + ["zoe", "max", "ben"]
W2I7 = {w: i for i, w in enumerate(WORDS7)}
VOCAB7 = len(WORDS7)
from research.spiking_shift import clone_state, load_state
from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    fisher_importance,
    selective_rollback,
)

ROOT = Path(__file__).resolve().parents[1]


def seq_len(n_stmt):
    return 5 * n_stmt + 7


def ans_pos(n_stmt):
    return seq_len(n_stmt) - 3


def sample_story(allowed, rng, n_stmt):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES7}
    avail = [n for n in NAMES7 if by_name[n]]
    for _ in range(2000):
        names = rng.sample(avail, min(n_stmt, len(avail)))
        if len(names) < n_stmt:
            raise RuntimeError("not enough names for this story")
        stmts = [rng.choice(by_name[n]) for n in names]
        rng.shuffle(stmts)
        q = rng.choice(stmts)
        others = [s for s in stmts if s != q]
        same_color = any(s[1] == q[1] for s in others)
        same_object = any(s[2] == q[2] for s in others)
        pair_unique = all(not (s[1] == q[1] and s[2] == q[2]) for s in others)
        if same_color and same_object and pair_unique:
            toks = []
            for n, c, o in stmts:
                toks += [n, "has", c, o, "."]
            toks += ["who", "has", q[1], q[2], "?", q[0], "<eos>"]
            assert len(toks) == seq_len(n_stmt)
            return [W2I7[t] for t in toks]
    raise RuntimeError("could not build a 4-statement story")


def make_data(n, allowed, seed, n_stmt):
    rng = random.Random(seed)
    return torch.tensor([sample_story(allowed, rng, n_stmt)
                         for _ in range(n)], dtype=torch.long)


def permute_stories(test, n_stmt):
    out = test.clone()
    for i in range(len(test)):
        row = test[i]
        stmts = [row[5 * j:5 * j + 5] for j in range(n_stmt)]
        query = row[5 * n_stmt:5 * n_stmt + 5]
        ans = row[5 * n_stmt + 5:5 * n_stmt + 7]
        out[i] = torch.cat([stmts[j] for j in reversed(range(n_stmt))]
                           + [query, ans])
    return out




def evaluate7(model, test, device, n_stmt):
    length = seq_len(n_stmt)
    pos = ans_pos(n_stmt)
    model.eval()
    with torch.no_grad():
        out = model(test.to(device))
        pred = out[:, pos].argmax(-1)
        ans = test[:, length - 2].to(device)
        acc = (pred == ans).float().mean().item()
        by_pos = {}
        for i in range(n_stmt):
            p = 5 * i
            mask = test[:, p].to(device) == ans
            if int(mask.sum()) > 0:
                by_pos[f"answer_at_stmt{i}"] = {
                    "n": int(mask.sum()),
                    "acc": float((pred[mask] == ans[mask]).float().mean().item()),
                }
        counts = pred.cpu().bincount(minlength=VOCAB7).tolist()
    model.train()
    return {"accuracy": acc, "by_answer_pos": by_pos,
            "pred_counts": {WORDS7[i]: c for i, c in enumerate(counts) if c}}


def majority_baseline(data, n_stmt):
    ans = data[:, seq_len(n_stmt) - 2]
    counts = torch.bincount(ans, minlength=VOCAB7)
    return float(counts.max()) / max(len(ans), 1)


def eval_sets(model, sets, device, n_stmt):
    out = {}
    for name, data in sets.items():
        ev = evaluate7(model, data, device, n_stmt)
        ev.pop("pred", None)
        out[name] = ev
    return out


def train_phase(model, new_data, old_data, steps, batch, lr, device, amp,
                n_stmt, replay=0.0, ewc_lambda=0.0, anchor=None, fisher=None,
                aux_weight=0.5, aux_attn=0.5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    n_old = int(round(batch * replay))
    n_new = batch - n_old
    pos = ans_pos(n_stmt)
    for _ in range(steps):
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
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB7),
                                   x[:, 1:].reshape(-1))
            loss = loss + binding_aux_loss(model, x, pos,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-statements", type=int, default=7)
    ap.add_argument("--steps-a", type=int, default=4000)
    ap.add_argument("--steps-b", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--aux-attn-weight", type=float, default=0.5)
    ap.add_argument("--slots", type=int, default=7,
                    help="per-statement memory slots (0 = dynamic memory)")
    ap.add_argument("--slot-heads", type=int, default=4)
    ap.add_argument("--slot-attr-keys", action="store_true", default=True,
                    help="use the statement color+object embeddings as slot keys")
    ap.add_argument("--replay", type=float, default=0.2)
    ap.add_argument("--rollback", type=float, default=0.10)
    ap.add_argument("--ewc-lambda", type=float, default=1.0)
    ap.add_argument("--arms", default="naive,replay_20")
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--fisher-batch-size", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_binding_long7.json"))
    args = ap.parse_args()

    n = args.n_statements
    length = seq_len(n)
    pos = ans_pos(n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    old_triples = [(nm, c, o) for nm in NAMES7 for c in COLORS[:2]
                   for o in OBJECTS]
    new_triples = [(nm, c, o) for nm in NAMES7 for c in COLORS[2:]
                   for o in OBJECTS[:2]]
    held_triples = [(nm, c, o) for nm in NAMES7 for c in COLORS[2:]
                    for o in OBJECTS[2:]]
    old_train = make_data(args.n_train, old_triples, args.seed + 11, n)
    new_train = make_data(args.n_train, new_triples, args.seed + 12, n)
    sets = {
        "old": make_data(args.n_test, old_triples, args.seed + 13, n),
        "new": make_data(args.n_test, new_triples, args.seed + 14, n),
        "held": make_data(args.n_test, held_triples, args.seed + 15, n),
    }
    print(json.dumps({"n_statements": n, "seq_len": length, "ans_pos": pos,
                      "old_triples": len(old_triples),
                      "new_triples": len(new_triples),
                      "held_triples": len(held_triples),
                      "majority": {k: majority_baseline(v, n)
                                   for k, v in sets.items()}}), flush=True)

    model = SpikingLM(SpikingConfig(
        vocab=VOCAB7, dim=args.dim, layers=args.layers, mode="spike",
        adapt=args.adapt, memory=args.slots == 0, slots=args.slots,
        slot_heads=args.slot_heads, slot_attr_keys=args.slot_attr_keys,
        mem_read_pos=pos)).to(device)
    losses_a, sec_a = train_phase(
        model, old_train, old_train, args.steps_a, args.batch, args.lr,
        device, args.amp, n, aux_weight=args.aux_weight,
        aux_attn=args.aux_attn_weight)
    state_a = clone_state(model)
    anchor = {k: v.to(device) for k, v in state_a.items()}
    before = eval_sets(model, sets, device, n)
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

    arms = {
        "naive": (0.0, 0.0, 0.0),
        "replay_20": (args.replay, 0.0, 0.0),
        "rollback_10": (0.0, args.rollback, 0.0),
        "replay_20_rollback_10": (args.replay, args.rollback, 0.0),
        "replay_20_ewc": (args.replay, 0.0, args.ewc_lambda),
    }
    rows = []
    for name in [x.strip() for x in args.arms.split(",") if x.strip()]:
        replay, rollback, ewc = arms[name]
        load_state(model, state_a)
        losses_b, sec_b = train_phase(
            model, new_train, old_train, args.steps_b, args.batch, args.lr,
            device, args.amp, n, replay=replay, ewc_lambda=ewc,
            anchor=anchor if ewc else None, fisher=fisher if ewc else None,
            aux_weight=args.aux_weight, aux_attn=args.aux_attn_weight)
        state_b = clone_state(model)
        if rollback:
            state_b = selective_rollback(state_a, state_b, importance,
                                         model, rollback)
            load_state(model, state_b)
        after = eval_sets(model, sets, device, n)
        perm = permute_stories(sets["old"], n)
        ev_perm = evaluate7(model, perm, device, n)
        ev_perm.pop("pred", None)
        row = {"arm": name, "replay": replay, "rollback": rollback,
               "ewc_lambda": ewc, "sec": round(sec_b, 1),
               "train_loss": losses_b[-1], "after": after,
               "old_permuted": ev_perm}
        rows.append(row)
        print(json.dumps({"event": "arm", "arm": name,
                          "old": after["old"]["accuracy"],
                          "new": after["new"]["accuracy"],
                          "held": after["held"]["accuracy"],
                          "old_perm": ev_perm["accuracy"]}), flush=True)

    payload = {"question": "Does four-statement binding retain replay-first?",
               "config": vars(args), "split": {"old": len(old_triples),
                                               "new": len(new_triples),
                                               "held": len(held_triples)},
               "before": before, "arms": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

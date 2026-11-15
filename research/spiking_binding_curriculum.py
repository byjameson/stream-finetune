"""Curriculum for the statement memory without the matching-attention loss.

The copy-loss-only model fails on the full 3-statement task at 6k steps
(spike 0.308, rate 0.696) and the spike model is still position-biased at 12k.
This script trains the same memory model on a curriculum instead:

    1 statement  -> 2 statements -> 3 statements

The sequence length stays 22 and the query/answer positions stay 15-21; missing
statements are replaced by a pad token between the statements and the query. The
memory read is trained only by the next-token copy loss (no matching-attention
auxiliary loss).

    python research/spiking_binding_curriculum.py --arms spike,rate
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
from research.spiking_binding import (
    LEN,
    NAMES,
    W2I,
    WORDS,
    make_split,
)
from research.spiking_binding_conj import make_data, permute_test
from streamft.spiking import SpikingConfig, SpikingLM

ROOT = Path(__file__).resolve().parents[1]
PAD = len(WORDS)
VOCAB = len(WORDS) + 1
ANS_POS = LEN - 3  # 19


def sample_cur_story(allowed, rng, n_stmt):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES}
    for _ in range(1000):
        names = rng.sample(NAMES, n_stmt)
        stmts = [rng.choice(by_name[n]) for n in names]
        pairs = [(c, o) for _, c, o in stmts]
        if len(set(pairs)) != n_stmt:
            continue
        rng.shuffle(stmts)
        q = rng.choice(stmts)
        toks = []
        for n, c, o in stmts:
            toks += [n, "has", c, o, "."]
        toks += ["<pad>"] * (5 * (3 - n_stmt))
        toks += ["who", "has", q[1], q[2], "?", q[0], "<eos>"]
        assert len(toks) == LEN
        return [W2I[t] if t != "<pad>" else PAD for t in toks]
    raise RuntimeError("could not build curriculum story")


def make_cur_data(n, allowed, seed, n_stmt):
    rng = random.Random(seed)
    return torch.tensor([sample_cur_story(allowed, rng, n_stmt)
                         for _ in range(n)], dtype=torch.long)


def evaluate_cur(model, test, device):
    model.eval()
    with torch.no_grad():
        out = model(test.to(device))
        pred = out[:, ANS_POS].argmax(-1)
        ans = test[:, LEN - 2].to(device)
        acc = (pred == ans).float().mean().item()
        name_pos = [0, 5, 10]
        by_pos = {}
        for i, p in enumerate(name_pos):
            mask = test[:, p].to(device) == ans
            if int(mask.sum()) > 0:
                by_pos[f"answer_at_stmt{i}"] = {
                    "n": int(mask.sum()),
                    "acc": float((pred[mask] == ans[mask]).float().mean()),
                }
        words = WORDS + ["<pad>"]
        counts = pred.cpu().bincount(minlength=VOCAB).tolist()
    model.train()
    return {"accuracy": acc, "by_answer_pos": by_pos,
            "pred_counts": {words[i]: c for i, c in enumerate(counts) if c}}


def train_stage(model, data, steps, batch, lr, device, amp, aux_weight=0.5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(0, len(data), (batch,))
        x = data[idx].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            mask = x[:, 1:] != PAD
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB),
                x[:, 1:].reshape(-1), reduction="none").reshape(x.size(0), -1)
            loss = (ce * mask).sum() / mask.sum().clamp_min(1)
            loss = loss + binding_aux_loss(model, x, ANS_POS,
                                           copy_weight=aux_weight, attn_weight=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def run_arm(mode, train_triples, held_triples, args, device):
    torch.manual_seed(args.seed)
    model = SpikingLM(SpikingConfig(
        vocab=VOCAB, dim=args.dim, layers=args.layers, mode=mode,
        adapt=args.adapt, memory=True, mem_read_pos=ANS_POS)).to(device)
    test3 = make_data(args.n_test, held_triples, args.seed + 30)
    perm3 = permute_test(test3)
    stages = []
    total_sec = 0.0
    for n_stmt, steps in ((1, args.steps_1), (2, args.steps_2),
                          (3, args.steps_3)):
        if n_stmt == 3:
            data = make_data(args.n_train, train_triples,
                             args.seed + 10 + n_stmt)
        else:
            data = make_cur_data(args.n_train, train_triples,
                                 args.seed + 10 + n_stmt, n_stmt)
        losses, sec = train_stage(model, data, steps, args.batch, args.lr,
                                  device, args.amp, args.aux_weight)
        total_sec += sec
        ev = evaluate_cur(model, test3, device)
        print(json.dumps({"arm": mode, "stage": n_stmt, "steps": steps,
                          "loss": round(losses[-1], 4),
                          "test3_acc": round(ev["accuracy"], 4),
                          "sec": round(sec, 1)}), flush=True)
        stages.append({"n_statements": n_stmt, "steps": steps,
                       "train_loss": losses[-1], "eval3": ev,
                       "sec": round(sec, 1)})
    ev = evaluate_cur(model, test3, device)
    ev_perm = evaluate_cur(model, perm3, device)
    return {
        "arm": mode,
        "params": sum(p.numel() for p in model.parameters()),
        "test_acc": ev["accuracy"],
        "permuted_acc": ev_perm["accuracy"],
        "eval": ev, "permuted_eval": ev_perm,
        "stages": stages,
        "sec": round(total_sec, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="spike,rate")
    ap.add_argument("--steps-1", type=int, default=2000)
    ap.add_argument("--steps-2", type=int, default=2000)
    ap.add_argument("--steps-3", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out",
                    default=str(ROOT / "research/results/spiking_binding_curriculum.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_triples, held_triples = make_split(args.seed)
    print(json.dumps({"train_triples": len(train_triples),
                      "held_triples": len(held_triples),
                      "vocab": VOCAB, "pad": PAD}), flush=True)
    rows = [run_arm(m.strip(), train_triples, held_triples, args, device)
            for m in args.arms.split(",") if m.strip()]
    payload = {"question": "Can a 1->2->3 statement curriculum replace the "
                           "matching-attention auxiliary loss?",
               "config": vars(args), "results": rows,
               "note": "Pads fill the missing statements; query/answer positions are fixed."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

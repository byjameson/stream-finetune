"""Stronger variable-binding benchmark: conjunction ambiguity.

The first binding task (`spiking_binding.py`) had two benchmark flaws:

- a single attribute is enough: matching the query color alone gets ~80%,
  matching the object alone ~78%;
- the same (color, object) pair can appear twice in one story, so even perfect
  pair matching is ambiguous in ~7% of cases.

This version forces **conjunction binding**: for every story, the query's color
appears in at least one other statement and the query's object appears in at
least one other statement, but the (color, object) pair is unique. Neither
attribute alone identifies the answer; the model must bind both.

    python research/spiking_binding_conj.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from research.binding_baselines import TinyTransformer, evaluate, train_lm
from research.spiking_binding import LEN, NAMES, W2I, WORDS, make_split
from streamft.spiking import SpikingConfig, SpikingLM

ROOT = Path(__file__).resolve().parents[1]


def sample_story_conj(allowed, rng, n_statements=3):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES}
    for _ in range(1000):
        names = rng.sample(NAMES, n_statements)
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
            assert len(toks) == LEN
            return [W2I[t] for t in toks]
    raise RuntimeError("could not build a conjunction-ambiguous story")


def make_data(n, allowed, seed):
    rng = random.Random(seed)
    return torch.tensor([sample_story_conj(allowed, rng) for _ in range(n)],
                        dtype=torch.long)


def heuristic_baselines(test):
    # One row: 3 statements x 5 tokens, then who has C O ? NAME <eos>.
    n = len(test)
    pos = [0, 0, 0]
    color = 0
    obj = 0
    pair = 0
    for i in range(n):
        row = test[i].tolist()
        names = [row[0], row[5], row[10]]
        colors = [row[2], row[7], row[12]]
        objects = [row[3], row[8], row[13]]
        qc, qo, ans = row[17], row[18], row[20]
        for j in range(3):
            pos[j] += int(names[j] == ans)
        ci = next((j for j in range(3) if colors[j] == qc), None)
        oi = next((j for j in range(3) if objects[j] == qo), None)
        pi = next((j for j in range(3)
                   if colors[j] == qc and objects[j] == qo), None)
        color += int(ci is not None and names[ci] == ans)
        obj += int(oi is not None and names[oi] == ans)
        pair += int(pi is not None and names[pi] == ans)
    return {
        "position": [round(x / n, 4) for x in pos],
        "color_only": round(color / n, 4),
        "object_only": round(obj / n, 4),
        "pair_match": round(pair / n, 4),
    }


def run_arm(name, args, train, test, device):
    torch.manual_seed(args.seed)
    if name == "transformer":
        model = TinyTransformer(len(WORDS), args.dim, args.layers,
                                args.heads, dropout=args.dropout).to(device)
    else:
        model = SpikingLM(SpikingConfig(vocab=len(WORDS), dim=args.dim,
                                        layers=args.layers, mode=name,
                                        adapt=args.adapt,
                                        attention=args.attention,
                                        pointer_offset=args.pointer_offset,
                                        dropout=args.dropout)).to(device)
    loss, sec = train_lm(model, train, args.steps, args.batch, args.lr,
                         device, args.amp, args.eval_every, test, LEN,
                         tag=name)
    ev = evaluate(model, test, device, LEN)
    return {"arm": name, "test_acc": ev["accuracy"], "eval": ev,
            "train_loss": loss, "sec": round(sec, 1),
            "params": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="spike,rate,transformer")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--attention", action="store_true",
                    help="add causal attention over the spiking history")
    ap.add_argument("--pointer-offset", type=int, default=0,
                    help="copy the token this many positions before the key")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out",
                    default=str(ROOT / "research/results/spiking_binding_conj.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_triples, test_triples = make_split(args.seed)
    train = make_data(args.n_train, train_triples, args.seed + 1)
    test = make_data(args.n_test, test_triples, args.seed + 2)
    base = heuristic_baselines(test)
    print(json.dumps({"train": len(train), "test": len(test), "len": LEN,
                      "vocab": len(WORDS), "heuristics": base}), flush=True)
    rows = [run_arm(name.strip(), args, train, test, device)
            for name in args.arms.split(",") if name.strip()]
    payload = {"task": "conjunction variable binding", "config": vars(args),
               "heuristics": base, "chance": 0.25, "results": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

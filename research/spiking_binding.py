"""Non-TinyStories semantic task: variable binding in minature stories.

Each story has three statements: a person has a color and an object. The question
asks who has one color/object pair; the answer is the person. Train and test use
disjoint (person, color, object) triples, so the model must bind the answer from
the context rather than memorize a global pair.

    python research/spiking_binding.py --steps 3000 --mode both
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from streamft.spiking import SpikingConfig, SpikingLM

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["tom", "mia", "sam", "lily"]
COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["ball", "cup", "book", "car"]
WORDS = NAMES + COLORS + OBJECTS + ["has", "who", "?", ".", "<eos>"]
W2I = {w: i for i, w in enumerate(WORDS)}
I2W = {i: w for w, i in W2I.items()}
LEN = 3 * 5 + 5 + 1 + 1  # 22


def make_split(seed=0):
    rng = random.Random(seed)
    train, test = [], []
    for n in NAMES:
        triples = [(n, c, o) for c in COLORS for o in OBJECTS]
        rng.shuffle(triples)
        train.extend(triples[:12])
        test.extend(triples[12:])
    rng.shuffle(train)
    rng.shuffle(test)
    assert len(train) == 48 and len(test) == 16
    return train, test


def sample_story(allowed, rng):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES}
    names = rng.sample(NAMES, 3)
    stmts = [rng.choice(by_name[n]) for n in names]
    rng.shuffle(stmts)
    q = rng.choice(stmts)
    toks = []
    for n, c, o in stmts:
        toks += [n, "has", c, o, "."]
    toks += ["who", "has", q[1], q[2], "?", q[0], "<eos>"]
    assert len(toks) == LEN
    return [W2I[t] for t in toks]


def make_data(n, allowed, seed):
    rng = random.Random(seed)
    return torch.tensor([sample_story(allowed, rng) for _ in range(n)],
                        dtype=torch.long)


def train_one(mode, train, test, args, device):
    torch.manual_seed(args.seed)
    model = SpikingLM(SpikingConfig(vocab=len(WORDS), dim=args.dim,
                                    layers=args.layers, mode=mode,
                                    adapt=args.adapt)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    model.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train), (args.batch,))
        x = train[idx].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(args.amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(WORDS)),
                                   x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                out = model(test.to(device))
                pred = out[:, LEN - 3].argmax(-1)
                acc = (pred == test[:, LEN - 2].to(device)).float().mean().item()
            print(json.dumps({"mode": mode, "step": step,
                              "loss": round(float(loss.detach()), 4),
                              "test_acc": round(acc, 4)}), flush=True)
            model.train()
    return {
        "mode": mode,
        "params": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "test_acc": acc,
        "chance": 0.25,
        "sec": round(time.perf_counter() - t0, 1),
        "final_loss": float(loss.detach()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spike", "rate", "both"], default="both")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_binding.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_triples, test_triples = make_split(args.seed)
    train = make_data(args.n_train, train_triples, args.seed + 1)
    test = make_data(args.n_test, test_triples, args.seed + 2)
    print(json.dumps({"train": len(train), "test": len(test),
                      "len": LEN, "vocab": len(WORDS),
                      "train_triples": len(train_triples),
                      "test_triples": len(test_triples)}), flush=True)
    modes = ["spike", "rate"] if args.mode == "both" else [args.mode]
    results = [train_one(m, train, test, args, device) for m in modes]
    payload = {"task": "tiny variable binding", "config": vars(args),
               "results": results,
               "note": "Train/test (person,color,object) triples are disjoint."}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

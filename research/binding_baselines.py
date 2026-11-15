"""Binding baselines: tiny transformer and a 2->3 statement curriculum.

The variable-binding task in `spiking_binding.py` is too hard for the current
17-word recurrent spiking model on the disjoint split. This script asks two
follow-up questions:

1. Is the task learnable at all at this budget? A tiny causal transformer is
   trained on the same data/split as a control.
2. Does a curriculum help the spiking model? Train first on two-statement
   stories (one distractor) and then on the three-statement test format.

    python research/binding_baselines.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.spiking_binding import (
    LEN,
    NAMES,
    W2I,
    WORDS,
    make_split,
)
from streamft.spiking import SpikingConfig, SpikingLM

ROOT = Path(__file__).resolve().parents[1]


def sample_story_n(allowed, rng, n_statements):
    by_name = {n: [t for t in allowed if t[0] == n] for n in NAMES}
    names = rng.sample(NAMES, n_statements)
    stmts = [rng.choice(by_name[n]) for n in names]
    rng.shuffle(stmts)
    q = rng.choice(stmts)
    toks = []
    for n, c, o in stmts:
        toks += [n, "has", c, o, "."]
    toks += ["who", "has", q[1], q[2], "?", q[0], "<eos>"]
    return [W2I[t] for t in toks]


def make_data(n, allowed, seed, n_statements):
    rng = random.Random(seed)
    return torch.tensor([sample_story_n(allowed, rng, n_statements)
                         for _ in range(n)], dtype=torch.long)


def answer_position(seq_len):
    return seq_len - 3


def evaluate(model, test, device, seq_len):
    model.eval()
    with torch.no_grad():
        out = model(test.to(device))
        pred = out[:, answer_position(seq_len)].argmax(-1)
        ans = test[:, seq_len - 2].to(device)
        acc = (pred == ans).float().mean().item()
        n_stmt = (seq_len - 7) // 5
        name_pos = [5 * i for i in range(n_stmt)]
        pos_base = {f"stmt{i}": float((test[:, p].to(device) == ans)
                                      .float().mean().item())
                    for i, p in enumerate(name_pos)}
        by_pos = {}
        for i, p in enumerate(name_pos):
            mask = test[:, p].to(device) == ans
            if int(mask.sum()) > 0:
                by_pos[f"answer_at_stmt{i}"] = {
                    "n": int(mask.sum()),
                    "acc": float((pred[mask] == ans[mask]).float().mean().item()),
                }
        pred_counts = pred.cpu().bincount(minlength=len(WORDS)).tolist()
    model.train()
    return {"accuracy": acc, "pos_baseline": pos_base, "by_answer_pos": by_pos,
            "pred_counts": {WORDS[i]: c for i, c in enumerate(pred_counts) if c},
            "pred": pred.cpu()}


def binding_aux_loss(model, x, aux_pos, copy_weight=0.0, attn_weight=0.0):
    loss = torch.zeros((), device=x.device)
    if copy_weight and getattr(model, "last_copy_logits", None) is not None:
        copy_logits = model.last_copy_logits
        if copy_logits.dim() == 3:
            copy_logits = copy_logits[:, aux_pos]
        loss = loss + copy_weight * F.cross_entropy(copy_logits,
                                                    x[:, aux_pos + 1])
    if attn_weight and getattr(model, "last_mem_attn", None) is not None:
        # Self-supervised target: match the query (color, object) to the
        # statement with the same pair. No answer token is used here.
        qc = x[:, aux_pos - 2]
        qo = x[:, aux_pos - 1]
        n_stmt = (aux_pos - 4) // 5
        target = torch.full_like(qc, 4)
        for i in range(n_stmt):
            color = x[:, 5 * i + 2]
            obj = x[:, 5 * i + 3]
            match = (color == qc) & (obj == qo)
            target = torch.where(
                match, torch.full_like(target, 5 * i + 4), target)
        loss = loss + attn_weight * F.cross_entropy(model.last_mem_attn,
                                                    target)
    return loss


def train_lm(model, train, steps, batch, lr, device, amp, eval_every,
             test=None, seq_len=None, tag="", aux_weight=0.0, aux_pos=None,
             aux_attn_weight=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0 = time.perf_counter()
    loss = torch.tensor(0.0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (batch,))
        x = train[idx].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(WORDS)),
                                   x[:, 1:].reshape(-1))
            if aux_pos is not None:
                loss = loss + binding_aux_loss(
                    model, x, aux_pos, copy_weight=aux_weight,
                    attn_weight=aux_attn_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if test is not None and (step % eval_every == 0 or step == steps):
            ev = evaluate(model, test, device, seq_len)
            print(json.dumps({"tag": tag, "step": step,
                              "loss": round(float(loss.detach()), 4),
                              "test_acc": round(ev["accuracy"], 4)}),
                  flush=True)
    return float(loss.detach()), time.perf_counter() - t0


class TinyTransformer(nn.Module):
    def __init__(self, vocab, dim=64, layers=2, heads=4, max_len=32,
                 dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool),
                          diagonal=1)
        h = self.emb(x) + self.pos(torch.arange(n, device=x.device))
        h = self.encoder(h, mask=mask)
        return self.head(h)


def run_arm(name, args, train2, train3, test3, device):
    torch.manual_seed(args.seed)
    if name == "transformer":
        model = TinyTransformer(len(WORDS), args.dim, args.layers,
                                args.heads).to(device)
        loss, sec = train_lm(model, train3, args.steps, args.batch, args.lr,
                             device, args.amp, args.eval_every, test3, LEN,
                             tag=name)
        ev = evaluate(model, test3, device, LEN)
        return {"arm": name, "test_acc": ev["accuracy"], "eval": ev,
                "train_loss": loss, "sec": round(sec, 1),
                "params": sum(p.numel() for p in model.parameters())}
    mode = "spike" if name.startswith("curriculum_spike") else "rate"
    model = SpikingLM(SpikingConfig(vocab=len(WORDS), dim=args.dim,
                                    layers=args.layers, mode=mode,
                                    adapt=args.adapt)).to(device)
    warm = args.curriculum_steps
    loss2, sec2 = train_lm(model, train2, warm, args.batch, args.lr, device,
                           args.amp, args.eval_every, None)
    loss3, sec3 = train_lm(model, train3, args.steps, args.batch, args.lr,
                           device, args.amp, args.eval_every, test3, LEN,
                           tag=name)
    ev = evaluate(model, test3, device, LEN)
    return {"arm": name, "test_acc": ev["accuracy"], "eval": ev,
            "train_loss": loss3, "warm_loss": loss2,
            "sec": round(sec2 + sec3, 1),
            "params": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="transformer,curriculum_spike,curriculum_rate")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--curriculum-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/binding_baselines.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_triples, test_triples = make_split(args.seed)
    train2 = make_data(args.n_train, train_triples, args.seed + 11, 2)
    train3 = make_data(args.n_train, train_triples, args.seed + 1, 3)
    test3 = make_data(args.n_test, test_triples, args.seed + 2, 3)
    print(json.dumps({"len2": train2.size(1), "len3": train3.size(1),
                      "train_triples": len(train_triples),
                      "test_triples": len(test_triples)}), flush=True)
    rows = [run_arm(name.strip(), args, train2, train3, test3, device)
            for name in args.arms.split(",") if name.strip()]
    payload = {"task": "binding baselines", "config": vars(args),
               "chance": 0.25, "results": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

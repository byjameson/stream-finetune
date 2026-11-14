"""Two-phase spiking LM: TinyStories -> WikiText, with shadow rollback.

Phase A trains a small LIF language model on TinyStories. Phase B continues on
WikiText-2. At the end of phase B, three families of states are compared:

  * phase A only (the old-task reference)
  * phase B with no rollback (the new-task reference)
  * global rollback to phase A
  * importance-based per-synapse rollback: restore the top-k phase-A
    parameters by Fisher importance and keep the rest of phase B

The last one is the engineering version of the "orphan synapse" idea: a
per-synapse shadow copy that can be restored if plasticity damages an old
behaviour. It is a post-hoc consolidation test, not online learning.

    python research/spiking_shift.py --steps-a 300 --steps-b 300

Requires data/ts_wt2_{train,valid}.pt from research/prep_spiking_shift.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from research.spiking_lm import SpikingLM, WindowSampler, eval_model

ROOT = Path(__file__).resolve().parents[1]


def load_tokens(path: str, limit: int = 0) -> torch.Tensor:
    try:
        t = torch.load(path, weights_only=True, mmap=True)
    except TypeError:
        t = torch.load(path, weights_only=True)
    t = t.to(torch.int64)
    if limit:
        t = t[:limit]
    return t


def fixed_batches(tokens: torch.Tensor, seq: int, n: int, batch: int,
                  seed: int, device: torch.device):
    gen = torch.Generator().manual_seed(seed)
    hi = len(tokens) - seq - 1
    out = []
    for _ in range(n):
        starts = torch.randint(0, hi, (batch,), generator=gen).tolist()
        xs = torch.stack([tokens[s:s + seq] for s in starts]).to(device)
        ys = torch.stack([tokens[s + 1:s + seq + 1] for s in starts]).to(device)
        out.append((xs, ys))
    return out


def eval_both(model, old_batches, new_batches, tokenizer):
    return {
        "old": eval_model(model, old_batches, tokenizer),
        "new": eval_model(model, new_batches, tokenizer),
    }


def train_phase(model, opt, sampler, steps, eval_every, old_batches,
                new_batches, tokenizer, label):
    history = []
    model.train()
    for step in range(1, steps + 1):
        x, y = sampler.next()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            ev = eval_both(model, old_batches, new_batches, tokenizer)
            row = {"phase": label, "step": step,
                   "train_loss": float(loss.detach()), **ev}
            history.append(row)
            print(json.dumps(row), flush=True)
            model.train()
    return history


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


@torch.enable_grad()
def fisher_importance(model, batches):
    imp = {name: torch.zeros_like(p, device="cpu")
           for name, p in model.named_parameters()}
    model.eval()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                               y.reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                imp[name] += p.grad.detach().float().pow(2).cpu()
    model.zero_grad(set_to_none=True)
    return imp


def flatten_state(state, params):
    return torch.cat([state[name].float().flatten() for name, _ in params])


def state_from_flat(flat, params):
    out = {}
    offset = 0
    for name, p in params:
        n = p.numel()
        out[name] = flat[offset:offset + n].view_as(p)
        offset += n
    return out


def selective_rollback(state_a, state_b, importance, params, fraction):
    flat_a = flatten_state(state_a, params)
    flat_b = flatten_state(state_b, params)
    flat_i = torch.cat([importance[name].float().flatten()
                        for name, _ in params])
    k = int(round(fraction * flat_i.numel()))
    flat = flat_b.clone()
    if k > 0:
        idx = torch.topk(flat_i, k).indices
        flat[idx] = flat_a[idx]
    return state_from_flat(flat, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=300)
    ap.add_argument("--steps-b", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--fisher-batches", type=int, default=8)
    ap.add_argument("--fractions", default="0.10,0.25,0.50")
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--b-tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a-train", default=str(ROOT / "data/ts_train.pt"))
    ap.add_argument("--a-valid", default=str(ROOT / "data/ts_valid.pt"))
    ap.add_argument("--b-train", default=str(ROOT / "data/ts_wt2_train.pt"))
    ap.add_argument("--b-valid", default=str(ROOT / "data/ts_wt2_valid.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/ts_bpe8k.json"))
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_shift.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    a_train = load_tokens(args.a_train, args.a_tokens)
    a_valid = load_tokens(args.a_valid)
    b_train = load_tokens(args.b_train, args.b_tokens)
    b_valid = load_tokens(args.b_valid)
    print(json.dumps({
        "device": str(device),
        "a_train": len(a_train), "a_valid": len(a_valid),
        "b_train": len(b_train), "b_valid": len(b_valid),
        "steps_a": args.steps_a, "steps_b": args.steps_b,
        "adapt": args.adapt,
    }), flush=True)

    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 100, device)
    new_batches = fixed_batches(b_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 200, device)

    model = SpikingLM(vocab=tokenizer.get_vocab_size(), dim=args.dim,
                      layers=args.layers, mode="spike",
                      adapt=args.adapt).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Phase A: TinyStories.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sampler_a = WindowSampler(a_train, args.seq, args.batch,
                              args.seed, device)
    history_a = train_phase(model, opt, sampler_a, args.steps_a,
                            args.eval_every, old_batches, new_batches,
                            tokenizer, "A")
    state_a = clone_state(model)

    # Fisher importance on phase A. The fixed batches are deterministic.
    fisher_batches = fixed_batches(a_train, args.seq, args.fisher_batches,
                                   args.eval_batch_size, args.seed + 300,
                                   device)
    importance = fisher_importance(model, fisher_batches)
    print(json.dumps({"fisher_batches": len(fisher_batches)}), flush=True)

    # Phase B: WikiText-2, starting from the phase-A weights.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sampler_b = WindowSampler(b_train, args.seq, args.batch,
                              args.seed + 1, device)
    base_b = eval_both(model, old_batches, new_batches, tokenizer)
    history_b = [{"phase": "B", "step": 0, "train_loss": None, **base_b}]
    print(json.dumps(history_b[0]), flush=True)
    history_b += train_phase(model, opt, sampler_b, args.steps_b,
                             args.eval_every, old_batches, new_batches,
                             tokenizer, "B")
    state_b = clone_state(model)

    params = list(model.named_parameters())
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    arms = [
        ("phase_a", None, state_a),
        ("phase_b_no_rollback", 0.0, state_b),
        ("global_rollback", 1.0, state_a),
    ]
    for frac in fractions:
        arms.append((f"importance_{int(round(frac * 100))}", frac,
                     selective_rollback(state_a, state_b, importance,
                                        params, frac)))

    results = []
    for name, frac, state in arms:
        load_state(model, state)
        ev = eval_both(model, old_batches, new_batches, tokenizer)
        row = {"name": name, "fraction": frac, **ev}
        results.append(row)
        print(json.dumps(row), flush=True)

    peak = (torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None)
    payload = {
        "question": ("Does a per-synapse shadow copy recover old-task "
                     "performance better than global rollback after a "
                     "TinyStories -> WikiText shift?"),
        "data": {
            "phase_a": args.a_train, "phase_a_tokens": len(a_train),
            "phase_b": args.b_train, "phase_b_tokens": len(b_train),
            "old_valid": args.a_valid, "new_valid": args.b_valid,
        },
        "config": {
            "steps_a": args.steps_a, "steps_b": args.steps_b,
            "batch": args.batch, "seq": args.seq, "dim": args.dim,
            "layers": args.layers, "lr": args.lr, "adapt": args.adapt,
            "seed": args.seed, "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "fisher_batches": args.fisher_batches,
        },
        "peak_gpu_mb": round(peak, 1) if peak is not None else None,
        "history_a": history_a,
        "history_b": history_b,
        "importance": {"method": "Fisher diagonal on phase A",
                       "batches": len(fisher_batches)},
        "arms": results,
        "note": ("Post-hoc rollback after phase B. The global arm restores all "
                 "phase-A weights; the importance arms restore the top-k "
                 "phase-A parameters by Fisher importance. Old = TinyStories, "
                 "new = WikiText-2; both use the TinyStories BPE-8k tokenizer."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

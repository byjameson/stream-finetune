"""Controlled topology test: top-down feedback from the last layer.

Both arms start from the same 21M checkpoint. The feedback model adds a zero-
initialized top-down connection (so it is identical at step 0). Both are
fine-tuned for the same number of steps on TinyStories and scored on the held-out
narrative benchmark.

    python research/spiking_feedback.py --steps 500
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from research.spiking_replay import RandomWindows
from research.spiking_semantic_eval import evaluate as semantic_eval
from research.spiking_shift import clone_state, fixed_batches, load_tokens
from streamft.spiking import SpikingConfig, SpikingLM, evaluate_bpb, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def train(model, sampler, steps, lr, device, amp):
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
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", default=str(ROOT / "checkpoints/spiking_21m"))
    ap.add_argument("--cases", default=str(ROOT / "research/semantic_cases.json"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--a-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research/results/spiking_feedback.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    base, tok, meta = load_checkpoint(args.init_model)
    base.to(device)
    cfg = asdict(base.config)
    fb = SpikingLM(SpikingConfig(**{**cfg, "feedback": True})).to(device)
    fb.load_state_dict(base.state_dict(), strict=False)
    a_valid = load_tokens(str(ROOT / "data/ts_valid.pt"))
    old_batches = fixed_batches(a_valid, args.seq, args.eval_batches,
                                args.eval_batch_size, args.seed + 100, device)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    train_tokens = load_tokens(str(ROOT / "data/ts_train.pt"), args.a_tokens)
    initial = clone_state(base)

    rows = []
    for name, model in (("baseline", base), ("feedback", fb)):
        # reset to the same initial state for each arm
        model.load_state_dict({k: v.to(device) for k, v in initial.items()},
                              strict=False)
        sampler = RandomWindows(train_tokens, args.seq, args.batch,
                                args.seed, device)
        losses, sec = train(model, sampler, args.steps, args.lr, device, args.amp)
        model.eval()
        ev = {
            "old": evaluate_bpb(model, tok, old_batches),
            "semantic": semantic_eval(model, tok, cases, device),
            "params": sum(p.numel() for p in model.parameters()),
            "train_loss": losses[-1],
            "sec": round(sec, 1),
        }
        rows.append({"arm": name, **ev})
        print(json.dumps({"event": "arm", "arm": name,
                          "old": ev["old"], "semantic": ev["semantic"],
                          "sec": ev["sec"]}), flush=True)
        base.train(False)

    payload = {"init_model": args.init_model, "meta": meta,
               "config": vars(args), "arms": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

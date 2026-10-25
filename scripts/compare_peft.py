"""Matched comparison: PEFT/transformers QLoRA vs streamft on the same data.

Both sides: Qwen2.5-3B, NF4 4-bit, LoRA r=8 alpha=16 on q/k/v/o, AdamW lr 1e-4,
batch 1, Dolly-15k, identical rows and prompt masking (streamft's _encode_chat),
8 held-out rows for eval. PEFT uses gradient checkpointing (the QLoRA default).

  python scripts/compare_peft.py --model checkpoints/qwen3b_bnb4bit --steps 30 --seq 512

Prints a JSON row and appends it to results/compare_engines.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamft.cli import _encode_chat  # noqa: E402
from streamft.metrics import append_jsonl, now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/qwen3b_bnb4bit")
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--train_size", type=int, default=128)
    ap.add_argument("--eval_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--tag", default="peft_3b")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(ROOT / args.model)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map={"": 0}, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.train()
    print(f"[load] {args.model} in {time.time()-t0:.1f}s "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M",
          flush=True)

    ds = load_dataset(args.dataset, split="train").shuffle(seed=1234)
    train_rows = [ds[i] for i in range(min(args.train_size, len(ds)))]
    eval_rows = [ds[len(train_rows) + i]
                 for i in range(min(args.eval_size, len(ds) - len(train_rows)))]

    @torch.no_grad()
    def evaluate() -> float:
        """Token-weighted masked CE, the same aggregation as streamft's _masked_eval.

        A plain mean over rows is a different number (rows differ 17..190 response
        tokens) and made the two engines look 0.7 apart before this was matched.
        """
        model.eval()
        total, n = 0.0, 0
        for row in eval_rows:
            enc = _encode_chat(tok, row, args.seq)
            if enc is None:
                continue
            ids, labels = enc
            v = int((labels != -100).sum())
            if v == 0:
                continue
            total += float(model(input_ids=ids.cuda(),
                                 labels=labels.cuda()).loss) * v
            n += v
        model.train()
        return total / max(n, 1)

    ce0 = evaluate()
    print(f"[eval] before loss={ce0:.4f}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    losses, times, peaks = [], [], []
    for i in range(args.steps):
        enc = _encode_chat(tok, train_rows[i % len(train_rows)], args.seq)
        if enc is None:
            continue
        ids, labels = enc
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        out = model(input_ids=ids.cuda(), labels=labels.cuda())
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        losses.append(round(float(out.loss), 6))
        times.append(round(dt, 3))
        peaks.append(round(torch.cuda.max_memory_reserved() / 1e9, 3))
        if (i + 1) % 10 == 0:
            print(f"[step] {i+1}/{args.steps} loss={losses[-1]:.4f} {dt:.2f}s "
                  f"peak={peaks[-1]:.2f}GB", flush=True)

    ce1 = evaluate()
    row = {
        "engine": "peft-transformers",
        "tag": args.tag,
        "model": Path(args.model).name,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "peak_gb_max": max(peaks) if peaks else None,
        "ce_before": round(ce0, 5),
        "ce_after": round(ce1, 5),
        "ce_delta": round(ce1 - ce0, 6),
        "losses": losses,
        "wall_s": round(time.time() - t0, 1),
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "compare_engines.jsonl"), row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

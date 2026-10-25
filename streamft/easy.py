"""Two on-ramps for people without a training GPU.

  streamft cpu    SFT a small model on the CPU (the model must fit in RAM)
  streamft synth  generate training pairs locally with a small model (no API, no data hunt)

Both are deliberately simple: plain transformers forward/backward, our LoRA module, the
same prompt masking as the GPU path. The streaming engine is not involved.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

from .cli import _encode_chat, _format_messages  # noqa: F401  (same masking rules)
from .lora import apply_lora, lora_parameters, lora_state_dict


# --------------------------------------------------------------------------- cpu train

def _load_tiny(model_dir: str):
    from transformers import AutoModelForCausalLM

    dtype = torch.float32  # CPU: bf16 is slow and fp16 is unsupported on many CPUs
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def _eval_masked(model, rows, tok, seq: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for row in rows:
        enc = _encode_chat(tok, row, seq)
        if enc is None:
            continue
        ids, labels = enc
        loss = model(input_ids=ids, labels=labels).loss
        v = int((labels != -100).sum())
        total += float(loss) * v
        n += v
    model.train()
    if n == 0:
        raise RuntimeError("no eval row has a valid prompt mask")
    return total / n


def cpu_main(argv=None) -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft cpu")
    ap.add_argument("--model", required=True, help="small model dir or HF id (fits in RAM)")
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--train_size", type=int, default=128)
    ap.add_argument("--eval_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="cpu_sft")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = _load_tiny(args.model)
    apply_lora(model, r_attn=args.lora_r, r_mlp=0)
    params = list(lora_parameters(model))
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"[cpu] {args.model} loaded in {time.time()-t0:.1f}s, "
          f"{sum(p.numel() for p in params)/1e6:.2f}M adapter params", flush=True)

    ds = load_dataset(args.dataset, split="train").shuffle(seed=1234)
    train_rows = [ds[i] for i in range(min(args.train_size, len(ds)))]
    eval_rows = [ds[len(train_rows) + i]
                 for i in range(min(args.eval_size, len(ds) - len(train_rows)))]

    valid = [r for r in train_rows if _encode_chat(tok, r, args.seq) is not None]
    dropped = len(train_rows) - len(valid)
    if not valid:
        raise SystemExit("[cpu] no training row has a valid prompt mask")
    if dropped:
        print(f"[cpu] {dropped}/{len(train_rows)} rows dropped (prompt not a prefix)",
              flush=True)

    ce0 = _eval_masked(model, eval_rows, tok, args.seq)
    print(f"[cpu] eval before {ce0:.4f}", flush=True)

    model.train()
    losses, times = [], []
    for i in range(args.steps):
        enc = _encode_chat(tok, valid[i % len(valid)], args.seq)
        ids, labels = enc
        t = time.time()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(float(loss.detach()), 6))
        times.append(round(time.time() - t, 3))
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[cpu] step {i+1}/{args.steps} loss={losses[-1]:.4f} "
                  f"{times[-1]:.2f}s", flush=True)

    ce1 = _eval_masked(model, eval_rows, tok, args.seq)
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state_dict(model), str(out / "adapter.pt"))
    row = {
        "engine": "cpu-dense", "tag": args.tag, "model": args.model, "steps": len(losses),
        "seq": args.seq, "lr": args.lr, "lora_r": args.lora_r,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "ce_before": round(ce0, 5), "ce_after": round(ce1, 5),
        "ce_delta": round(ce1 - ce0, 5), "losses": losses,
        "sft_skipped": dropped, "saved": args.save or None,
        "license": "Apache-2.0", "device": "cpu",
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return 0


# --------------------------------------------------------------------------- synth

PROMPT = """You write training data. Produce ONE example as JSON with keys
"instruction" and "response". Topic: {task}.
Rules: the instruction is a short request; the response is a direct answer of 1-3
sentences; no markdown, no preamble, no extra keys.
JSON:"""


PLACEHOLDER_MARKERS = ("<instruction>", "<response>", "<your", "your instruction",
                      "lorem ipsum", "example.com")


def parse_pair(text: str):
    """Extract one {"instruction","response"} object; reject template junk.

    Small models happily echo the schema back ("<instruction>"), and a single such row
    poisons a dataset. Everything is filtered here so the caller only has to dedupe.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    ins = str(obj.get("instruction", "")).strip()
    res = str(obj.get("response", "")).strip()
    low = (ins + " " + res).lower()
    if any(p in low for p in PLACEHOLDER_MARKERS):
        return None
    if "{" in ins or "{" in res or "}" in ins or "}" in res:
        return None
    if not (5 <= len(ins) <= 200 and 10 <= len(res) <= 1200):
        return None
    return {"instruction": ins, "response": res}


def synth_main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft synth")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="data/synth.jsonl")
    ap.add_argument("--n", type=int, default=50, help="examples to attempt")
    ap.add_argument("--task", default="practical everyday knowledge")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()

    seen, kept = set(), []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(args.n):
        prompt = PROMPT.format(task=args.task)
        ids = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=args.seq, do_sample=True,
                                 temperature=args.temperature, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pair = parse_pair(text)
        if pair is None:
            continue
        key = pair["instruction"].lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(pair)
        if (len(kept) % 10) == 0:
            print(f"[synth] {len(kept)} kept / {i+1} attempts", flush=True)

    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out), "attempts": args.n, "kept": len(kept),
        "keep_rate": round(len(kept) / max(args.n, 1), 3),
        "seconds": round(time.time() - t0, 1), "model": args.model, "task": args.task,
    }, indent=1))
    return 0

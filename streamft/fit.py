"""`streamft fit` — will this model fit on my card, and what do we actually know?

Two sources of truth, kept separate on purpose:

  * **memory** is arithmetic from the config: NF4 weight bytes + the shared tensors that
    stay in fp32 (embeddings / LM head) + a fixed reserve. It answers "whole model, or
    streaming?" and nothing more.
  * **speed and block count** are printed only when this exact size+seq was measured on
    the reference machine (RTX 3050 Laptop 6 GB) and is listed in MEASURED. For anything
    else the tool refuses to guess and tells you to start at one block and let the
    trainer's autotune measure the marginal.

Usage:
  streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512
  streamft fit --model checkpoints/qwen14b_nf4 --gpu 6 --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured end-to-end on an RTX 3050 Laptop, 6 GB, Windows 11, AC power, NF4 base,
# LoRA r=8 on q/k/v/o, batch 1, Dolly-15k. step_s = median warm step.
MEASURED = [
    # Matched on (num_hidden_layers, hidden_size), not on a rounded parameter count: a
    # different 7B with a different architecture must not inherit this measurement.
    {"arch": ("qwen2", 36, 2048, 11008, 16, 2, 151936), "label": "Qwen2.5-3B", "seq": 512, "blocks": 36,
     "peak_gb": 3.85, "step_s": 0.94, "mode": "all resident",
     "tokens": "SFT: one Dolly row per step, median ~134 tokens"},
    {"arch": ("qwen2", 28, 3584, 18944, 28, 4, 152064), "label": "Qwen2.5-7B", "seq": 512, "blocks": 28,
     "peak_gb": 5.51, "step_s": 1.55, "mode": "all resident",
     "tokens": "SFT: one Dolly row per step, median ~134 tokens"},
    {"arch": ("qwen2", 48, 5120, 13824, 40, 8, 152064), "label": "Qwen2.5-14B", "seq": 512, "blocks": 6,
     "peak_gb": 3.86, "step_s": 5.86, "mode": "streaming",
     "tokens": "SFT: one Dolly row per step, median ~134 tokens"},
    {"arch": ("qwen2", 64, 5120, 27648, 40, 8, 152064), "label": "Qwen2.5-32B", "seq": 1024, "blocks": 2,
     "peak_gb": 3.90, "step_s": 12.48, "mode": "streaming",
     "tokens": "SFT: one Dolly row per step, median ~134 tokens"},
]

RESERVE_GB = 0.9  # CUDA context + allocator fragmentation + the bf16 recompute buffers


def load_config(model: str) -> dict:
    p = Path(model)
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text(encoding="utf-8"))


def model_fingerprint(cfg: dict) -> tuple:
    """Everything that changes the measurement: two 7B models are not interchangeable."""
    return (
        str(cfg.get("model_type", "")),
        int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0),
        int(cfg.get("hidden_size") or cfg.get("n_embd") or 0),
        int(cfg.get("intermediate_size") or 0),
        int(cfg.get("num_attention_heads") or 0),
        int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 0),
        int(cfg.get("vocab_size") or 0),
    )


def model_math(cfg: dict) -> dict:
    h = cfg.get("hidden_size") or cfg.get("n_embd")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    inter = cfg.get("intermediate_size") or 4 * h
    vocab = cfg.get("vocab_size", 0)
    heads = cfg.get("num_attention_heads", 1)
    kv = cfg.get("num_key_value_heads", heads)
    if not (h and layers):
        raise SystemExit("config.json has no hidden_size / num_hidden_layers")

    attn = h * h + 2 * h * (h * kv // heads) + h * h
    mlp = 3 * h * inter
    params = float(layers * (attn + mlp) + vocab * h)

    nf4_gb = params * 0.5 / 1e9 * 1.07          # 4-bit + bitsandbytes overhead
    tied = bool(cfg.get("tie_word_embeddings", False))
    # What actually sits on the GPU: the embedding is pinned to CPU by the engine
    # (only the gathered activation moves over), the head stays in bf16, and the
    # presets use --shared-dtype as-stored. A tied head has no separate tensor.
    head_gb = 0.0 if tied else vocab * h * 2 / 1e9
    return {"params": params, "layers": int(layers), "nf4_gb": nf4_gb,
            "head_gb": head_gb, "tied": tied, "hidden": int(h)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="streamft fit")
    ap.add_argument("--model", required=True, help="local dir or HF id")
    ap.add_argument("--gpu", type=float, default=6.0, help="VRAM in GB (default 6)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.model)
    m = model_math(cfg)
    need = m["nf4_gb"] + m["head_gb"] + RESERVE_GB
    fits_whole = need <= args.gpu
    size_b = round(m["params"] / 1e9)

    arch = model_fingerprint(cfg)
    measured = None
    for row in MEASURED:
        if row["arch"] == arch and row["seq"] == args.seq:
            measured = dict(row)
            break

    out = {
        "model": args.model,
        "params_b": round(m["params"] / 1e9, 2),
        "layers": m["layers"],
        "nf4_weights_gb": round(m["nf4_gb"], 1),
        "embedding_on_cpu": True,
        "lm_head_bf16_gb": round(m["head_gb"], 1),
        "needed_whole_gb": round(need, 1),
        "gpu_gb": args.gpu,
        "seq": args.seq,
        "fits_whole": fits_whole,
        "needs_streaming": not fits_whole,
        "measured_on_rtx3050_6gb": measured,
        "measured_elsewhere": measured is None,
    }
    if args.json:
        print(json.dumps(out, indent=1))
        return 0

    print(args.model)
    print(f"  {out['params_b']}B params, {m['layers']} layers")
    print(f"  NF4 blocks ~{out['nf4_weights_gb']} GB + bf16 head ~{out['lm_head_bf16_gb']} GB "
          f"+ reserve {RESERVE_GB} GB (embedding stays on CPU)")
    print(f"  -> whole model needs ~{out['needed_whole_gb']} GB on {args.gpu:g} GB")
    if fits_whole:
        print("  verdict: budget says the whole model fits at this seq")
        print("  caution: activations and allocator headroom are not fully modelled;")
        print("           treat this as a budget check, not a guarantee. >1k can still spill.")
    else:
        print("  verdict: needs streaming (one or more blocks on the GPU at a time)")
    if measured:
        print(f"  measured reference: {measured['label']} "
              f"(model_type {arch[0]}, layers {arch[1]}, hidden {arch[2]}) "
              f"on RTX 3050 6 GB -> {measured['blocks']} block(s), "
              f"{measured['peak_gb']} GB peak, {measured['step_s']} s/step at seq {measured['seq']}")
        print(f"  note: {measured['tokens']} (seq is a cap on this path)")
    else:
        print("  speed/blocks: not measured for this size at this seq — start with")
        print("                 --blocks-on-gpu 1 and let autotune grow it (it measures")
        print("                 the real MB/block on your card instead of guessing)")
    print("  next: streamft prepare --model <dir> --save-quant checkpoints/<name>")
    print("        streamft train --model-dir checkpoints/<name> --sft --steps 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

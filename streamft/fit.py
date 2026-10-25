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
MEASURED = {
    (3, 512): {"blocks": 36, "peak_gb": 3.85, "step_s": 0.938, "mode": "all resident"},
    (7, 512): {"blocks": 28, "peak_gb": 5.51, "step_s": 1.696, "mode": "all resident"},
    (14, 512): {"blocks": 6, "peak_gb": 3.86, "step_s": 6.726, "mode": "streaming"},
    (32, 1024): {"blocks": 2, "peak_gb": 3.90, "step_s": 12.541, "mode": "streaming"},
}

RESERVE_GB = 0.9  # CUDA context + allocator fragmentation + the bf16 recompute buffers


def load_config(model: str) -> dict:
    p = Path(model)
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text(encoding="utf-8"))


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
            "head_gb": head_gb, "tied": tied}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="streamft fit")
    ap.add_argument("--model", required=True, help="local dir or HF id")
    ap.add_argument("--gpu", type=float, default=6.0, help="VRAM in GB (default 6)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    m = model_math(load_config(args.model))
    need = m["nf4_gb"] + m["head_gb"] + RESERVE_GB
    fits_whole = need <= args.gpu
    size_b = round(m["params"] / 1e9)

    measured = None
    for (b, seq), row in MEASURED.items():
        if abs(size_b - b) <= max(1, round(b * 0.1)) and seq == args.seq:
            measured = dict(row, ref_size_b=b, ref_seq=seq)
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
        print("  verdict: fits whole at this seq (no streaming needed; >1k can still spill)")
    else:
        print("  verdict: needs streaming (one or more blocks on the GPU at a time)")
    if measured:
        print(f"  measured on RTX 3050 6 GB: {measured['blocks']} block(s), "
              f"{measured['peak_gb']} GB peak, {measured['step_s']} s/step at seq {measured['ref_seq']}")
    else:
        print("  speed/blocks: not measured for this size at this seq — start with")
        print("                 --blocks-on-gpu 1 and let autotune grow it (it measures")
        print("                 the real MB/block on your card instead of guessing)")
    print("  next: streamft prepare --model <dir> --save-quant checkpoints/<name>")
    print("        streamft train --model-dir checkpoints/<name> --sft --steps 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

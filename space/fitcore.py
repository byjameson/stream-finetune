"""Pure calculator behind the 'Will it fit?' Space.

No gradio import here so the arithmetic can be tested on its own:

    python space/fitcore.py --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512

Memory is arithmetic from config.json. Speed and block count are printed only for
configurations measured on the reference machine (RTX 3050 Laptop 6 GB); everything
else says "not measured" instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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

RESERVE_GB = 0.9


def read_config(model: str) -> dict:
    p = Path(model)
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text(encoding="utf-8"))


def fingerprint(cfg: dict) -> tuple:
    return (
        str(cfg.get("model_type", "")),
        int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0),
        int(cfg.get("hidden_size") or cfg.get("n_embd") or 0),
        int(cfg.get("intermediate_size") or 0),
        int(cfg.get("num_attention_heads") or 0),
        int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 0),
        int(cfg.get("vocab_size") or 0),
    )


def math_for(cfg: dict) -> dict:
    h = cfg.get("hidden_size") or cfg.get("n_embd")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    inter = cfg.get("intermediate_size") or 4 * h
    vocab = cfg.get("vocab_size", 0)
    heads = cfg.get("num_attention_heads", 1)
    kv = cfg.get("num_key_value_heads", heads)
    if not (h and layers):
        raise ValueError("config.json has no hidden_size / num_hidden_layers")

    attn = h * h + 2 * h * (h * kv // heads) + h * h
    mlp = 3 * h * inter
    params = float(layers * (attn + mlp) + vocab * h)

    nf4_gb = params * 0.5 / 1e9 * 1.07
    tied = bool(cfg.get("tie_word_embeddings", False))
    head_gb = 0.0 if tied else vocab * h * 2 / 1e9
    return {"params": params, "layers": int(layers), "nf4_gb": nf4_gb,
            "head_gb": head_gb, "tied": tied, "hidden": int(h), "vocab": int(vocab)}


def evaluate(model: str, gpu: float, seq: int) -> dict:
    m = math_for(read_config(model))
    need = m["nf4_gb"] + m["head_gb"] + RESERVE_GB
    fits = need <= gpu
    size_b = round(m["params"] / 1e9)

    arch = fingerprint(cfg)
    measured = None
    for row in MEASURED:
        if row["arch"] == arch and row["seq"] == seq:
            measured = dict(row)
            break

    return {
        "model": model,
        "params_b": round(m["params"] / 1e9, 2),
        "layers": m["layers"],
        "nf4_gb": round(m["nf4_gb"], 1),
        "head_gb": round(m["head_gb"], 1),
        "needed_gb": round(need, 1),
        "gpu_gb": gpu,
        "seq": seq,
        "fits_whole": fits,
        "measured": measured,
        "hidden": int(m["hidden"]),
        "fingerprint": list(arch),
    }


def render(r: dict) -> str:
    verdict = ("fits whole — no streaming needed at this sequence length"
               if r["fits_whole"] else
               "needs streaming — one or more blocks on the GPU at a time")
    lines = [
        f"### {r['model']}",
        "",
        f"- {r['params_b']}B params, {r['layers']} layers",
        f"- NF4 blocks ~{r['nf4_gb']} GB + bf16 head ~{r['head_gb']} GB + reserve {RESERVE_GB} GB",
        f"- whole model needs **~{r['needed_gb']} GB** on **{r['gpu_gb']:g} GB** at seq {r['seq']}",
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    if r["measured"]:
        m = r["measured"]
        lines += [
            f"Measured reference: **{m['label']}** on an RTX 3050 Laptop 6 GB "
            f"(layers {r['layers']}, hidden {r['hidden']}) — {m['blocks']} block(s), "
            f"{m['peak_gb']} GB peak, {m['step_s']} s/step at seq {m['seq']}. "
            f"{m['tokens']}.",
        ]
    else:
        lines += [
            "Speed and block count are *not* predicted for this size/seq — that would be a "
            "guess. Start with `--blocks-on-gpu 1` and let the trainer's autotune measure "
            "the real MB/block on your card.",
        ]
    lines += [
        "",
        "Memory is arithmetic from `config.json` (NF4 weight bytes, bf16 head, the "
        "embedding pinned to CPU, a 0.9 GB reserve). Activations grow with sequence "
        "length, so a borderline model can still spill above 1k tokens.",
        "",
        "```bash",
        f"pip install streamft",
        f"streamft fit --model {r['model']} --gpu {r['gpu_gb']:g} --seq {r['seq']}",
        "```",
        "",
        "Full measurements, the PEFT comparison and the gradient check: "
        "[byjameson/stream-finetune](https://github.com/byjameson/stream-finetune).",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=float, default=6.0)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()
    print(render(evaluate(a.model, a.gpu, a.seq)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

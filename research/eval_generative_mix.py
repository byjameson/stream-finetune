"""Evaluate generative spiking checkpoints on TinyStories and WikiText.

The mixed-corpus experiment asks whether data diversity improves WikiText bpb
at a fixed training budget, without destroying TinyStories. All rows use the
same tokenizer (TinyStories BPE-8k), sequence length and evaluation windows.

    python research/eval_generative_mix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb, load_checkpoint
from streamft.spiking_cli import fixed_windows, load_tokens

ROOT = Path(__file__).resolve().parents[1]
SEQ = 128
BATCHES = 8
BATCH_SIZE = 8
MAX_TOKENS = 200_000


def evaluate_model(path: str, device) -> dict:
    model, tok, _ = load_checkpoint(path)
    model.to(device)
    model.eval()
    rows = {}
    for name, data in (("tinystories", "data/ts_valid.pt"),
                       ("wikitext", "data/ts_wt2_valid.pt")):
        tokens = load_tokens(str(ROOT / data), MAX_TOKENS)
        batches = fixed_windows(tokens, SEQ, BATCHES, BATCH_SIZE, 0, device)
        rows[name] = evaluate_bpb(model, tok, batches)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        ("spiking_4m_10k", "checkpoints/spiking_4m_10k"),
        ("spiking_21m", "checkpoints/spiking_21m"),
        ("spiking_50m", "checkpoints/spiking_50m"),
        ("ts_only_3k", "checkpoints/spiking_ts4m_3k"),
        ("mix25_3k", "checkpoints/spiking_mix25_4m"),
        ("mix_3k", "checkpoints/spiking_mix4m"),
        ("mix75_3k", "checkpoints/spiking_mix75_4m"),
        ("mix25_21m_3k", "checkpoints/spiking_mix25_21m"),
    ]
    runs = []
    for name, path in models:
        rows = evaluate_model(str(ROOT / path), device)
        run = {"name": name, "model": path,
               "tinystories_bpb": rows["tinystories"]["bpb"],
               "wikitext_bpb": rows["wikitext"]["bpb"],
               "tinystories": rows["tinystories"],
               "wikitext": rows["wikitext"]}
        runs.append(run)
        print(json.dumps({k: run[k] for k in
                          ("name", "tinystories_bpb", "wikitext_bpb")}),
              flush=True)
    payload = {
        "question": "Does data diversity improve WikiText bpb at a fixed budget?",
        "config": {"seq": SEQ, "batches": BATCHES, "batch_size": BATCH_SIZE,
                   "max_tokens": MAX_TOKENS, "seed": 0},
        "runs": runs,
        "note": ("ts_only_3k and the mix runs use the same 4.3M architecture, "
                 "3000 steps, 16M training tokens and seed. mix25/mix/mix75 "
                 "are 12M+4M, 8M+8M and 4M+12M TinyStories+WikiText. The "
                 "21M mix run uses 3000 steps and is undertrained at that "
                 "budget; the older 4.3M/21M/50M models are TinyStories-only "
                 "baselines."),
    }
    out = ROOT / "research/results/spiking_generative_mix.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

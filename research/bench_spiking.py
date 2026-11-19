"""Measure generation speed and peak VRAM for the tiny spiking models."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from streamft.spiking import generate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("mix4m", "checkpoints/spiking_mix25_4m_10k"),
    ("ts21m", "checkpoints/spiking_21m"),
    ("story_safe", "checkpoints/spiking_space_safe2/versions/v0001"),
]
PROMPT = "Once upon a time"
TOKENS = 80


def bench(name, path, device, tokens):
    model, tok, _ = load_checkpoint(str(ROOT / path))
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    generate(model, tok, "hello", max_new_tokens=4, device=device,
             temperature=0.8, top_k=40)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = generate(model, tok, PROMPT, max_new_tokens=tokens, device=device,
                    temperature=0.8, top_k=40)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    n = len(tok.encode(text).ids)
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else None)
    rec = {"model": name, "device": str(device), "tokens": n,
           "sec": round(sec, 3), "tokens_per_s": round(n / max(sec, 1e-9), 2),
           "peak_vram_mb": None if peak is None else round(peak, 1)}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main():
    rows = []
    if torch.cuda.is_available():
        for name, path in MODELS:
            rows.append(bench(name, path, torch.device("cuda"), TOKENS))
    for name, path in MODELS[:2]:
        rows.append(bench(name, path, torch.device("cpu"), 20))
    payload = {"prompt": PROMPT, "rows": rows}
    out = ROOT / "research/results/spiking_efficiency.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

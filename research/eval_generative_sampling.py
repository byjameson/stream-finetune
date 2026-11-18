"""Sampling sweep for the generative spiking LMs.

bpb-optimal models can still produce repetitive text; tighter sampling can
change the picture. This sweep compares the TinyStories-only 10k model, the
25% mix 10k model and the 21M TinyStories model under three sampling settings.

    python research/eval_generative_sampling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from research.eval_generative_samples import metrics
from streamft.spiking import generate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Once upon a time", "The city of", "In the beginning"]
MODELS = [
    ("ts_4m_10k", "checkpoints/spiking_4m_10k"),
    ("mix25_4m_10k", "checkpoints/spiking_mix25_4m_10k"),
    ("ts_21m", "checkpoints/spiking_21m"),
]
SETTINGS = [
    ("loose", 0.8, 40, 1.0),
    ("tight", 0.5, 10, 1.1),
    ("tightest", 0.3, 10, 1.1),
]
MAX_NEW = 80


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, path in MODELS:
        model, tok, _ = load_checkpoint(str(ROOT / path))
        model.to(device)
        for prompt in PROMPTS:
            for label, temp, top_k, rep in SETTINGS:
                torch.manual_seed(0)
                text = generate(model, tok, prompt, max_new_tokens=MAX_NEW,
                                device=device, temperature=temp, top_k=top_k,
                                repetition_penalty=rep)
                rec = {"model": name, "prompt": prompt, "setting": label,
                       "temperature": temp, "top_k": top_k,
                       "repetition_penalty": rep, "text": text}
                rec.update(metrics(text))
                rows.append(rec)
                print(json.dumps({k: rec[k] for k in
                                  ("model", "prompt", "setting", "words",
                                   "adjacent_repeat_rate", "distinct_2",
                                   "max_repeat_3gram")}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "question": "Does tighter sampling rescue generation quality?",
        "config": {"max_new": MAX_NEW, "seed": 0, "settings": SETTINGS},
        "rows": rows,
    }
    out = ROOT / "research/results/spiking_generative_sampling.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

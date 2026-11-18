"""Sample the generative spiking LMs and measure repetition.

bpb says how well the model predicts held-out tokens; it does not say whether
generations are coherent. This script samples the TinyStories-only and mixed
models with the same prompts and reports distinct n-gram ratios and the largest
repeated 4-gram count.

    python research/eval_generative_samples.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from streamft.spiking import generate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ["Once upon a time", "The city of", "In the beginning"]
MODELS = [
    ("spiking_4m_10k", "checkpoints/spiking_4m_10k"),
    ("ts_only_3k", "checkpoints/spiking_ts4m_3k"),
    ("mix25_3k", "checkpoints/spiking_mix25_4m"),
    ("mix25_10k", "checkpoints/spiking_mix25_4m_10k"),
    ("mix25_21m_10k", "checkpoints/spiking_mix25_21m_10k"),
]
MAX_NEW = 80


def ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def metrics(text):
    words = text.lower().split()
    row = {"chars": len(text), "words": len(words)}
    for n in (1, 2, 3, 4):
        grams = ngrams(words, n)
        row[f"distinct_{n}"] = (len(set(grams)) / len(grams)) if grams else 0.0
        counts = Counter(grams)
        row[f"max_repeat_{n}gram"] = max(counts.values()) if counts else 0
    adjacent = sum(words[i] == words[i - 1]
                   for i in range(1, len(words)))
    row["adjacent_repeat_rate"] = adjacent / max(len(words) - 1, 1)
    counts = Counter(words)
    row["top_word_frac"] = (max(counts.values()) / len(words)
                            if words else 0.0)
    return row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, path in MODELS:
        model, tok, _ = load_checkpoint(str(ROOT / path))
        model.to(device)
        for prompt in PROMPTS:
            torch.manual_seed(0)
            text = generate(model, tok, prompt, max_new_tokens=MAX_NEW,
                            device=device, temperature=0.8, top_k=40)
            rec = {"model": name, "prompt": prompt, "text": text}
            rec.update(metrics(text))
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in
                              ("model", "prompt", "words", "distinct_4",
                               "max_repeat_4gram")}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "question": "Do lower WikiText bpb and mixed data improve generation?",
        "config": {"max_new": MAX_NEW, "temperature": 0.8, "top_k": 40,
                   "seed": 0},
        "rows": rows,
    }
    out = ROOT / "research/results/spiking_generative_samples.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

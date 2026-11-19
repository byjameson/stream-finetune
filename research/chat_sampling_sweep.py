"""Check whether the 21M chat model's word salad is a decoding problem.

    python research/chat_sampling_sweep.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from streamft.spiking import load_checkpoint

try:
    from research.chat_eval import chat_generate
except ImportError:
    from chat_eval import chat_generate

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [(0.3, 5, 1.2), (0.5, 10, 1.1), (0.7, 20, 1.1)]
PROMPTS = ["hello", "how are you?", "tell me a short story"]


def metrics(text: str, stop: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    trigrams = list(zip(words, words[1:], words[2:]))
    repeat = (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
    return {
        "text": text,
        "chars": len(text),
        "alpha": round(sum(c.isalpha() for c in text) / max(len(text), 1), 4),
        "repeat3": round(repeat, 4),
        "stop": stop,
        "tag_leak": int(stop == "tag" or "<|" in text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/spiking_chat_en_21m")
    ap.add_argument("--out",
                    default="research/results/spiking_chat_en_sampling.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, meta = load_checkpoint(args.model)
    model.to(device)
    rows = []
    for temp, top_k, penalty in SETTINGS:
        for prompt in PROMPTS:
            torch.manual_seed(7)
            text, stop = chat_generate(
                model, tok, prompt, max_new_tokens=48,
                temperature=temp, top_k=top_k,
                repetition_penalty=penalty, device=device)
            rows.append({"temperature": temp, "top_k": top_k,
                         "repetition_penalty": penalty,
                         "prompt": prompt, **metrics(text, stop)})
    payload = {"model": args.model, "step": meta.get("step"),
               "settings": SETTINGS, "rows": rows,
               "verdict": "DECODING_DOES_NOT_FIX_IT"}
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

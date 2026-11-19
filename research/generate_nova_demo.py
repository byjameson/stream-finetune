"""Before/after samples for the second (Nova) story-ingest corpus."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from streamft.spiking import generate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("base_21m", "checkpoints/spiking_21m"),
    ("nova_safe", "checkpoints/spiking_nova_safe/versions/v0001"),
]
PROMPTS = ["Nova the explorer", "Nova found a", "Nova learned that"]
MAX_NEW = 80


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, path in MODELS:
        model, tok, _ = load_checkpoint(str(ROOT / path))
        model.to(device)
        for prompt in PROMPTS:
            torch.manual_seed(0)
            text = generate(model, tok, prompt, max_new_tokens=MAX_NEW,
                            device=device, temperature=0.5, top_k=10,
                            repetition_penalty=1.1)
            alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
            rec = {"model": name, "prompt": prompt, "text": text,
                   "alpha_ratio": alpha,
                   "equals_ratio": text.count("=") / max(len(text), 1),
                   "replacement_count": text.count("\ufffd"),
                   "nova_count": text.lower().count("nova")}
            rows.append(rec)
            print(json.dumps({"model": name, "prompt": prompt,
                              "alpha_ratio": round(alpha, 3),
                              "equals_ratio": round(rec["equals_ratio"], 3),
                              "nova_count": rec["nova_count"],
                              "preview": text[:80].encode("ascii", "replace").decode()}),
                  flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = ROOT / "research/results/spiking_nova_generations.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

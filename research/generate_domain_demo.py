"""Generate before/after samples for the domain-ingest demo.

Writes UTF-8 JSON so Wikipedia text does not depend on the Windows console
codepage.

    python research/generate_domain_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from streamft.spiking import generate, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("base", "checkpoints/spiking_mix25_4m_10k"),
    ("domain_v0001", "checkpoints/spiking_domain_demo/versions/v0001"),
]
PROMPTS = ["Abuja is", "Afghanistan is", "The capital of Nigeria is"]
MAX_NEW = 60


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
            rows.append({"model": name, "prompt": prompt, "text": text,
                         "alpha_ratio": alpha,
                         "equals_ratio": text.count("=") / max(len(text), 1),
                         "replacement_count": text.count("\ufffd")})
            print(json.dumps({"model": name, "prompt": prompt,
                              "chars": len(text), "alpha_ratio": round(alpha, 3),
                              "equals_ratio": round(text.count("=") / max(len(text), 1), 3),
                              "replacement_count": text.count("\ufffd"),
                              "preview": text[:60].encode("ascii", "replace").decode()}),
                  flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = ROOT / "research/results/spiking_domain_generations.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

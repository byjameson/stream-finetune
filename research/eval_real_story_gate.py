"""Evaluate the real public-domain story ingest on two seeds.

Protocol matches the synthetic story-lab gate:
  fixed windows: 8 batches x 8 windows x 128 tokens, seed 0
  generation: 3 prompts x 3 seeds, 80 tokens, temp 0.8, top-k 40

    python research/eval_real_story_gate.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb, generate, load_checkpoint
from streamft.spiking_cli import fixed_windows, load_tokens

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "base": "checkpoints/spiking_21m",
    "seed0": "checkpoints/spiking_realstory_safe_s0/versions/v0001",
    "seed1": "checkpoints/spiking_realstory_safe_s1/versions/v0001",
}
DATA = {
    "target": "data/real_stories_valid.pt",
    "ts": "data/ts_valid.pt",
    "wt": "data/ts_wt2_valid.pt",
}
PROMPTS = ["Once upon a time", "Alice was", "Dorothy opened the door"]
GEN_SEEDS = [0, 1, 2]
GEN_TOKENS = 80


def eval_model(path: str) -> dict[str, dict]:
    model, tok, _ = load_checkpoint(str(ROOT / path))
    model.to(DEVICE)
    out = {}
    for name, rel in DATA.items():
        tokens = load_tokens(str(ROOT / rel), 200000)
        batches = fixed_windows(tokens, 128, 8, 8, 0, DEVICE)
        out[name] = evaluate_bpb(model, tok, batches)
    target_words = {"Alice", "alice", "Dorothy", "dorothy", "Oz", "oz",
                    "rabbit", "Queen", "queen", "wizard", "Wizard"}
    samples = []
    for prompt in PROMPTS:
        for seed in GEN_SEEDS:
            torch.manual_seed(seed)
            text = generate(model, tok, prompt, GEN_TOKENS, DEVICE,
                            temperature=0.8, top_k=40)
            samples.append({
                "prompt": prompt,
                "seed": seed,
                "text": text,
                "alpha": round(sum(c.isalpha() for c in text)
                               / max(len(text), 1), 4),
                "equals": round(text.count("=") / max(len(text), 1), 4),
                "replacement": text.count("\ufffd"),
                "target_words": sum(
                    len(re.findall(r"\b" + re.escape(w) + r"\b", text))
                    for w in target_words),
            })
    out["generation"] = samples
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    results = {name: eval_model(path) for name, path in MODELS.items()}
    base = results["base"]
    runs = []
    for label in ("seed0", "seed1"):
        cur = results[label]
        run = {
            "seed": int(label[-1]),
            "target_before": base["target"]["bpb"],
            "target_after": cur["target"]["bpb"],
            "target_improvement":
                (base["target"]["bpb"] - cur["target"]["bpb"])
                / base["target"]["bpb"],
            "ts_before": base["ts"]["bpb"],
            "ts_after": cur["ts"]["bpb"],
            "ts_regression":
                (cur["ts"]["bpb"] - base["ts"]["bpb"]) / base["ts"]["bpb"],
            "wt_before": base["wt"]["bpb"],
            "wt_after": cur["wt"]["bpb"],
            "wt_regression":
                (cur["wt"]["bpb"] - base["wt"]["bpb"]) / base["wt"]["bpb"],
        }
        gen = cur["generation"]
        run["generation"] = {
            "alpha_ratio": [s["alpha"] for s in gen],
            "equals_ratio": [s["equals"] for s in gen],
            "replacement_count": [s["replacement"] for s in gen],
            "target_word_hits": [s["target_words"] for s in gen],
        }
        run["worst_samples"] = [
            {"prompt": s["prompt"], "seed": s["seed"], "alpha": s["alpha"],
             "text": s["text"]}
            for s in gen if s["alpha"] < 0.60
        ]
        runs.append(run)

    checks = {
        "target_improves_10pct": all(r["target_improvement"] >= 0.10
                                     for r in runs),
        "old_ts_within_5pct": all(r["ts_regression"] <= 0.05 for r in runs),
        "general_wt_within_5pct": all(r["wt_regression"] <= 0.05
                                      for r in runs),
        "generation_alpha": all(
            a >= 0.60 for r in runs for a in r["generation"]["alpha_ratio"]),
        "generation_no_formatting": all(
            e <= 0.05 for r in runs
            for e in r["generation"]["equals_ratio"]),
        "generation_no_replacement": all(
            c == 0 for r in runs
            for c in r["generation"]["replacement_count"]),
    }
    passed = all(checks.values())
    payload = {
        "demo": "real public-domain Alice/Oz safe ingest",
        "base_model": MODELS["base"],
        "domain": {
            "name": "Alice in Wonderland (train) / Wizard of Oz (valid)",
            "train_passages": 200,
            "valid_passages": 100,
            "license": "public domain (Project Gutenberg via GITenberg)",
        },
        "recipe": {
            "steps": 120,
            "batch": 16,
            "seq": 128,
            "lr": 5e-05,
            "replay_fraction": 0.7,
            "old_data": "20M TinyStories + 5M WikiText",
            "rollback_fraction": 0.2,
        },
        "baseline": {
            "target": base["target"]["bpb"],
            "ts": base["ts"]["bpb"],
            "wt": base["wt"]["bpb"],
            "generation_alpha_ratio": [
                s["alpha"] for s in base["generation"]],
        },
        "runs": runs,
        "gate": checks,
        "verdict": "PASS_REAL_STORY_LAB" if passed else "BLOCKED",
        "caveats": [
            "one real corpus (two different books); 200 train / 100 valid passages",
            "200-passage train set is small; improvement is adaption, not general skill",
            "two seeds, one model size (21M)",
            "outputs are still repetitive and locally ungrammatical",
        ],
    }
    out = ROOT / "research/results/spiking_realstory_seedcheck.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

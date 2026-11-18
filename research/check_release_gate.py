"""Check the Tiny Spiking Lab release gate.

    python research/check_release_gate.py

This evaluates the base and ingested checkpoints on the target domain, the
monitored old shard and a general shard, then adds the generation-quality gate
from research/results/spiking_domain_generations.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb, load_checkpoint
from streamft.spiking_cli import fixed_windows, load_tokens

ROOT = Path(__file__).resolve().parents[1]
BASE = "checkpoints/spiking_mix25_4m_10k"
DOMAIN = "checkpoints/spiking_domain_demo/versions/v0001"
SEQ, BATCHES, BATCH_SIZE = 128, 8, 8


def bpb(model, tok, data, device):
    tokens = load_tokens(str(ROOT / data), 200_000)
    batches = fixed_windows(tokens, SEQ, BATCHES, BATCH_SIZE, 0, device)
    return evaluate_bpb(model, tok, batches)["bpb"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, tok, _ = load_checkpoint(str(ROOT / BASE))
    base.to(device)
    before = {name: bpb(base, tok, data, device) for name, data in (
        ("target", "data/domain_valid.pt"),
        ("old", "data/ts_valid.pt"),
        ("general", "data/ts_wt2_valid.pt"))}
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    domain, tok, _ = load_checkpoint(str(ROOT / DOMAIN))
    domain.to(device)
    after = {name: bpb(domain, tok, data, device) for name, data in (
        ("target", "data/domain_valid.pt"),
        ("old", "data/ts_valid.pt"),
        ("general", "data/ts_wt2_valid.pt"))}

    samples = json.loads(
        (ROOT / "research/results/spiking_domain_generations.json")
        .read_text(encoding="utf-8"))["rows"]
    gen_ok = all(
        row["alpha_ratio"] >= 0.60
        and row["equals_ratio"] <= 0.05
        and row["replacement_count"] == 0
        for row in samples)
    target_ok = after["target"] <= before["target"] * 0.99
    old_ok = after["old"] <= before["old"] * 1.03
    general_ok = after["general"] <= before["general"] * 1.05
    passed = target_ok and old_ok and general_ok and gen_ok
    result = {
        "gate": "PASS" if passed else "BLOCKED",
        "bpb": {"before": before, "after": after},
        "checks": {
            "target_improves_1pct": target_ok,
            "old_within_3pct": old_ok,
            "general_within_5pct": general_ok,
            "generation_quality": gen_ok,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the narrow story-lab release gate on two corpora and two seeds.

    python research/check_story_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATION = ["spiking_space_safe2.json", "spiking_nova_safe.json"]


def main():
    seed = json.loads(
        (ROOT / "research/results/spiking_story_seedcheck.json")
        .read_text(encoding="utf-8"))
    runs = seed["runs"]
    bpb_checks = {
        "target_improves_10pct": all(
            r["target_improvement"] >= 0.10 for r in runs),
        "old_within_5pct": all(r["ts_regression"] <= 0.05 for r in runs),
        "general_within_5pct": all(r["wt_regression"] <= 0.05 for r in runs),
    }
    gen_checks = {"generation_alpha": True,
                  "generation_no_formatting": True,
                  "generation_no_replacement": True}
    for fname in GENERATION:
        data = json.loads(
            (ROOT / "research/results" / fname).read_text(encoding="utf-8"))
        gen = data["generation"]
        gen_checks["generation_alpha"] &= all(
            a >= 0.60 for a in gen["after_alpha_ratio"])
        gen_checks["generation_no_formatting"] &= all(
            e <= 0.05 for e in gen["after_equals_ratio"])
        gen_checks["generation_no_replacement"] &= all(
            r == 0 for r in gen["after_replacement_count"])
    checks = {**bpb_checks, **gen_checks}
    passed = all(checks.values())
    result = {
        "gate": "PASS_NARROW_STORY_LAB" if passed else "BLOCKED",
        "checks": checks,
        "runs": [
            {"corpus": r["corpus"], "seed": r["seed"],
             "target_improvement": round(r["target_improvement"], 4),
             "ts_regression": round(r["ts_regression"], 4),
             "wt_regression": round(r["wt_regression"], 4)}
            for r in runs
        ],
        "caveats": seed["caveats"],
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

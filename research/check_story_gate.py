"""Check the narrow story-lab release gate on both story corpora.

    python research/check_story_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    "space": "spiking_space_safe2.json",
    "nova": "spiking_nova_safe.json",
}


def check(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    full = data["full_eval"]
    gen = data["generation"]
    checks = {
        "target_improves_10pct": full["target_improvement"] >= 0.10,
        "old_within_5pct": full["old_regression"] <= 0.05,
        "general_within_5pct": full["general_regression"] <= 0.05,
        "generation_alpha": all(a >= 0.60 for a in gen["after_alpha_ratio"]),
        "generation_no_formatting": all(
            e <= 0.05 for e in gen["after_equals_ratio"]),
        "generation_no_replacement": all(
            r == 0 for r in gen["after_replacement_count"]),
    }
    return {"checks": checks, "passed": all(checks.values()),
            "data": data}


def main():
    results = {}
    for name, fname in CORPORA.items():
        results[name] = check(ROOT / "research/results" / fname)
    passed = all(r["passed"] for r in results.values())
    result = {
        "gate": "PASS_NARROW_STORY_LAB" if passed else "BLOCKED",
        "corpora": {name: {"passed": r["passed"], "checks": r["checks"]}
                    for name, r in results.items()},
        "caveats": [
            "Two synthetic 200/100 story corpora, one seed each.",
            "Target bpb 0.60-0.72 is close to memorization.",
            "Samples are still repetitive; this is a style lab, not a general model.",
            "Packaging, licence and model card are still missing.",
        ],
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

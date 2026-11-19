"""Check the narrow story-lab release gate.

    python research/check_story_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads(
        (ROOT / "research/results/spiking_space_safe2.json")
        .read_text(encoding="utf-8"))
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
    passed = all(checks.values())
    result = {
        "gate": "PASS_NARROW_STORY_LAB" if passed else "BLOCKED",
        "checks": checks,
        "caveats": data["release_gate"]["caveats"],
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

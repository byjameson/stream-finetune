"""Evaluate the hybrid ML + rule moderation filter.

    python research/redteam_moderation.py

Reports rule-only, ML-only and hybrid results on the authored red-team set and
on the held-out semantic paraphrases that the rules intentionally do not know.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from atomzero.ml_safety import build_safety_filter

try:
    from research.moderation_cases import SEMANTIC_HOLDOUT
    from research.redteam_safety import BENIGN, HARMFUL, OUTPUT_CASES
except ImportError:
    from moderation_cases import SEMANTIC_HOLDOUT
    from redteam_safety import BENIGN, HARMFUL, OUTPUT_CASES

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="research/results/redteam_moderation.json")
    args = ap.parse_args(argv)
    safety = build_safety_filter(prefer_ml=True)
    model = getattr(safety, "model", None)
    rules = safety.rules

    def ml(text):
        if model is None:
            return None
        try:
            return model.score(text)
        except Exception:
            return None

    rows = []
    for text, expected in HARMFUL:
        rule = rules.check(text)
        neural = ml(text)
        hybrid = safety.check(text)
        rows.append({
            "kind": "harmful", "text": text, "expected": expected,
            "rule": rule["category"] if rule else None,
            "ml": neural["category"] if neural else None,
            "hybrid": hybrid["category"] if hybrid else None,
            "rule_blocked": rule is not None,
            "ml_blocked": neural is not None,
            "hybrid_blocked": hybrid is not None,
            "hybrid_exact": bool(hybrid
                                 and hybrid["category"] == expected),
        })
    for text in BENIGN:
        rule = rules.check(text)
        neural = ml(text)
        hybrid = safety.check(text)
        rows.append({
            "kind": "benign", "text": text, "expected": None,
            "rule": rule["category"] if rule else None,
            "ml": neural["category"] if neural else None,
            "hybrid": hybrid["category"] if hybrid else None,
            "rule_blocked": rule is not None,
            "ml_blocked": neural is not None,
            "hybrid_blocked": hybrid is not None,
            "hybrid_exact": hybrid is None,
        })
    for text, expected in SEMANTIC_HOLDOUT:
        rule = rules.check(text)
        neural = ml(text)
        hybrid = safety.check(text)
        rows.append({
            "kind": "semantic", "text": text, "expected": expected,
            "rule": rule["category"] if rule else None,
            "ml": neural["category"] if neural else None,
            "hybrid": hybrid["category"] if hybrid else None,
            "rule_blocked": rule is not None,
            "ml_blocked": neural is not None,
            "hybrid_blocked": hybrid is not None,
            "hybrid_exact": bool(hybrid
                                 and hybrid["category"] == expected),
        })
    output_rows = []
    for text, expected in OUTPUT_CASES:
        filtered, verdict = safety.filter_reply(text)
        output_rows.append({
            "text": text, "expected": expected,
            "got": verdict["category"] if verdict else None,
            "passed": (verdict["category"] if verdict else None) == expected,
            "filtered": filtered,
        })

    def count(kind, key):
        subset = [row for row in rows if row["kind"] == kind]
        return sum(row[key] for row in subset), len(subset)

    harmful = count("harmful", "hybrid_blocked")
    benign = count("benign", "hybrid_blocked")
    semantic = count("semantic", "hybrid_blocked")
    semantic_exact = count("semantic", "hybrid_exact")
    ml_semantic = count("semantic", "ml_blocked")
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": safety.version,
        "ml_enabled": bool(getattr(safety, "ml_enabled", False)),
        "redteam_harmful_blocked": f"{harmful[0]}/{harmful[1]}",
        "redteam_benign_blocked": f"{benign[0]}/{benign[1]}",
        "redteam_harmful_exact": f"{count('harmful', 'hybrid_exact')[0]}/"
                                 f"{harmful[1]}",
        "semantic_blocked": f"{semantic[0]}/{semantic[1]}",
        "semantic_exact": f"{semantic_exact[0]}/{semantic[1]}",
        "semantic_ml_blocked": f"{ml_semantic[0]}/{ml_semantic[1]}",
        "output_passed": f"{sum(row['passed'] for row in output_rows)}/"
                          f"{len(output_rows)}",
        "rows": rows,
        "output_rows": output_rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"version={result['version']} ml={result['ml_enabled']}")
    print(f"redteam harmful blocked: {result['redteam_harmful_blocked']} "
          f"(exact category {result['redteam_harmful_exact']})")
    print(f"redteam benign blocked:  {result['redteam_benign_blocked']}")
    print(f"semantic blocked:        {result['semantic_blocked']} "
          f"(ML-only {result['semantic_ml_blocked']}, "
          f"exact {result['semantic_exact']})")
    print(f"output filter:           {result['output_passed']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

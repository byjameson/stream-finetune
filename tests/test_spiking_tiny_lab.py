import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_domain_ingest_improves_target_but_release_is_blocked():
    data = json.loads(
        (ROOT / "research/results/spiking_domain_ingest.json")
        .read_text(encoding="utf-8"))
    gate = data["ingest_gate"]
    full = data["full_eval_8192"]
    assert gate["accepted"] is True
    assert gate["new_improvement"] > 0.20
    assert gate["old_regression"] < 0.03
    # The full eval exceeds the ingest gate's old-regression budget.
    assert full["old_regression"] > 0.03
    assert full["general_regression"] > 0.05
    assert data["release_gate"]["verdict"] == "BLOCKED"
    assert min(data["generation"]["base_alpha_ratio"]) < 0.6

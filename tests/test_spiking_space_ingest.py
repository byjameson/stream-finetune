import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_space_style_adapts_but_full_retention_fails():
    data = json.loads(
        (ROOT / "research/results/spiking_space_ingest.json")
        .read_text(encoding="utf-8"))
    gate = data["ingest_gate"]
    full = data["full_eval"]
    assert gate["accepted"] is True
    assert gate["rolled_back"] is True
    assert gate["after"]["new_bpb_4096"] < 1.8
    assert (gate["before"]["new_bpb_4096"]
            - gate["after"]["new_bpb_4096"]) > 0.4
    assert full["target_improvement"] > 0.20
    assert full["old_regression"] > 0.10
    assert full["general_regression"] > 0.30
    assert min(data["generation"]["after_alpha_ratio"]) > 0.6
    assert data["release_gate"]["verdict"] == "BLOCKED"

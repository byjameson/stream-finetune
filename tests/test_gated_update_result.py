import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gated_update_rejected_the_small_queue_candidate():
    data = json.loads(
        (ROOT / "research/results/gated_update.json")
        .read_text(encoding="utf-8"))
    assert data["queue"]["queued"] == 55
    assert data["decision"]["accepted"] is False
    assert data["decision"]["checks"]["clean_bpb"]
    assert data["decision"]["checks"]["clean_assistant"]
    assert not data["decision"]["checks"]["target_gain"]
    assert data["decision"]["target_gain"] < 0.05
    assert "accepted_to" not in data

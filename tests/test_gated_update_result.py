import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gated_format_update_is_rejected_by_the_clean_smoke_gate():
    data = json.loads(
        (ROOT / "research/results/gated_update.json")
        .read_text(encoding="utf-8"))
    assert data["queue"]["queued"] == 283
    assert data["decision"]["accepted"] is False
    assert data["decision"]["checks"]["target_gain"]
    assert data["decision"]["checks"]["clean_bpb"]
    assert data["decision"]["checks"]["clean_assistant"]
    assert data["decision"]["checks"]["behavior"]
    assert not data["decision"]["checks"]["clean_smoke"]
    assert "accepted_to" not in data

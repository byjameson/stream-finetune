import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gated_update_accepted_the_novel_skill_queue():
    data = json.loads(
        (ROOT / "research/results/gated_update.json")
        .read_text(encoding="utf-8"))
    assert data["queue"]["queued"] == 267
    assert data["decision"]["accepted"] is True
    assert data["decision"]["checks"]["target_gain"]
    assert data["decision"]["checks"]["clean_bpb"]
    assert data["decision"]["checks"]["clean_assistant"]
    assert data["decision"]["target_gain"] >= 0.05
    assert data["accepted_to"] == "checkpoints\\updates\\v0001"

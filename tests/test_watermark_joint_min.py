import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_joint_min.json"


def test_joint_training_does_not_fix_the_0_50_merge():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in payload["results"]}
    assert len(rows) == 6
    assert rows["sst2"]["watermark_hits"] == 0
    assert rows["joint"]["watermark_hits"] == 8
    # The mark is gone at 0.25, 0.50 and 0.75; only the pure joint adapter keeps it.
    for f in ("joint+sst2_0.25", "joint+sst2_0.50", "joint+sst2_0.75"):
        assert rows[f]["watermark_hits"] == 0
    assert rows["joint+sst2_1.00"]["watermark_hits"] == 8

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_joint.json"


def test_joint_adapter_keeps_the_watermark_and_lowers_sst2_loss():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert payload["joint"] is True
    assert payload["base_trigger_hits"] == 0
    assert payload["base_loss"] > 10.0
    for variant in payload["variants"]:
        assert variant["seen_hits"] == 5
        assert variant["unseen_hits"] == 3
        assert variant["loss"] < payload["base_loss"]

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_threshold.json"


def test_watermark_threshold_pins_the_generation_cliff():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {(r["host"], r["wm_fraction"]): r for r in payload["results"]}
    assert len(rows) == 20
    for host in ("sst2", "agnews"):
        # Generation is a step function: 0, 1, 5, 6, 8, 8 on the fine grid.
        assert rows[(host, 0.50)]["hits"] == 0
        assert rows[(host, 0.60)]["hits"] == 0
        assert rows[(host, 0.65)]["hits"] == 1
        assert rows[(host, 0.70)]["hits"] == 5
        assert rows[(host, 0.80)]["hits"] == 6
        assert rows[(host, 0.85)]["hits"] == 8
        assert rows[(host, 1.00)]["hits"] == 8
        # The logprob rises monotonically even while generation is still 0/8.
        logps = [rows[(host, f)]["mean_logprob"] for f in payload["fractions"]]
        assert logps == sorted(logps)
        assert rows[(host, 0.60)]["mean_logprob"] > -2.1

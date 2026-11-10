import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_threshold_seed2.json"


def test_watermark_threshold_seed2_shifts_the_generation_cliff():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert "seed2" in payload["adapter"]
    rows = {(r["host"], r["wm_fraction"]): r for r in payload["results"]}
    assert len(rows) == 20
    for host in ("sst2", "agnews"):
        # Seed 2 is later than seed 1: still 0/8 at 0.65, 5/8 at 0.85, 8/8 only at 1.0.
        assert rows[(host, 0.65)]["hits"] == 0
        assert rows[(host, 0.85)]["hits"] == 5
        assert rows[(host, 0.90)]["hits"] == 7
        assert rows[(host, 1.00)]["hits"] == 8
        logps = [rows[(host, f)]["mean_logprob"] for f in payload["fractions"]]
        assert logps == sorted(logps)
        assert rows[(host, 0.65)]["mean_logprob"] < -1.5
    # The SST-2 host gets its first hits at 0.70; the AG host only at 0.75.
    assert rows[("sst2", 0.70)]["hits"] == 2
    assert rows[("agnews", 0.70)]["hits"] == 0

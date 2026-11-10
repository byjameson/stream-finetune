import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/epistasis_sweep.json"


def test_epistasis_sweep_pins_the_threshold():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert payload["fractions"] == [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    rows = {(r["host"], r["wm_fraction"]): r for r in payload["results"]}
    assert len(rows) == 18
    for host in ("sst2", "agnews"):
        # The 0.5 merge erases the watermark; 0.9 is the first fully reliable point.
        assert rows[(host, 0.5)]["watermark"] == 0.0
        assert 0.0 < rows[(host, 0.75)]["watermark"] < 1.0
        assert rows[(host, 0.9)]["watermark"] == 1.0
        assert rows[(host, 1.0)]["watermark"] == 1.0
    # The task adapter alone is strong; the watermark-only end is weaker on SST-2.
    assert rows[("sst2", 0.0)]["sst2"] > 0.9
    assert rows[("sst2", 1.0)]["sst2"] < 0.8
    assert rows[("agnews", 0.0)]["agnews"] < 0.5

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_threshold_tok2.json"


def test_token2_has_a_higher_logprob_and_a_later_cliff():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert payload["token"] == "AUTHENTIC-9K2M-4Q8X"
    rows = {(r["host"], r["wm_fraction"]): r for r in payload["results"]}
    assert len(rows) == 20
    for host in ("sst2", "agnews"):
        # The logit is already confident at 0.65/0.70, but generation is still 0/8.
        assert rows[(host, 0.65)]["hits"] == 0
        assert rows[(host, 0.65)]["mean_logprob"] > -1.1
        assert rows[(host, 0.70)]["hits"] == 0
        assert rows[(host, 0.70)]["mean_logprob"] > -0.6
        # The generation cliff is at 0.90; 7/8 at 0.85.
        assert rows[(host, 0.85)]["hits"] == 7
        assert rows[(host, 0.90)]["hits"] == 8
        assert rows[(host, 1.00)]["hits"] == 8
    # The first hits differ between hosts: SST-2 2/8, AG 1/8 at 0.75.
    assert rows[("sst2", 0.75)]["hits"] == 2
    assert rows[("agnews", 0.75)]["hits"] == 1

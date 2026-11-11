import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/watermark_rank.json"


def test_first_token_can_be_top1_while_the_full_token_fails():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {(r["run"], r["host"], r["wm_fraction"]): r for r in payload["results"]}
    assert len(rows) == 36

    # Token 1: the first token is rank 1 at 0.70/0.75, but the full string is 5/8.
    assert rows[("token1", "sst2", 0.50)]["first_rank_median"] > 10
    assert rows[("token1", "sst2", 0.50)]["first_top1"] == 0
    assert rows[("token1", "sst2", 0.70)]["first_rank_median"] == 1
    assert rows[("token1", "sst2", 0.70)]["first_top1"] == 7
    assert rows[("token1", "sst2", 0.70)]["hits"] == 5
    assert rows[("token1", "sst2", 0.75)]["first_top1"] == 8
    assert rows[("token1", "sst2", 0.75)]["hits"] == 5
    assert rows[("token1", "sst2", 0.85)]["hits"] == 8

    # Token 2: the first token is top-1 at 0.75/0.80, but the full string is 2/8 and 5/8.
    assert rows[("token2", "sst2", 0.70)]["first_rank_median"] > 1
    assert rows[("token2", "sst2", 0.70)]["hits"] == 0
    assert rows[("token2", "sst2", 0.75)]["first_rank_median"] == 1
    assert rows[("token2", "sst2", 0.75)]["first_top1"] == 7
    assert rows[("token2", "sst2", 0.75)]["hits"] == 2
    assert rows[("token2", "sst2", 0.80)]["first_top1"] == 8
    assert rows[("token2", "sst2", 0.80)]["hits"] == 5
    assert rows[("token2", "sst2", 0.90)]["hits"] == 8

    # Host independence: the AG host shows the same first-token/full-token gap.
    assert rows[("token1", "agnews", 0.70)]["first_top1"] == 7
    assert rows[("token1", "agnews", 0.70)]["hits"] == 5
    assert rows[("token2", "agnews", 0.75)]["first_top1"] == 6
    assert rows[("token2", "agnews", 0.75)]["hits"] == 1

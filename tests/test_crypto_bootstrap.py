import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_crypto_bootstrap_is_fragile_not_a_profit_claim():
    data = json.loads(
        (ROOT / "research/results/crypto_bootstrap.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "FRAGILE"
    assert data["bootstrap"]["samples"] == 2000
    assert data["bootstrap"]["block_days"] == 30
    for symbol in ("BTCUSDT", "ETHUSDT"):
        asset = data["assets"][symbol]
        assert asset["days"] > 2000
        prob = asset["bootstrap"]["p_strat_beats_bh"]
        assert 0.0 <= prob <= 1.0
        assert prob < 0.90
        assert len(asset["start_dates"]) == 6
    assert data["summary"]["start_date_total"] == 12

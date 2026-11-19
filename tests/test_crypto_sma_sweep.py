import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_sma_sweep_is_a_candidate_not_a_profit_claim():
    data = json.loads(
        (ROOT / "research/results/crypto_sma_sweep.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "TREND_CANDIDATE"
    assert data["summary"]["sma50_beats_bh_both"]
    for symbol in ("BTCUSDT", "ETHUSDT"):
        asset = data["assets"][symbol]
        assert len(asset["results"]) == 4
        labels = {row["label"] for row in asset["results"]}
        assert labels == {"sma20", "sma50", "sma100", "sma200"}
        # Window sensitivity: at least one window must fail to beat
        # buy-and-hold, so the file cannot be read as a universal edge.
        assert any(row["total_return"] <= row["buy_hold_return"]
                   for row in asset["results"])

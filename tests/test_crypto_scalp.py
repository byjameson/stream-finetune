import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_short_term_1pct_tp_sl_has_no_edge():
    data = json.loads(
        (ROOT / "research/results/crypto_scalp_btc.json")
        .read_text(encoding="utf-8"))
    assert data["symbol"] == "BTCUSDT"
    assert data["interval"] == "15m"
    assert data["rules"]["take_profit"] == 0.01
    assert data["rules"]["stop_loss"] == 0.01
    assert data["rules"]["risk_fraction"] == 0.75
    assert data["verdict"] == "NO_EDGE"
    for name, result in data["results"].items():
        test = result["test"]
        assert test["trades"] > 500
        assert test["win_rate"] < 0.50
        assert test["avg_net"] < 0
        assert test["total_return"] < 0
        # Net worst trade is the 1% stop plus round-trip costs, not a
        # 1% price move; the 0.75 position size caps the equity loss.
        assert test["worst_trade"] >= -0.013
    assert data["results"]["random"]["test"]["avg_net"] < 0

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_crypto_paper_result_schema_and_honest_verdict():
    data = json.loads(
        (ROOT / "research/results/crypto_paper_btc.json")
        .read_text(encoding="utf-8"))
    assert data["symbol"] == "BTCUSDT"
    assert data["verdict"] in {"CHECK", "NO_EDGE"}
    assert len(data["results"]) == 4
    names = {row["model"] for row in data["results"]}
    assert names == {"sma", "ridge", "hgb", "logistic"}
    for row in data["results"]:
        assert row["test_days"] > 2500
        assert "total_return" in row
        assert "sharpe" in row
        assert "max_drawdown" in row
    # The learned models are the point: at least one must fail to beat
    # buy-and-hold, so this file cannot be read as a profit claim.
    learned = [row for row in data["results"]
               if row["model"] in {"ridge", "hgb", "logistic"}]
    assert any(row["total_return"] <= row["buy_hold_return"]
               for row in learned)

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_short_term_sweep_has_no_edge_at_any_horizon_or_cost():
    data = json.loads(
        (ROOT / "research/results/crypto_scalp_sweep.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "NO_EDGE"
    assert len(data["rows"]) == 24
    assert {row["timeframe"] for row in data["rows"]} == {"15m", "1h", "4h"}
    assert {row["cost"] for row in data["rows"]} == {"taker", "maker"}
    for row in data["rows"]:
        assert row["trades"] > 50
        assert row["total_return"] < 0
        assert row["avg_net"] < 0
    best = max(data["rows"], key=lambda row: row["total_return"])
    assert best["total_return"] < 0
    assert best["total_return"] < best["buy_hold_return"]

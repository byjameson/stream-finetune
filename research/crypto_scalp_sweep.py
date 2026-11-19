"""Horizon and cost sensitivity for the short-term 1% TP / 1% SL test.

No real money. Checks whether 1h/4h horizons or maker-like costs rescue the
strategy; reports every combination, not a cherry-picked one.

    python research/crypto_scalp_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from research.crypto_scalp import (add_features, fetch, signals,
                                       simulate)
except ImportError:
    from crypto_scalp import add_features, fetch, signals, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results/crypto_scalp_sweep.json"
TEST_START = "2024-01-01"
HOLDS = {"15m": 16, "1h": 8, "4h": 4}
COSTS = {
    "taker": {"fee": 0.001, "slip": 0.0005},
    "maker": {"fee": 0.0002, "slip": 0.0002},
}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "15m":
        return df
    out = (df.set_index("date")
           .resample(rule)
           .agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
           .dropna()
           .reset_index())
    return out


def main():
    base = fetch()
    rows = []
    for timeframe, rule in (("15m", "15m"), ("1h", "1h"), ("4h", "4h")):
        df = add_features(resample(base, rule))
        sig = signals(df)
        test_idx = int(df.index[df["date"] >= pd.Timestamp(TEST_START,
                                                           tz="UTC")][0])
        for signal_name, signal in sig.items():
            for cost_name, cost in COSTS.items():
                result = simulate(
                    df.iloc[test_idx:].reset_index(drop=True),
                    signal[test_idx:],
                    fee=cost["fee"], slip=cost["slip"],
                    max_hold=HOLDS[timeframe])
                rows.append({
                    "timeframe": timeframe,
                    "signal": signal_name,
                    "cost": cost_name,
                    **{key: result.get(key) for key in (
                        "trades", "win_rate", "avg_net", "total_return",
                        "buy_hold_return", "profit_factor", "worst_trade")},
                })
    candidates = [row for row in rows
                  if (row.get("trades") or 0) >= 50
                  and (row.get("total_return") or -1) > 0
                  and (row.get("total_return") or -1)
                  > (row.get("buy_hold_return") or 1)]
    payload = {
        "symbol": "BTCUSDT",
        "test_start": TEST_START,
        "timeframes": list(HOLDS),
        "costs": COSTS,
        "rows": rows,
        "verdict": "CHECK" if candidates else "NO_EDGE",
        "caveats": [
            "maker fill is assumed certain; real limit fills are not",
            "15m bars are resampled; intrabar path is unknown",
            "when TP and SL are both in one bar, SL is assumed first",
            "no order book, funding or leverage",
            "not financial advice; no orders were placed",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

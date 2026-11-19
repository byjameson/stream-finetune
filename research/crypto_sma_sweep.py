"""SMA robustness sweep for the crypto paper baseline (BTC + ETH).

No real money. Checks whether the SMA-50 trend result survives window changes
and a second asset.

    python research/crypto_sma_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from research.crypto_paper import (FEES, SLIPPAGE, TRAIN_DAYS,
                                       add_features, fetch_klines)
except ImportError:
    from crypto_paper import (FEES, SLIPPAGE, TRAIN_DAYS, add_features,
                              fetch_klines)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results/crypto_sma_sweep.json"
WINDOWS = (20, 50, 100, 200)


def metrics(df, positions, label: str) -> dict:
    eval_df = df.iloc[TRAIN_DAYS:]
    pos = positions.iloc[TRAIN_DAYS:].astype(int).to_numpy()
    target = eval_df["target"].to_numpy()
    changes = np.abs(np.diff(np.concatenate([[0], pos])))
    strat = pos * target - changes * (FEES + SLIPPAGE)
    equity = np.cumprod(1 + strat)
    peak = np.maximum.accumulate(equity)
    bh = np.cumprod(1 + target)
    bh = np.concatenate([[1.0], bh])
    bh_peak = np.maximum.accumulate(bh)
    half = len(strat) // 2
    return {
        "label": label,
        "test_days": int(len(eval_df)),
        "total_return": float(equity[-1] - 1),
        "buy_hold_return": float(eval_df["close"].iloc[-1]
                                 / eval_df["close"].iloc[0] - 1),
        "cagr": float(equity[-1] ** (365 / max(len(eval_df), 1)) - 1),
        "sharpe": float(np.mean(strat) / np.std(strat) * np.sqrt(365))
        if np.std(strat) > 0 else 0.0,
        "max_drawdown": float(np.min(equity / peak - 1)),
        "buy_hold_max_drawdown": float(np.min(bh / bh_peak - 1)),
        "first_half_return": float(np.cumprod(1 + strat[:half])[-1] - 1),
        "second_half_return": float(np.cumprod(1 + strat[half:])[-1] - 1),
        "exposure": float(np.mean(pos)),
        "trades": int(changes.sum()),
    }


def main():
    payload = {"costs": {"fee_per_side": FEES,
                         "slippage_per_side": SLIPPAGE},
               "train_days": TRAIN_DAYS, "assets": {}}
    for symbol, data_path in (
        ("BTCUSDT", ROOT / "data/crypto_btcusdt_1d.json"),
        ("ETHUSDT", ROOT / "data/crypto_ethusdt_1d.json"),
    ):
        raw = fetch_klines(symbol, data_path)
        df = add_features(raw)
        for window in WINDOWS:
            df[f"sma{window}"] = df["close"].rolling(window).mean()
        df = df.dropna().reset_index(drop=True)
        rows = []
        for window in WINDOWS:
            pos = (df["close"] > df[f"sma{window}"]).astype(int)
            rows.append(metrics(df, pos, f"sma{window}"))
        payload["assets"][symbol] = {
            "rows": int(len(df)),
            "start": str(df["date"].iloc[0]),
            "end": str(df["date"].iloc[-1]),
            "results": rows,
        }
    btc = {r["label"]: r for r in payload["assets"]["BTCUSDT"]["results"]}
    eth = {r["label"]: r for r in payload["assets"]["ETHUSDT"]["results"]}
    payload["summary"] = {
        "sma50_beats_bh_btc":
            btc["sma50"]["total_return"] > btc["sma50"]["buy_hold_return"],
        "sma50_beats_bh_eth":
            eth["sma50"]["total_return"] > eth["sma50"]["buy_hold_return"],
        "sma50_beats_bh_both":
            btc["sma50"]["total_return"] > btc["sma50"]["buy_hold_return"]
            and eth["sma50"]["total_return"] > eth["sma50"]["buy_hold_return"],
    }
    payload["verdict"] = ("TREND_CANDIDATE"
                          if payload["summary"]["sma50_beats_bh_both"]
                          else "NO_EDGE")
    payload["caveats"] = [
        "window sweep on one historical path; still in-sample selection",
        "daily close-only execution; no order book or funding",
        "long/flat only; fees and slippage included",
        "not financial advice; no orders were placed",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Bootstrap and start-date robustness for the BTC/ETH SMA-50 paper rule.

No real money. Answers: is the trend result distinguishable from luck, or is it
one path through history?

    python research/crypto_bootstrap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from research.crypto_paper import (FEES, SLIPPAGE, TRAIN_DAYS,
                                       add_features, fetch_klines)
except ImportError:
    from crypto_paper import (FEES, SLIPPAGE, TRAIN_DAYS, add_features,
                              fetch_klines)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results/crypto_bootstrap.json"
BLOCK = 30
N_BOOT = 2000
STARTS = ["2018-01-01", "2019-01-01", "2020-01-01",
          "2021-01-01", "2022-01-01", "2023-01-01"]


def strategy_returns(df):
    pos = (df["close"] > df["sma50"]).astype(int).to_numpy()
    target = df["target"].to_numpy()
    changes = np.abs(np.diff(np.concatenate([[0], pos])))
    return pos * target - changes * (FEES + SLIPPAGE), target


def sharpe(rets):
    std = np.std(rets)
    return float(np.mean(rets) / std * np.sqrt(365)) if std > 0 else 0.0


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1))


def bootstrap(strat, bh, seed=0):
    rng = np.random.default_rng(seed)
    n = len(strat)
    blocks = int(np.ceil(n / BLOCK))
    rows = []
    for _ in range(N_BOOT):
        idx = []
        for _ in range(blocks):
            start = int(rng.integers(0, max(n - BLOCK, 1)))
            idx.extend(range(start, min(start + BLOCK, n)))
        idx = idx[:n]
        s = strat[idx]
        b = bh[idx]
        rows.append((float(np.cumprod(1 + s)[-1] - 1),
                     float(np.cumprod(1 + b)[-1] - 1),
                     sharpe(s), sharpe(b)))
    arr = np.array(rows)
    return {
        "strat_total_p5": float(np.percentile(arr[:, 0], 5)),
        "strat_total_p50": float(np.percentile(arr[:, 0], 50)),
        "strat_total_p95": float(np.percentile(arr[:, 0], 95)),
        "bh_total_p50": float(np.percentile(arr[:, 1], 50)),
        "p_strat_beats_bh": float(np.mean(arr[:, 0] > arr[:, 1])),
        "strat_sharpe_p50": float(np.percentile(arr[:, 2], 50)),
        "bh_sharpe_p50": float(np.percentile(arr[:, 3], 50)),
        "p_strat_sharpe_beats_bh": float(np.mean(arr[:, 2] > arr[:, 3])),
    }


def start_date_table(df, strat, bh):
    rows = []
    for start in STARTS:
        ts = pd.Timestamp(start, tz="UTC")
        idx = np.where(df["date"] >= ts)[0]
        if len(idx) < 120:
            continue
        i = int(idx[0])
        s = strat[i:]
        b = bh[i:]
        eq = np.cumprod(1 + s)
        bheq = np.cumprod(1 + b)
        rows.append({
            "start": start,
            "days": int(len(s)),
            "strat_return": float(eq[-1] - 1),
            "bh_return": float(bheq[-1] - 1),
            "strat_sharpe": sharpe(s),
            "bh_sharpe": sharpe(b),
            "strat_max_dd": max_dd(eq),
            "bh_max_dd": max_dd(bheq),
            "strat_wins": bool(eq[-1] > bheq[-1]),
        })
    return rows


def main():
    payload = {"costs": {"fee_per_side": FEES,
                         "slippage_per_side": SLIPPAGE},
               "bootstrap": {"block_days": BLOCK, "samples": N_BOOT},
               "assets": {}}
    for symbol, path in (
        ("BTCUSDT", ROOT / "data/crypto_btcusdt_1d.json"),
        ("ETHUSDT", ROOT / "data/crypto_ethusdt_1d.json"),
    ):
        df = add_features(fetch_klines(symbol, path))
        df["sma50"] = df["close"].rolling(50).mean()
        df = df.dropna().reset_index(drop=True)
        eval_df = df.iloc[TRAIN_DAYS:].reset_index(drop=True)
        strat, bh = strategy_returns(eval_df)
        payload["assets"][symbol] = {
            "days": int(len(strat)),
            "bootstrap": bootstrap(strat, bh),
            "start_dates": start_date_table(eval_df, strat, bh),
        }
    btc = payload["assets"]["BTCUSDT"]
    eth = payload["assets"]["ETHUSDT"]
    wins = sum(row["strat_wins"] for row in btc["start_dates"]
               + eth["start_dates"])
    total = len(btc["start_dates"]) + len(eth["start_dates"])
    payload["summary"] = {
        "btc_p_beats_bh": btc["bootstrap"]["p_strat_beats_bh"],
        "eth_p_beats_bh": eth["bootstrap"]["p_strat_beats_bh"],
        "start_date_wins": wins,
        "start_date_total": total,
    }
    payload["verdict"] = (
        "ROBUST_TREND"
        if (btc["bootstrap"]["p_strat_beats_bh"] >= 0.90
            and eth["bootstrap"]["p_strat_beats_bh"] >= 0.90
            and wins / max(total, 1) >= 0.75)
        else "FRAGILE"
    )
    payload["caveats"] = [
        "block bootstrap assumes a stationary return process",
        "one history, two assets, no order book or funding",
        "SMA-50 was chosen after looking at the same history",
        "not financial advice; no orders were placed",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Short-term BTC paper research: 1% take-profit / 1% stop-loss.

No API keys, no orders, no real money. The position is sized so that a
stop-loss plus round-trip costs loses at most about 1% of equity.

    python research/crypto_scalp.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/crypto_btcusdt_15m.json"
OUT = ROOT / "research/results/crypto_scalp_btc.json"
SYMBOL = "BTCUSDT"
FEE = 0.001
SLIP = 0.0005
TP = 0.01
SL = 0.01
MAX_HOLD = 16          # 15m bars = 4 hours
RISK = 0.75            # cap net loss near 1% after costs
TEST_START = "2024-01-01"


def fetch(symbol: str = SYMBOL) -> pd.DataFrame:
    if DATA.exists():
        rows = json.loads(DATA.read_text(encoding="utf-8"))
    else:
        rows = []
        start = 1672531200000  # 2023-01-01
        while True:
            r = requests.get(
                "https://data-api.binance.vision/api/v3/klines",
                params={"symbol": symbol, "interval": "15m",
                        "startTime": start, "limit": 1000}, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            start = int(batch[-1][6]) + 1
            if len(batch) < 1000:
                break
            time.sleep(0.2)
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").sort_values(
        "open_time").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].pct_change()
    out["ret4"] = out["close"].pct_change(4)
    out["ret16"] = out["close"].pct_change(16)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma50"] = out["close"].rolling(50).mean()
    out["high20"] = out["high"].rolling(20).max().shift(1)
    out["vol20"] = out["ret1"].rolling(20).std()
    return out.dropna().reset_index(drop=True)


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "momentum": ((df["close"] > df["sma20"])
                     & (df["ret4"] > 0)).to_numpy(),
        "mean_reversion": (df["rsi14"] < 30).to_numpy(),
        "breakout": (df["close"] > df["high20"]).to_numpy(),
        "random": rng.random(len(df)) < 0.05,
    }


def simulate(df: pd.DataFrame, signal: np.ndarray) -> dict:
    trades = []
    equity = 1.0
    i = 0
    n = len(df)
    while i < n - 1:
        if not signal[i]:
            i += 1
            continue
        entry_idx = i + 1
        entry = float(df["open"].iloc[entry_idx]) * (1 + SLIP)
        exit_idx = min(entry_idx + MAX_HOLD - 1, n - 1)
        exit_raw = float(df["close"].iloc[exit_idx])
        reason = "timeout"
        for j in range(entry_idx, min(entry_idx + MAX_HOLD, n)):
            o = float(df["open"].iloc[j])
            h = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            if o <= entry * (1 - SL):
                exit_raw = o
                reason = "sl_gap"
                exit_idx = j
                break
            if o >= entry * (1 + TP):
                exit_raw = o
                reason = "tp_gap"
                exit_idx = j
                break
            hit_sl = low <= entry * (1 - SL)
            hit_tp = h >= entry * (1 + TP)
            if hit_sl and hit_tp:
                exit_raw = entry * (1 - SL)
                reason = "sl_ambiguous"
                exit_idx = j
                break
            if hit_sl:
                exit_raw = entry * (1 - SL)
                reason = "sl"
                exit_idx = j
                break
            if hit_tp:
                exit_raw = entry * (1 + TP)
                reason = "tp"
                exit_idx = j
                break
        exit_fill = exit_raw * (1 - SLIP)
        net = exit_fill / entry - 1 - 2 * FEE
        equity *= 1 + RISK * net
        trades.append({
            "entry_time": str(df["date"].iloc[entry_idx]),
            "exit_time": str(df["date"].iloc[exit_idx]),
            "bars": int(exit_idx - entry_idx + 1),
            "reason": reason,
            "net": float(net),
            "equity": float(equity),
        })
        i = exit_idx + 1
    if not trades:
        return {"trades": 0}
    nets = np.array([t["net"] for t in trades])
    eq = np.array([t["equity"] for t in trades])
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    bh = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    return {
        "trades": len(trades),
        "win_rate": float(np.mean(nets > 0)),
        "avg_net": float(np.mean(nets)),
        "median_net": float(np.median(nets)),
        "best_trade": float(np.max(nets)),
        "worst_trade": float(np.min(nets)),
        "total_return": float(equity - 1),
        "buy_hold_return": bh,
        "max_drawdown": float(np.min(eq / peak - 1)),
        "profit_factor": float(nets[nets > 0].sum()
                               / abs(nets[nets < 0].sum()))
        if (nets < 0).any() else None,
        "tp_hits": int(sum(t["reason"] in ("tp", "tp_gap")
                           for t in trades)),
        "sl_hits": int(sum(t["reason"] in ("sl", "sl_gap",
                                           "sl_ambiguous")
                           for t in trades)),
        "timeouts": int(sum(t["reason"] == "timeout" for t in trades)),
        "avg_bars": float(np.mean([t["bars"] for t in trades])),
        "sample": trades[:5],
    }


def main():
    df = add_features(fetch())
    sig = signals(df)
    test_idx = int(np.where(df["date"] >= pd.Timestamp(TEST_START,
                                                        tz="UTC"))[0][0])
    results = {}
    for name, signal in sig.items():
        full = simulate(df, signal)
        test = simulate(df.iloc[test_idx:].reset_index(drop=True),
                        signal[test_idx:])
        results[name] = {"full": full, "test": test}
    payload = {
        "symbol": SYMBOL,
        "interval": "15m",
        "rows": int(len(df)),
        "start": str(df["date"].iloc[0]),
        "end": str(df["date"].iloc[-1]),
        "test_start": TEST_START,
        "rules": {"take_profit": TP, "stop_loss": SL,
                  "max_hold_bars": MAX_HOLD, "risk_fraction": RISK,
                  "fee_per_side": FEE, "slippage_per_side": SLIP},
        "results": results,
        "verdict": "NO_EDGE",
        "caveats": [
            "15m OHLC only; intrabar path is unknown",
            "when TP and SL are both inside one bar, SL is assumed first",
            "gaps can fill beyond the 1% stop",
            "no order book, funding or leverage",
            "not financial advice; no orders were placed",
        ],
    }
    # Candidate only if the out-of-sample test is positive after costs and
    # beats buy-and-hold in that same period.
    if any(r["test"].get("trades", 0) >= 20
           and r["test"].get("total_return", -1) > 0
           and r["test"].get("total_return", -1)
           > r["test"].get("buy_hold_return", 1)
           for r in results.values()):
        payload["verdict"] = "CANDIDATE"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

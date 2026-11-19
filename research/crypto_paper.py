"""Paper-trading research baseline for BTCUSDT daily candles.

No API keys, no orders, no real money. The goal is to test whether a simple
price-only model has any edge after fees and slippage, before considering the
language model as a feature.

    python research/crypto_paper.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/crypto_btcusdt_1d.json"
OUT = ROOT / "research/results/crypto_paper_btc.json"
SYMBOL = "BTCUSDT"
FEES = 0.001       # 0.10% per side
SLIPPAGE = 0.0005  # 0.05% per side
TRAIN_DAYS = 365
TEST_DAYS = 30


def fetch_klines(symbol: str = SYMBOL) -> pd.DataFrame:
    if DATA.exists():
        rows = json.loads(DATA.read_text(encoding="utf-8"))
    else:
        rows = []
        start = 1502928000000  # 2017-08-17
        while True:
            r = requests.get(
                "https://data-api.binance.vision/api/v3/klines",
                params={"symbol": symbol, "interval": "1d",
                        "startTime": start, "limit": 1000},
                timeout=30)
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
    df = df.drop_duplicates("open_time").sort_values("open_time")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["close"].pct_change()
    for n in (3, 7, 14, 30):
        out[f"ret{n}"] = out["close"].pct_change(n)
    for n in (7, 14, 30):
        out[f"vol{n}"] = out["ret1"].rolling(n).std()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)
    for n in (7, 30, 50):
        out[f"sma{n}"] = out["close"].rolling(n).mean()
        out[f"close_sma{n}"] = out["close"] / out[f"sma{n}"] - 1
    out["sma7_sma30"] = out["sma7"] / out["sma30"] - 1
    out["vol_z"] = ((out["volume"] - out["volume"].rolling(20).mean())
                    / out["volume"].rolling(20).std())
    out["range"] = (out["high"] - out["low"]) / out["close"]
    out["target"] = out["close"].pct_change().shift(-1)
    return out.dropna().reset_index(drop=True)


FEATURES = [
    "ret1", "ret3", "ret7", "ret14", "ret30",
    "vol7", "vol14", "vol30", "rsi14",
    "close_sma7", "close_sma30", "close_sma50", "sma7_sma30",
    "vol_z", "range",
]


def make_model(name: str):
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, random_state=0)
    if name == "logistic":
        return LogisticRegression(max_iter=1000, C=0.1)
    raise ValueError(name)


def walk_forward(df: pd.DataFrame, name: str) -> dict:
    if name == "sma":
        positions = (df["close"] > df["sma50"]).astype(int)
        preds = positions.astype(float)
    else:
        positions = pd.Series(0, index=df.index, dtype=int)
        preds = pd.Series(np.nan, index=df.index, dtype=float)
        start = TRAIN_DAYS
        feature_cols = FEATURES
        while start < len(df):
            train = df.iloc[:start]
            test = df.iloc[start:start + TEST_DAYS]
            if test.empty:
                break
            x_train = train[feature_cols].to_numpy()
            y_train = train["target"].to_numpy()
            x_test = test[feature_cols].to_numpy()
            if name == "logistic":
                y_bin = (y_train > 0).astype(int)
                scaler = StandardScaler().fit(x_train)
                model = make_model(name)
                model.fit(scaler.transform(x_train), y_bin)
                prob = model.predict_proba(scaler.transform(x_test))[:, 1]
                pos = (prob >= 0.5).astype(int)
                preds.loc[test.index] = prob
            else:
                scaler = StandardScaler().fit(x_train)
                model = make_model(name)
                if name == "ridge":
                    model.fit(scaler.transform(x_train), y_train)
                    pred = model.predict(scaler.transform(x_test))
                else:
                    model.fit(x_train, y_train)
                    pred = model.predict(x_test)
                pos = (pred > 0).astype(int)
                preds.loc[test.index] = pred
            positions.loc[test.index] = pos
            start += TEST_DAYS

    eval_df = df.iloc[TRAIN_DAYS:].copy()
    pos = positions.iloc[TRAIN_DAYS:].astype(int)
    target = eval_df["target"].to_numpy()
    pos_arr = pos.to_numpy()
    changes = np.abs(np.diff(np.concatenate([[0], pos_arr])))
    cost = changes * (FEES + SLIPPAGE)
    strat = pos_arr * target - cost
    equity = np.cumprod(1 + strat)
    bh_ret = eval_df["close"].iloc[-1] / eval_df["close"].iloc[0] - 1
    bh_equity = np.cumprod(1 + target)
    bh_equity = np.concatenate([[1.0], bh_equity])
    peak = np.maximum.accumulate(equity)
    bh_peak = np.maximum.accumulate(bh_equity)

    def sharpe(rets):
        std = np.std(rets)
        return float(np.mean(rets) / std * np.sqrt(365)) if std > 0 else 0.0

    direction = None
    if name in ("ridge", "hgb", "logistic"):
        pred_eval = preds.iloc[TRAIN_DAYS:].to_numpy()
        if name == "logistic":
            direction = float(np.mean((pred_eval >= 0.5) == (target > 0)))
        else:
            direction = float(np.mean(np.sign(pred_eval) == np.sign(target)))
    return {
        "model": name,
        "test_days": int(len(eval_df)),
        "total_return": float(equity[-1] - 1),
        "buy_hold_return": float(bh_ret),
        "cagr": float(equity[-1] ** (365 / max(len(eval_df), 1)) - 1),
        "sharpe": sharpe(strat),
        "max_drawdown": float(np.min(equity / peak - 1)),
        "buy_hold_max_drawdown": float(np.min(bh_equity / bh_peak - 1)),
        "exposure": float(np.mean(pos_arr)),
        "trades": int(changes.sum()),
        "direction_accuracy": direction,
        "final_equity": float(equity[-1]),
    }


def main():
    df = fetch_klines()
    df = add_features(df)
    print(f"rows {len(df)} from {df.date.iloc[0]} to {df.date.iloc[-1]}")
    results = [walk_forward(df, name)
               for name in ("sma", "ridge", "hgb", "logistic")]
    payload = {
        "symbol": SYMBOL,
        "interval": "1d",
        "rows": int(len(df)),
        "start": str(df.date.iloc[0]),
        "end": str(df.date.iloc[-1]),
        "costs": {"fee_per_side": FEES, "slippage_per_side": SLIPPAGE},
        "walk_forward": {"train_days": TRAIN_DAYS,
                         "test_days": TEST_DAYS},
        "results": results,
        "verdict": "NO_EDGE" if all(
            r["total_return"] <= r["buy_hold_return"] for r in results)
            else "CHECK",
        "caveats": [
            "daily close-only simulation; no order book, funding or liquidation",
            "long/flat only; no short, leverage or portfolio allocation",
            "fixed 0.10% fee + 0.05% slippage per side",
            "walk-forward test, but still one asset and one history",
            "not financial advice; no real orders were placed",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

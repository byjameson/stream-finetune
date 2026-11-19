# Crypto paper-trading baseline (BTCUSDT, daily)

**No API keys, no orders, no real money.** This is a research baseline, not
financial advice.

## Question

Does a simple price-only model have any edge on BTCUSDT daily candles after
fees and slippage? If the simple baselines cannot beat buy-and-hold, there is
no reason to connect the language model to a trading loop.

## Setup

- Data: Binance public `BTCUSDT` daily klines, 2017-10-05 to 2026-11-18,
  3,332 rows (cached in `data/crypto_btcusdt_1d.json`).
- Features: 1/3/7/14/30-day returns, 7/14/30-day volatility, RSI-14,
  close/SMA-7/30/50, SMA-7/SMA-30, volume z-score, daily range.
- Target: next-day close-to-close return.
- Walk-forward: 365 training days, 30 test days, expanding window, retrain
  every 30 days. No shuffling and no test data in training.
- Costs: **0.10% fee + 0.05% slippage per side**.
- Long/flat only; no short, leverage, funding or order book.

## Results (2,967 out-of-sample days)

| model | total return | CAGR | Sharpe | max drawdown | exposure | trades | direction acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMA-50 trend | **36.0x** | 55.9% | **1.26** | -58.5% | 53.7% | 162 | - |
| Ridge | 8.80x | 32.4% | 0.82 | -65.1% | 60.3% | 584 | 50.9% |
| HistGradientBoosting | 1.66x | 6.5% | 0.37 | -58.9% | 57.7% | 818 | 48.9% |
| Logistic | 1.56x | 5.6% | 0.37 | -76.3% | 64.2% | 923 | 51.3% |
| Buy-and-hold | 8.50x | 31.4% | - | -76.6% | 100% | 1 | - |

Verdict: **CHECK**, not a profit claim.

- The learned models do **not** show a reliable edge: Ridge slightly beats
  buy-and-hold in total return but has a worse Sharpe; HGB and logistic are
  far behind.
- The only strong result is a hand-coded SMA-50 trend rule. That is a
  **trend premium candidate**, not a machine-learning edge and not the
  spiking LM.
- One asset, one history, one arbitrary SMA window. Before believing it, sweep
  SMA-20/50/100/200 on BTC and ETH, vary start dates, and paper-trade forward.

## Robustness sweep (BTC + ETH)

`research/crypto_sma_sweep.py` repeats the trend rule on BTC and ETH with
SMA-20/50/100/200. The common test window is 2018-04-22 to 2026-11-18 (2,768
out-of-sample days).

| asset | SMA-20 | SMA-50 | SMA-100 | SMA-200 | buy-and-hold |
|---|---:|---:|---:|---:|---:|
| BTC total return | 5.00x | **30.51x** | 19.06x | 8.01x | 10.77x |
| ETH total return | 13.15x | **35.87x** | 6.79x | 13.67x | 9.42x |

- SMA-50 beats buy-and-hold on both assets, with lower drawdowns.
- The result is **window-sensitive**: SMA-20 loses on BTC, SMA-100 and SMA-200
  lose on ETH. A single 50-day window on one history is not proof.
- Second-half returns are positive but much smaller than first-half returns;
  the edge is regime-dependent.
- Verdict: **TREND_CANDIDATE**, not a deployable strategy.

## What this means for the spiking model

- The chat model has no price input and no risk model; it cannot trade.
- If text is used later, it should be **one feature** in a separate
  time-series/risk system, not a direct order generator.
- Continual learning can adapt to a regime, but it can also learn noise;
  walk-forward evaluation and costs are the gate.

## Next experiments

1. Bootstrap confidence intervals and start-date sensitivity.
2. Forward paper-trading ledger with daily marks, no exchange credentials.
3. Add order-book/funding data and execution-at-next-open instead of
   close-only.
4. Only if a robust edge survives: a tiny risk-managed allocation, still
   paper-only until reviewed.

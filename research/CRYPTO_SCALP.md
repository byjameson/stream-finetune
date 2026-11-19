# Short-term BTC paper research: 15m, 1% TP / 1% SL

**No API keys, no orders, no real money.** This tests whether a short-term
long-only rule can survive a 1% take-profit and 1% stop-loss after costs.

## Setup

- BTCUSDT 15-minute candles, 2023-01-01 to 2026-11-19 (136,146 bars).
- Entry at the next bar open; exit at +1% / -1% / 16 bars (4 hours).
- Position size 0.75 of equity, so a stop plus round-trip costs loses about
  **0.94% of equity** (gross stop is 1%).
- Costs: 0.10% fee + 0.05% slippage per side. Net TP is +0.75%, net SL is
  -1.25% on the position.
- If a bar contains both TP and SL, the stop is assumed first.
- Out-of-sample test: entries from **2024-01-01** onward.

## Out-of-sample results

| signal | trades | win rate | avg net/trade | total return | profit factor |
|---|---:|---:|---:|---:|---:|
| momentum | 5,334 | 32.98% | -0.304% | -99.9995% | 0.375 |
| mean reversion | 1,130 | 38.85% | -0.331% | -94.09% | 0.410 |
| breakout | 2,272 | 32.79% | -0.312% | -99.53% | 0.373 |
| random | 3,108 | 32.79% | -0.301% | -99.91% | 0.362 |
| buy-and-hold | - | - | - | +47.71% | - |

With net +0.75% on a win and -1.25% on a loss, the break-even win rate is
**62.5%**. The best signal reaches 38.9%; the random control is 32.8%.

## Verdict

**NO_EDGE.** Short-term 1% TP / 1% SL does not survive realistic costs in these
baselines. The per-trade loss is small but repeated thousands of times, so the
compounded equity collapses. This is a cost/win-rate problem, not a parameter
tuning problem.

## Horizon and cost sweep

`research/crypto_scalp_sweep.py` repeats the test on 15m, 1h and 4h bars with
taker costs (0.10% + 0.05%) and optimistic maker costs (0.02% + 0.02%).

- all **24** signal/timeframe/cost combinations are negative after costs;
- the best row is 4h breakout with maker costs: win rate 48.4%, average net
  -0.068% per trade, total **-11.3%** versus buy-and-hold **+48.3%**;
- at maker costs the break-even win rate is about 52%, so 48.4% is still short;
- the 15m rows are the worst: thousands of trades and near-total equity loss.

Verdict: **NO_EDGE** across horizons and cost assumptions. A shorter timeframe
does not fix a low hit rate; it multiplies the cost drag.

## What would have to change

- a signal with a genuinely higher hit rate (not a different TP/SL only),
- maker fees or lower costs,
- a longer horizon where the average move is larger than costs,
- or a different asset/venue with less spread.

Even then, paper trading and forward validation come first. No real money.

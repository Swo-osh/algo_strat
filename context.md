
# AlgoGators Pitch: VIX Term Structure Mean Reversion Strategy

## Context
Building a systematic trading strategy pitch for AlgoGators (University of Florida quant fund).
Required sections: Core Concept, Economic Rationale, Detailed Implementation, Risk Assessment,
Capital and Liquidity, Analytical Evidence. Limit is roughly 2-3 pages plus 1 page of charts/tables.

---

## The Strategy
**Trade SPY (or short volatility instruments) when the VIX term structure enters steep backwardation**, betting on mean reversion of implied volatility back to its long-run average.

- **Mechanism**: The VIX measures 30-day implied volatility on the S&P 500. VIX3M measures
90-day implied volatility. Normally the term structure is in **contango** (VIX < VIX3M) —
short-term fear is lower than longer-term uncertainty. When markets spike, the curve
**inverts into backwardation** (VIX > VIX3M), meaning short-term fear is unusually elevated
relative to the medium term. This backwardation is historically short-lived and tends to
mean-revert, creating a predictable trading window.
- **Signal**: `VIX / VIX3M ratio > 1.0` = backwardation (inverted curve). The steeper the
inversion, the stronger the mean-reversion signal.
- **Framing**: This is a **volatility mean-reversion trade**, not a directional equity bet.
The edge comes from the structural tendency of fear spikes to overshoot and revert, not from
predicting market direction. Important distinction to state explicitly in the pitch.
- **Tradeable instruments**:
- **SPY** (long, as volatility crush = equity rally) — simplest, liquid, no futures account needed
- **SVXY** (ProShares Short VIX Short-Term Futures ETF) — direct short-vol exposure
- **VIX futures** (short front-month) — for futures-enabled accounts
- **Current relevance**: VIX spikes above 30-40 have historically been followed by rapid
mean reversion within 10-30 trading days, making this a well-documented, repeatable setup
(COVID crash 2020, Aug 2024 Yen carry unwind, etc.).

---

## The 8-Step Build Plan
1. **Pull historical data** — VIX and VIX3M daily closing prices (free via `yfinance`: `^VIX`, `^VIX3M`),
SPY prices for P&L calculation.
2. **Compute the VIX/VIX3M ratio** — identify all backwardation events (ratio > 1.0) and classify
by severity (mild: 1.0–1.1, moderate: 1.1–1.2, severe: > 1.2).
3. **Build the event series** — define entry (ratio crosses above threshold) and exit
(ratio reverts below 1.0, or fixed holding period of 10/15/20 trading days).
4. **Test the relationship statistically** — measure average SPY return in the N days following
each backwardation event, split by severity tier. Run a t-test vs. unconditional average SPY
returns.
5. **Define concrete trading rules** — entry trigger (ratio > 1.1), position sizing (fixed
fractional), exit rule (ratio < 1.0 OR 15-day max hold), stop-loss (ratio > 1.4 = sizing down).
6. **Backtest and validate out-of-sample** — train on 2004–2019 events, test on 2020–present
(includes COVID spike and Aug 2024 — two very different regimes).
7. **Write the pitch** to the six required sections.
8. **Build 1-2 charts** (VIX/VIX3M ratio over time with entry signals marked; cumulative P&L
vs. SPY buy-and-hold) and stress-test against the worst drawdown event before submitting.

---

## Key Metrics to Report in Pitch
- **Win rate** of entries (% of backwardation events where SPY was higher 15 days later)
- **Average return per trade** vs. unconditional 15-day SPY return
- **Sharpe ratio** of the strategy vs. buy-and-hold
- **Max drawdown** — especially during prolonged vol regimes (e.g., 2008, COVID)
- **Number of trades** — need enough events to be statistically meaningful (target: 30+)

---

## Risk Factors to Address
- **Volatility regime persistence**: during 2008 and COVID, backwardation persisted for weeks —
the mean reversion trade can experience deep drawdowns before working.
- **Gap risk**: VIX can spike further overnight; stops may not protect against gap moves.
- **Instrument decay**: SVXY and similar ETFs have structural decay in prolonged vol regimes —
must flag this if using leveraged instruments.
- **Crowded trade**: short-vol is a well-known strategy; crowding can exacerbate losses during
unwinds (e.g., Feb 2018 "Volmageddon").

---

## Data Sources
| Series | Ticker | Source |
|--------|--------|--------|
| VIX (30-day implied vol) | `^VIX` | yfinance (free) |
| VIX3M (90-day implied vol) | `^VIX3M` | yfinance (free) |
| SPY (S&P 500 ETF) | `SPY` | yfinance (free) |
| SVXY (short-vol ETF) | `SVXY` | yfinance (free) |

---

## Next Steps
1. Pull VIX, VIX3M, and SPY data in Python
2. Compute ratio and classify backwardation events
3. Run event-study: average forward returns by severity tier
4. Define trade rules and backtest
5. Draft the six pitch sections using results
OUTPUT
{
  "status": "success",
  "id": "a0d18bd2-5421-4b5f-9d32-92aa9cdaf019",
  "title": "plan.md — VIX Term Structure Mean Reversion Strategy",
  "created_at": 1788398399815428400
}
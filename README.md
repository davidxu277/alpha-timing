<div align="center">

# Alpha Timing

### Regime-Aware Quantitative Trading

*A market-regime classifier routes each trading day to a specialist return predictor —*
*trading a small drag in bull markets for large protection in bear markets.*

**English** | [中文文档](README.zh.md)

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![sklearn](https://img.shields.io/badge/built%20with-scikit--learn-orange)
![disclaimer](https://img.shields.io/badge/⚠-not%20investment%20advice-lightgrey)

Xu Shuyao · NUS · 2026 · [LinkedIn](https://www.linkedin.com/in/shuyao-xu-4841103a7)

</div>

---

## Overview

**Alpha Timing** asks one question: *if a model can tell what state the market is currently in,
can it size positions more intelligently than simply staying fully invested?*

The system has three stages. A **regime classifier** identifies the market's current state each day
(Bull / Bear / Sideways / Crisis). Based on that state, one of **four specialist models** predicts the
near-term return. A **decision layer** turns that prediction into a position between 0 (cash) and 1
(fully long), with a defensive bias in risk-off regimes and an automatic volatility brake.

```
                       ┌──────────────────────────────┐
 daily features ──────►│ regime nowcaster (RandomForest)
 (trailing only)       │ Bull / Bear / Sideways / Crisis
                       └───────────────┬──────────────┘
                                       │ routes to
                       ┌───────────────▼──────────────┐
                       │ 4 regime specialists (GBT)   │
                       │ predict 5-day excess return  │
                       └───────────────┬──────────────┘
                                       │ ŷ
                       ┌───────────────▼──────────────┐
                       │ dual-threshold hysteresis    │
                       │ (defensive in Bear/Crisis)   │
                       │ + volatility targeting       │
                       └───────────────┬──────────────┘
                                       ▼
                              position ∈ [0, 1]
```

**One-line result:** on an out-of-universe SPY test (Nov 2021 → 2026, dividends, 5 bps costs) the
system delivers **Sharpe 1.05 vs 0.69** for buy-and-hold and **max drawdown −14% vs −25%**.

---

## How it works (step by step)

### 1 · Regime nowcaster

A `RandomForestClassifier` (300 trees, balanced class weights) labels each day as one of four regimes
from **16 trailing features**: momentum, moving-average deviation, realized volatility and price
position (12 technical), plus VIX, fed-funds rate, unemployment and the 10y–2y yield spread (4 macro).

- **Nowcasting, not forecasting.** Every feature ends at day *t* and the label is day *t*'s regime —
  it answers *"what state are we in now?"*, not *"what happens next?"*.
- Out-of-sample accuracy ≈ **88%**; the model independently recovers the textbook description of
  market states (trend + volatility), consistent with the regime-switching literature
  (Hamilton 1989; Ang & Bekaert 2002). Studied in isolation in the companion repo
  [regime-classifier](https://github.com/davidxu277/regime-classifier).

### 2 · Regime specialists

Four `HistGradientBoostingRegressor` models — one per regime. Each is trained **only on days of its own
regime** and predicts the **same target: the 5-day return in excess of cash**, from the 12
vol-normalized technical factors.

- **Specialization comes from data, not from the objective.** The Bull expert only ever sees bull
  days, the Crisis expert only crisis days — so they learn opposite reactions to the same signal
  (a sharp drop is a dip to buy in a bull, a knife not to catch in a crisis). One pooled model would
  average these conflicting rules away; four specialists keep them.
- **What a prediction actually is.** A gradient-boosted tree is a stack of ~300 shallow if-else trees
  whose leaves store *the average forward return of the historical days that landed in that leaf*. A
  prediction of +0.35% literally means *"across the historical days most similar to today, the next
  5 days beat cash by 0.35% on average."* No hand-set formulas — the model is a compression of history
  into a lookup table.

### 3 · Decision layer

The specialist's prediction ŷ becomes a position through two mechanisms, detailed as highlights below:
a **dual-threshold hysteresis** rule (when to be in/out) and a **volatility-target overlay** (how much).

**Causality throughout:** features are trailing; at deployment the regime is always the classifier's
*prediction* (never a label); a position chosen at close *t* earns the return of *t*→*t+1*; every
position change pays 5 bps; idle cash earns interest.

---

## Highlight 1 · Volatility-target position sizing

A risk *thermostat*. Instead of holding a fixed amount of money, the system holds a fixed amount of
**risk**:

```
position = min(1, 25% / current_volatility)
```

*current_volatility* is the standard deviation of the last 20 daily returns, annualized (×√252). The
logic is one line of algebra — portfolio risk ≈ position × asset volatility, so to hold risk constant
at 25% you set `position = 25% / volatility`. The `min(1, …)` caps at 100% (no leverage).

| Market state | Current volatility | Position `min(1, 25%/vol)` |
|---|---|---|
| Calm bull | 15% | 100% (capped) |
| Normal | 30% | 83% |
| Turbulent | 50% | 50% |
| Crash | 80% | 31% |

The wilder the market, the more risk each dollar carries, so the system automatically invests less.
It is continuous, automatic and requires **no prediction** — during the March-2020 volatility spike,
exposure was already falling before any model raised a flag.

---

## Highlight 2 · Dual-threshold hysteresis

Position changes require crossing a **band**, not a line — this absorbs the day-to-day flicker of a
low-signal prediction so transaction costs don't eat the strategy alive. When flat, ŷ must clear the
**entry line** to buy; when long, ŷ must fall below the **exit line** to sell; in between, nothing
changes.

The two threshold pairs are **regime-asymmetric** — and the asymmetry was *chosen by the validation
grid search*, not imposed:

| Regime group | Entry line | Exit line | Behaviour |
|---|---|---|---|
| **Risk-on** (Bull / Sideways) | +0.1% | −0.3% | easy in, hard out → **stays invested** |
| **Risk-off** (Bear / Crisis) | +0.3% | 0.0% | needs a strong signal to enter, exits the moment ŷ turns negative → **defensive** |

"Slow in, fast out" during Bear/Crisis is the structural source of the drawdown protection. That the
optimizer picked this asymmetric shape over the symmetric candidates in the grid means **the data
itself supports "ride bull markets, respect bear markets."**

---

## Highlight 3 · Is the regime classifier even necessary? (ablation)

A fair challenge: if similar feature-days are already similar, isn't the classifier redundant — can't
one strong model learn everything? We tested it by **removing the classifier** and re-measuring on the
same SPY test:

| Variant | Total | Sharpe | maxDD | Role of classifier |
|---|---|---|---|---|
| **A · full system** (4 specialists + regime thresholds) | **+83.8%** | **1.05** | −13.9% | fully used |
| C · single pooled model + regime thresholds | +55.4% | 0.78 | −15.4% | only sets defensive thresholds |
| B · single pooled model + one threshold | +52.6% | 0.80 | −15.7% | removed entirely |
| Buy & Hold | +56.5% | 0.69 | −24.5% | — |

Removing the classifier collapses Sharpe from **1.05 to ~0.80** — the routing contributes most of the
edge. Two reasons it is *not* a redundant re-partitioning of the same features:

1. **Different supervision targets.** The classifier is trained on the *regime label*; the specialists
   on *forward return*. Splitting by "what state is this" versus splitting by "what predicts return"
   carves genuinely different boundaries — the classifier injects a notion of market state that
   pure return-prediction, buried under noise, never discovers on its own.
2. **Inductive bias under low signal-to-noise.** Even if a single deep model *could* represent the
   interaction, in a regime where the true signal is ~2% of daily noise it fails to *learn* it. Routing
   hands the model that structure for free. (In the infinite-data limit a single model subsumes both —
   the two-stage design earns its keep precisely in the finite, noisy world we actually trade in.)

Reproduce with `python regime_trading.py --ablation`.

---

## Results — SPY transfer test

**Protocol:** SPY adjusted close (dividends included) from 2005, chronological split, **test = the last
20% (Nov 2021 → present)**; 5 bps costs, idle cash at 4%/yr. **Nothing is tuned on SPY** — the models
come from the 35-stock training universe, so this is a pure out-of-universe transfer.

| Strategy | Total return | Sharpe | maxDD | Annual vol | Time in market |
|---|---|---|---|---|---|
| **System** | **+83.8%** | **1.05** | **−13.9%** | 14.7% | 79% |
| Buy & Hold | +56.5% | 0.69 | −24.5% | 17.8% | 100% |
| MA20>60 | +28.7% | 0.59 | −17.2% | 11.2% | 65% |

![SPY NAV, drawdown and exposure](figures/spy_test.png)

![SPY metrics](figures/spy_metrics.png)

The bottom panel shows exposure: through the 2022 bear market the classifier flags risk-off (red
shading) and the system steps into cash, then stays near fully invested through the 2023–2025 recovery.

**Yearly attribution (System − Buy & Hold):**

| Year | System | Buy & Hold | Excess |
|---|---|---|---|
| **2022** | **−1.6%** | **−19.0%** | **+17.4pp** |
| 2023 | +21.7% | +26.0% | −4.3pp |
| 2024 | +30.1% | +25.3% | +4.8pp |
| 2025 | +14.0% | +18.2% | −4.2pp |

The edge is earned by sidestepping the 2022 bear market; the bull years roughly cancel out. This is
the design working as intended, not a lucky year — a regime-aware allocator is *supposed* to give up a
little upside for a lot of protection.

---

## Development process

The final architecture is the survivor of several discarded ones:

1. **Factor IC scan.** Before modelling, an information-coefficient study across 39 tickers showed that
   at 1–20 day horizons every trend factor has *negative* IC (short-term reversal) and fear factors
   (VIX spikes) positive IC. It also fixed expectations: median |IC| ≈ 0.02–0.06 — the signal is a few
   percent of the noise, which shaped every later choice.
2. **Reinforcement learning, abandoned.** A per-regime DQN was the first trading model. It proved
   structurally unstable: with the true long-vs-flat edge at ~2% of daily noise, the value estimates
   were dominated by noise and the learned policy flipped between runs (conclusions reversed between
   600 and 2500 training episodes on the same seed). Diagnosis: value-based RL is the wrong tool at
   this signal-to-noise ratio.
3. **Supervised specialists.** Replacing the DQN with gradient-boosted regressors — which average tens
   of thousands of samples rather than learning by trial and error — made results deterministic and
   reproducible.
4. **v1 → v2 defensive redesign.** A first version tuned by Sharpe on a benign window learned no
   defense (drawdown −35%, same as buy-and-hold). v2 fixed it with regime-asymmetric thresholds,
   Calmar-based selection on a window containing real corrections, and the volatility-target overlay.
5. **Ablations.** Oracle-routing showed the classifier's *errors* cost little; the ablation above
   showed the classifier itself carries most of the edge. Each experiment justified a component.

---

## Design decisions & tradeoffs

- **The edge is event-driven (n ≈ 2).** Excess return concentrates in a few correct risk-off calls
  (2020, 2022). This is the honest core limitation and the main thing future work should de-risk.
- **Losing Sharpe to buy-and-hold inside a pure bull year is structural, not a bug.** In a smooth
  uptrend buy-and-hold is optimal; any strategy that ever de-risks must give up some risk-adjusted
  return there. The right yardstick is the **full cycle** (Sharpe 1.05 vs 0.69), not any single year.
- **No label smoothing.** An experiment to smooth the choppy regime labels was rejected: merging short
  runs needs to know *future* run lengths, and it systematically delayed regime flips at market tops.
  All de-noising lives in the causal decision layer instead.
- **Macro enters only through the classifier.** The specialists use pure technical factors (comparable
  across stocks); macro (VIX, rates) is the same for all names on a day, so it informs *state*, not the
  cross-sectional return prediction.

---

## Improvement directions

- **More bear markets** (2008, 2000) to test whether the protection is a repeatable rule rather than a
  2022 artifact — the highest-value next step given the n ≈ 2 dependence.
- **Leading cross-asset signals** (credit spreads, VIX term structure) to shorten the classifier's lag.
- **Reduce 2025-style false-alarm risk-off** — cutting false positives is "free" because it doesn't
  weaken genuine bear-market protection.
- **Test macro inside the specialists**, and a continuous (multi-level) position instead of the binary
  entry/exit.

---

## Data

`stock_market_regimes_2000_2026.csv` (~35 MB, included): [Stock Market Regimes (2000–2026)](https://www.kaggle.com/datasets/mafaqbhatti/stock-market-regimes-20002026)
by Muhammad Afaq Bhatti on Kaggle, Apache 2.0 — daily prices, regime labels and macro data for 39 US
tickers. `regime_trading.py` additionally downloads SPY adjusted prices from Yahoo at runtime.

## Usage

```bash
pip install -r requirements.txt

python regime_trading.py             # train the full stack + SPY test + figures
python regime_trading.py --ablation  # reproduce the classifier-necessity ablation (A/B/C)
```

The entire system lives in a single file, [regime_trading.py](regime_trading.py).

## Limitations

Event-driven edge (n ≈ 2 bear markets); regime labels are third-party (price/volatility-defined);
single final test window; results assume daily-close execution at 5 bps.

## License

Code released under the [MIT License](LICENSE). Dataset under Apache 2.0 (see Data).

## Disclaimer

For research and educational purposes only. **Not investment advice**; backtest results are not
evidence of future performance.

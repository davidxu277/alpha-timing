"""
update_signals.py — daily inference for the live dashboard.

Loads the trained stack (local gzip cache if present, otherwise retrains from the
committed CSV), fetches live prices (Yahoo) + macro (FRED), runs the full pipeline
independently for each of the six dashboard tickers, and writes docs/data/signals.json.

The model is daily-close, so this is meant to run once per trading day (see the
GitHub Actions workflow). No look-ahead: every feature is trailing and the regime
is always the classifier's prediction.

    python docs/update_signals.py
"""
from __future__ import annotations

import gzip
import io
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from regime_trading import (  # noqa: E402
    build_factors, classifier_features, system_position, download_yahoo,
    FACTORS, CLF_FEATURES, REGIMES, RISK_OFF, VOL_TARGET_D, DATA,
    prepare, train_stack, select_thresholds,
)

CACHE = os.path.join(HERE, "data", "models.pkl.gz")
OUT = os.path.join(HERE, "data", "signals.json")
UA = {"User-Agent": "Mozilla/5.0"}
MACRO_COLS = ["vix", "fed_funds_rate", "unemployment_rate", "10y_treasury", "2y_treasury"]

# dashboard universe: symbol -> display label
TICKERS = {
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq-100 (QQQ)",
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
}

HIST_DAYS = 120          # points kept for the front-end sparklines
WARMUP_DAYS = 320        # trailing window fed to the path-dependent hysteresis


# ── models ────────────────────────────────────────────────────────────────────
def load_models() -> dict:
    if os.path.exists(CACHE):
        print(f"loading cached models: {CACHE}")
        with gzip.open(CACHE, "rb") as f:
            return pickle.load(f)
    print("no cache — training from CSV ...")
    df = prepare()
    clf, scaler, spec = train_stack(df)
    th_on, th_off = select_thresholds(df)
    return {"clf": clf, "scaler": scaler, "spec": spec, "th_on": th_on, "th_off": th_off}


# ── macro (all sources verified reachable; no API key) ────────────────────────
def fetch_treasury() -> pd.DataFrame:
    """Live daily 2y / 10y par yields from the U.S. Treasury (this year + last)."""
    year = pd.Timestamp.now().year
    frames = []
    for y in (year, year - 1):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
               f"daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve"
               f"&field_tdr_date_value={y}&page&_format=csv")
        with urlopen(Request(url, headers=UA), timeout=30) as resp:
            frames.append(pd.read_csv(io.StringIO(resp.read().decode())))
    t = pd.concat(frames)
    t.index = pd.to_datetime(t["Date"])
    return pd.DataFrame({"2y_treasury": pd.to_numeric(t["2 Yr"], errors="coerce"),
                         "10y_treasury": pd.to_numeric(t["10 Yr"], errors="coerce")}).sort_index()


def fetch_slow_macro() -> pd.DataFrame:
    """Fed-funds and unemployment from the bundled CSV (the series the model trained on).

    Both move at monthly-or-slower cadence, so carrying the last published value forward
    is a faithful proxy — and keeps macro consistent with training, no third-party key."""
    kg = pd.read_csv(DATA, parse_dates=["date"])
    g = kg[kg["ticker"] == "^GSPC"][["date", "fed_funds_rate", "unemployment_rate"]].dropna()
    return g.set_index("date").sort_index()


def fetch_macro() -> pd.DataFrame:
    """Daily, forward-filled macro frame with the five columns the classifier expects."""
    vix = download_yahoo("^VIX").set_index("date")["close"].rename("vix")
    tr = fetch_treasury()
    slow = fetch_slow_macro()

    idx = pd.date_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=900),
                        pd.Timestamp.now().normalize(), freq="D")
    macro = pd.DataFrame(index=idx)
    macro["vix"] = vix.reindex(idx).ffill()
    macro["10y_treasury"] = tr["10y_treasury"].reindex(idx).ffill()
    macro["2y_treasury"] = tr["2y_treasury"].reindex(idx).ffill()
    macro["fed_funds_rate"] = slow["fed_funds_rate"].reindex(idx).ffill()
    macro["unemployment_rate"] = slow["unemployment_rate"].reindex(idx).ffill()
    macro.index.name = "date"
    return macro


# ── per-ticker inference ──────────────────────────────────────────────────────
def advice_for(cur: float, prev: float, regime: str, yhat: float) -> tuple[str, str]:
    """Turn a position transition into a buy/sell call + one-line rationale."""
    pct = f"{yhat * 100:+.2f}%"
    if prev <= 0 < cur:
        return "BUY", f"{regime} regime — the 5-day forecast ({pct}) cleared the entry line."
    if prev > 0 >= cur:
        return "SELL", f"{regime} regime — forecast fell below the exit line; step to cash."
    if cur <= 0:
        return "CASH", f"{regime} regime — signal ({pct}) stays below the entry line; hold cash."
    if cur - prev > 0.03:
        return "ADD", f"{regime} regime — falling volatility lets exposure rise toward target."
    if prev - cur > 0.03:
        return "TRIM", f"{regime} regime — volatility rising; the vol target trims exposure."
    return "HOLD", f"{regime} regime — forecast {pct}; stay invested at target exposure."


def signal_for(symbol: str, label: str, macro: pd.DataFrame, m: dict) -> dict:
    px = download_yahoo(symbol)
    g = px.merge(macro.reset_index(), on="date", how="left").sort_values("date")
    g[MACRO_COLS] = g[MACRO_COLS].ffill()
    g = g.reset_index(drop=True)

    feat = pd.concat([g[["date", "close"]], build_factors(g["close"]),
                      classifier_features(g)], axis=1).dropna(subset=FACTORS + CLF_FEATURES)
    feat = feat.tail(WARMUP_DAYS).reset_index(drop=True)

    regp = m["clf"].predict(feat[CLF_FEATURES])
    conf = m["clf"].predict_proba(feat[CLF_FEATURES]).max(axis=1)
    X = m["scaler"].transform(feat[FACTORS])
    yhat = np.zeros(len(feat))
    for r in REGIMES:
        mask = regp == r
        if mask.any():
            yhat[mask] = m["spec"][r].predict(X[mask])

    riskoff = np.isin(regp, list(RISK_OFF))
    volscale = np.clip(VOL_TARGET_D / np.maximum(feat["vol_20"].to_numpy(), 1e-6), 0, 1)
    pos = system_position(yhat, riskoff, volscale, m["th_on"], m["th_off"])

    close = feat["close"].to_numpy()
    dates = feat["date"].dt.strftime("%Y-%m-%d").to_numpy()
    cur, prev = float(pos[-1]), float(pos[-2])
    regime = str(regp[-1])
    action, reason = advice_for(cur, prev, regime, float(yhat[-1]))

    tail = feat.tail(HIST_DAYS).index
    history = [{"date": dates[i], "price": round(float(close[i]), 2),
                "regime": str(regp[i]), "position": round(float(pos[i]), 3)}
               for i in tail]

    return {
        "label": label,
        "regime": regime,
        "regime_conf": round(float(conf[-1]), 3),
        "price": round(float(close[-1]), 2),
        "change_1d": round(float(close[-1] / close[-2] - 1), 4),
        "yhat_5d": round(float(yhat[-1]), 4),
        "position": round(cur, 3),
        "prev_position": round(prev, 3),
        "vol_annual": round(float(feat["vol_20"].iloc[-1] * np.sqrt(252)), 3),
        "vol_cap": round(float(volscale[-1]), 3),
        "advice": action,
        "advice_reason": reason,
        "as_of": dates[-1],
        "history": history,
    }


def main() -> None:
    m = load_models()
    macro = fetch_macro()
    print(f"macro through {macro.index.max().date()} "
          f"(fed funds {macro['fed_funds_rate'].iloc[-1]:.2f}, "
          f"10y {macro['10y_treasury'].iloc[-1]:.2f}, vix {macro['vix'].iloc[-1]:.1f})")

    out = {}
    for symbol, label in TICKERS.items():
        try:
            out[symbol] = signal_for(symbol, label, macro, m)
            s = out[symbol]
            print(f"  {symbol:<5} {s['regime']:<9} yhat {s['yhat_5d']:+.4f} "
                  f"pos {s['position']:.2f} -> {s['advice']}")
        except Exception as e:  # keep the board alive if one symbol fails
            print(f"  {symbol:<5} FAILED: {e}")

    if not out:
        raise SystemExit("no signals produced — aborting write")

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "note": "Signals update once per trading day; the model is daily-close. Research only — not investment advice.",
        "tickers": out,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT} ({len(out)} tickers)")


if __name__ == "__main__":
    main()

"""
build_models.py — train the stack once and cache it for fast local iteration.

Runs the exact same training as regime_trading.py's main() (nowcaster + 4 GBT
specialists on <2013, thresholds by Calmar on 2013-2018), then dumps everything
to docs/data/models.pkl.gz. This is a *local cache* only (git-ignored): the daily
CI job retrains from the committed CSV instead of shipping a large binary. Running
this first just makes local dashboard runs start instantly.

Run locally (re-run only if the model or training data changes):
    python build_models.py
"""
import gzip
import os
import pickle

from regime_trading import prepare, train_stack, select_thresholds, REGIMES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "data", "models.pkl.gz")


def main() -> None:
    print("preparing data ...")
    df = prepare()
    print(f"  {df['ticker'].nunique()} stocks, {len(df):,} rows")

    print("training nowcaster + 4 specialists (<2013) ...")
    clf, scaler, spec = train_stack(df)

    print("selecting thresholds on 2013-2018 by Calmar ...")
    th_on, th_off = select_thresholds(df)
    print(f"  risk-on(in,out)={th_on}  risk-off(in,out)={th_off}")

    assert set(spec) == set(REGIMES), f"missing specialists: {set(REGIMES) - set(spec)}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "spec": spec,
                     "th_on": th_on, "th_off": th_off}, f)
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\nsaved {OUT} ({size_mb:.1f} MB, local cache)")
    print("specialists:", ", ".join(sorted(spec)))


if __name__ == "__main__":
    main()

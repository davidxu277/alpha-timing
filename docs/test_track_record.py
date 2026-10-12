"""Tests for the track-record ledger — run: python docs/test_track_record.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

import append_track_record as T


def mk_signals(as_of, dates, prices, positions, sym="SPY"):
    """A minimal signals.json payload with a trailing history."""
    hist = [{"date": d, "price": p, "regime": "Bull", "position": q}
            for d, p, q in zip(dates, prices, positions)]
    return {"tickers": {sym: {
        "label": "S&P 500", "regime": "Bull", "regime_conf": 0.9,
        "price": prices[-1],
        "change_1d": (prices[-1] / prices[-2] - 1) if len(prices) > 1 else 0.0,
        "yhat_5d": 0.001, "position": positions[-1],
        "prev_position": positions[-2] if len(positions) > 1 else 0.0,
        "vol_annual": 0.15, "vol_cap": 1.0, "advice": "HOLD", "advice_reason": "x",
        "as_of": as_of, "history": hist}}}


def test_skipped_trading_day_is_backfilled_and_its_return_is_booked():
    """If the feed jumps past a day, that day must still be logged AND earn its return."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = [100.0, 110.0, 121.0]           # +10% then +10%
    pos = [1.0, 1.0, 1.0]

    genesis = T.build_rows(mk_signals("2026-01-05", dates[:1], prices[:1], pos[:1]), None)
    assert len(genesis) == 1, f"genesis should be one row, got {len(genesis)}"

    # feed skips 01-06 entirely and reports as_of 01-07
    rows = T.build_rows(mk_signals("2026-01-07", dates, prices, pos), genesis[-1])

    assert [r["date"] for r in rows] == ["2026-01-06", "2026-01-07"], \
        f"expected both days, got {[r['date'] for r in rows]}"
    assert rows[0].get("backfilled") is True, "recovered day must be flagged backfilled"
    assert rows[1].get("backfilled") is not True, "live day must not be flagged backfilled"

    # both +10% days must be compounded in
    nav = rows[-1]["paper"]["SPY"]["nav"]
    expected = (1 - T.COST) * 1.10 * 1.10          # genesis entry cost, then two +10% days
    assert abs(nav - expected) < 1e-6, f"NAV {nav:.6f} != {expected:.6f} (a day's return was dropped)"


def test_normal_consecutive_day_appends_one_row():
    dates = ["2026-01-05", "2026-01-06"]
    rows = T.build_rows(
        mk_signals("2026-01-06", dates, [100.0, 110.0], [1.0, 1.0]),
        T.build_rows(mk_signals("2026-01-05", dates[:1], [100.0], [1.0]), None)[-1])
    assert [r["date"] for r in rows] == ["2026-01-06"]
    assert rows[0].get("backfilled") is not True


def test_already_logged_day_produces_nothing():
    g = T.build_rows(mk_signals("2026-01-05", ["2026-01-05"], [100.0], [1.0]), None)
    rows = T.build_rows(mk_signals("2026-01-05", ["2026-01-05"], [100.0], [1.0]), g[-1])
    assert rows == [], f"re-running on a logged date must be a no-op, got {len(rows)} rows"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS  {name}")
            except Exception as e:
                fails += 1; print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILING'}")
    sys.exit(1 if fails else 0)

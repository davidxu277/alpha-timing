"""
append_track_record.py — immutable, forward, tamper-evident live record.

Runs right after update_signals.py in the daily job. It does two things, both
append-only, so the past is never rewritten:

  1. Logs the day's signals (regime / position / advice / price) to
     docs/data/track_record.jsonl — one line per trading day, each line
     hash-chained to the previous one (hash = sha256(prev_hash + line)). Any
     retroactive edit to a past line breaks every hash after it, and the public
     GitHub commit timestamps provide the third-party attestation. This is the
     provenance layer: proof that a call was made on a date and never changed.

  2. Advances a causal paper-trading NAV for each ticker and an equal-weight
     portfolio. The position recorded yesterday earns today's realized return
     (change_1d); rebalancing to today's target pays 5 bps; idle cash earns the
     same 4%/yr as the backtest. No look-ahead: a day's forward return is only
     ever booked on the following run, once it is actually realized.

The record starts (NAV = 1.0) the first time this runs — the clock begins now.
Idempotent: re-running on a date already logged is a no-op.

    python docs/append_track_record.py            # append today (after update_signals)
    python docs/append_track_record.py --verify   # re-walk and validate the whole chain
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from regime_trading import COST, CASH_D  # noqa: E402  single source of truth

SIGNALS = os.path.join(HERE, "data", "signals.json")
LEDGER = os.path.join(HERE, "data", "track_record.jsonl")
SUMMARY = os.path.join(HERE, "data", "track_summary.json")
GENESIS = "0" * 64


# ── hash chain ────────────────────────────────────────────────────────────────
def line_hash(prev_hash: str, core: dict) -> str:
    """sha256 over the previous hash plus a canonical serialization of the line."""
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()


def read_ledger() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def verify(rows: list[dict]) -> tuple[bool, str]:
    """Re-walk the chain: every hash recomputes, index increments, dates are unique."""
    prev = GENESIS
    seen = set()
    for i, row in enumerate(rows):
        core = {k: v for k, v in row.items() if k != "hash"}
        if core.get("prev_hash") != prev:
            return False, f"row {i} ({row.get('date')}): prev_hash mismatch"
        if core.get("index") != i:
            return False, f"row {i} ({row.get('date')}): index should be {i}"
        if line_hash(prev, core) != row.get("hash"):
            return False, f"row {i} ({row.get('date')}): hash mismatch (tampered?)"
        if row.get("date") in seen:
            return False, f"row {i}: duplicate date {row.get('date')}"
        seen.add(row.get("date"))
        prev = row["hash"]
    return True, f"chain OK — {len(rows)} rows, tip {prev[:12]}…" if rows else (True, "empty chain")


# ── append one day ──────────────────────────────────────────────────────────
def ma20_over_60(history: list[dict]) -> float:
    """MA20>60 crossover benchmark: fully invested when the 20-day mean tops the 60-day.

    Decided at today's close from the trailing price history — the third baseline
    from the SPY backtest (System vs Buy&Hold vs MA20>60)."""
    px = [float(p["price"]) for p in history]
    if len(px) < 60:
        return 0.0
    return 1.0 if sum(px[-20:]) / 20.0 > sum(px[-60:]) / 60.0 else 0.0


def build_day(sig: dict, last: dict | None) -> dict:
    """Compose the next ledger line: signals snapshot + causal paper-NAV step."""
    tickers = sig["tickers"]
    date = next(iter(tickers.values()))["as_of"]

    prev_paper = last["paper"] if last else {}
    signals_out, paper_out = {}, {}
    net_rets, ma_net_rets = [], []

    for sym, s in tickers.items():
        ret = float(s["change_1d"])                     # realized over the day just held
        pos_new = float(s["position"])
        ma_new = ma20_over_60(s["history"])             # MA20>60 target, decided at close

        # ── model strategy sleeve ──
        held = prev_paper.get(sym)                      # what we carried into today
        if held is None:                                # genesis: start flat in cash
            nav, pos_held, gross = 1.0, 0.0, 0.0
        else:
            nav, pos_held = float(held["nav"]), float(held["pos"])
            gross = pos_held * ret + (1.0 - pos_held) * CASH_D
            nav *= (1.0 + gross)
        nav *= (1.0 - COST * abs(pos_new - pos_held))   # rebalance to today's target
        net_rets.append(gross - COST * abs(pos_new - pos_held))

        # ── MA20>60 benchmark sleeve (same accounting, mechanical rule) ──
        ma_prev = prev_paper.get(sym, {}).get("ma_pos")
        if ma_prev is None:                             # ma baseline starts flat
            ma_nav, ma_held, ma_gross = 1.0, 0.0, 0.0
        else:
            ma_nav, ma_held = float(prev_paper[sym]["ma_nav"]), float(ma_prev)
            ma_gross = ma_held * ret + (1.0 - ma_held) * CASH_D
            ma_nav *= (1.0 + ma_gross)
        ma_nav *= (1.0 - COST * abs(ma_new - ma_held))
        ma_net_rets.append(ma_gross - COST * abs(ma_new - ma_held))

        signals_out[sym] = {
            "regime": s["regime"], "regime_conf": s["regime_conf"],
            "position": pos_new, "prev_position": float(s["prev_position"]),
            "advice": s["advice"], "price": s["price"],
            "ret_1d": s["change_1d"], "yhat_5d": s["yhat_5d"],
        }
        paper_out[sym] = {"nav": round(nav, 6), "pos": round(pos_new, 3),
                          "ma_nav": round(ma_nav, 6), "ma_pos": round(ma_new, 3)}

    def _port(key, rets):
        base = float(prev_paper.get("portfolio", {}).get(key, 1.0))
        return base * (1.0 + sum(rets) / len(rets)) if rets else 1.0
    paper_out["portfolio"] = {"nav": round(_port("nav", net_rets), 6),
                              "ma_nav": round(_port("ma_nav", ma_net_rets), 6)}

    core = {
        "date": date,
        "index": (last["index"] + 1) if last else 0,
        "prev_hash": last["hash"] if last else GENESIS,
        "signals": signals_out,
        "paper": paper_out,
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    core["hash"] = line_hash(core["prev_hash"], {k: v for k, v in core.items() if k != "hash"})
    return core


def buyhold_navs(rows: list[dict]) -> dict:
    """Equal-weight buy-and-hold benchmark, derived from the chained ret_1d.

    Always fully invested from genesis (no entry cost); a per-ticker NAV plus the
    equal-weight portfolio. Same universe and window as the strategy, so the two
    curves are directly comparable."""
    syms = list(rows[0]["signals"].keys())
    bh = {s: 1.0 for s in syms}
    port = 1.0
    for r in rows[1:]:                                    # skip genesis (no prior holding)
        rets = []
        for s in syms:
            ret = float(r["signals"][s]["ret_1d"])
            bh[s] *= (1.0 + ret)
            rets.append(ret)
        port *= (1.0 + sum(rets) / len(rets))
    out = {s: bh[s] for s in syms}
    out["portfolio"] = port
    return out


def write_summary(rows: list[dict]) -> None:
    tip = rows[-1]
    bh = buyhold_navs(rows)
    paper = {}
    for sym, p in tip["paper"].items():
        strat = (p["nav"] - 1.0) * 100
        bench = (bh.get(sym, 1.0) - 1.0) * 100
        ma = (p.get("ma_nav", 1.0) - 1.0) * 100
        paper[sym] = {"nav": p["nav"], "return_pct": round(strat, 2),
                      "buyhold_return_pct": round(bench, 2),
                      "ma_return_pct": round(ma, 2),
                      "vs_buyhold_pp": round(strat - bench, 2)}
    summary = {
        "since": rows[0]["date"],
        "through": tip["date"],
        "days": len(rows),
        "chain_valid": verify(rows)[0],
        "tip_hash": tip["hash"],
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "paper": paper,
        "note": "Forward paper record — positions logged before the outcome, never edited. "
                "Research only; not investment advice.",
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    if "--verify" in sys.argv:
        ok, msg = verify(read_ledger())
        print(("VERIFY OK — " if ok else "VERIFY FAILED — ") + msg)
        raise SystemExit(0 if ok else 1)

    with open(SIGNALS) as f:
        sig = json.load(f)
    date = next(iter(sig["tickers"].values()))["as_of"]

    rows = read_ledger()
    ok, msg = verify(rows)
    if not ok:
        raise SystemExit(f"refusing to append — existing chain is broken: {msg}")

    if rows and rows[-1]["date"] == date:
        print(f"already logged {date} — no-op (idempotent).")
        return

    row = build_day(sig, rows[-1] if rows else None)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    rows.append(row)
    write_summary(rows)

    port = row["paper"]["portfolio"]["nav"]
    print(f"appended {date} (row {row['index']}, hash {row['hash'][:12]}…) — "
          f"portfolio paper NAV {port:.4f} ({(port - 1) * 100:+.2f}%), {len(rows)} days tracked.")


if __name__ == "__main__":
    main()

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
def build_day(sig: dict, last: dict | None) -> dict:
    """Compose the next ledger line: signals snapshot + causal paper-NAV step."""
    tickers = sig["tickers"]
    date = next(iter(tickers.values()))["as_of"]

    prev_paper = last["paper"] if last else {}
    signals_out, paper_out, net_rets = {}, {}, []

    for sym, s in tickers.items():
        pos_new = float(s["position"])
        held = prev_paper.get(sym)                      # what we carried into today
        if held is None:                                # genesis: start flat in cash
            nav, pos_held, gross = 1.0, 0.0, 0.0
        else:
            nav, pos_held = float(held["nav"]), float(held["pos"])
            ret = float(s["change_1d"])                 # realized over the day just held
            gross = pos_held * ret + (1.0 - pos_held) * CASH_D
            nav *= (1.0 + gross)
        cost = COST * abs(pos_new - pos_held)           # rebalance to today's target
        nav *= (1.0 - cost)
        net_rets.append(gross - cost)

        signals_out[sym] = {
            "regime": s["regime"], "regime_conf": s["regime_conf"],
            "position": pos_new, "prev_position": float(s["prev_position"]),
            "advice": s["advice"], "price": s["price"],
            "ret_1d": s["change_1d"], "yhat_5d": s["yhat_5d"],
        }
        paper_out[sym] = {"nav": round(nav, 6), "pos": round(pos_new, 3)}

    port_prev = prev_paper.get("portfolio", {}).get("nav", 1.0)
    port_nav = float(port_prev) * (1.0 + sum(net_rets) / len(net_rets)) if net_rets else 1.0
    paper_out["portfolio"] = {"nav": round(port_nav, 6)}

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


def write_summary(rows: list[dict]) -> None:
    tip = rows[-1]
    paper = {}
    for sym, p in tip["paper"].items():
        paper[sym] = {"nav": p["nav"], "return_pct": round((p["nav"] - 1.0) * 100, 2)}
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

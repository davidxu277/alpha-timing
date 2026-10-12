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


def _step(payload: dict, last: dict | None, date: str, backfilled: bool) -> dict:
    """One causal ledger step: yesterday's position earns today's realized return."""
    prev_paper = last["paper"] if last else {}
    signals_out, paper_out = {}, {}
    net_rets, ma_net_rets = [], []

    for sym, s in payload.items():
        ret = float(s["ret_1d"])
        pos_new = float(s["position"])
        ma_new = s["ma_pos"] if s.get("ma_pos") is not None else ma20_over_60(s["_hist"])

        # model strategy sleeve
        held = prev_paper.get(sym)
        if held is None:
            nav, pos_held, gross = 1.0, 0.0, 0.0
        else:
            nav, pos_held = float(held["nav"]), float(held["pos"])
            gross = pos_held * ret + (1.0 - pos_held) * CASH_D
            nav *= (1.0 + gross)
        nav *= (1.0 - COST * abs(pos_new - pos_held))
        net_rets.append(gross - COST * abs(pos_new - pos_held))

        # MA20>60 benchmark sleeve
        ma_prev = prev_paper.get(sym, {}).get("ma_pos")
        if ma_prev is None:
            ma_nav, ma_held, ma_gross = 1.0, 0.0, 0.0
        else:
            ma_nav, ma_held = float(prev_paper[sym]["ma_nav"]), float(ma_prev)
            ma_gross = ma_held * ret + (1.0 - ma_held) * CASH_D
            ma_nav *= (1.0 + ma_gross)
        ma_nav *= (1.0 - COST * abs(ma_new - ma_held))
        ma_net_rets.append(ma_gross - COST * abs(ma_new - ma_held))

        signals_out[sym] = {k: s[k] for k in
                            ("regime", "regime_conf", "position", "prev_position",
                             "advice", "price", "ret_1d", "yhat_5d")}
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
    if backfilled:
        core["backfilled"] = True
    core["hash"] = line_hash(core["prev_hash"], {k: v for k, v in core.items() if k != "hash"})
    return core


def build_rows(sig: dict, last: dict | None) -> list[dict]:
    """Every trading day after the ledger tip, not just today — the self-healing part.

    The feed sometimes reports a stale as_of (Yahoo lag) or a CI run is missed, so
    as_of can jump past a trading day. Those days still exist in each ticker's
    trailing history, so we replay them in order and mark them `backfilled` (their
    positions are causal but were not publicly committed on the day)."""
    tickers = sig["tickers"]
    as_of = next(iter(tickers.values()))["as_of"]
    last_date = last["date"] if last else None

    idx = {s: {p["date"]: i for i, p in enumerate(v["history"])} for s, v in tickers.items()}
    if last_date is None:
        dates = [as_of]
    else:
        dates = sorted({d for m in idx.values() for d in m if last_date < d <= as_of})
        if as_of > last_date and as_of not in dates:
            dates.append(as_of)

    rows, prev = [], last
    for d in dates:
        live = (d == as_of)
        payload = {}
        for sym, v in tickers.items():
            h, i = v["history"], idx[sym].get(d)
            if live:
                payload[sym] = {
                    "regime": v["regime"], "regime_conf": v["regime_conf"],
                    "position": float(v["position"]), "prev_position": float(v["prev_position"]),
                    "advice": v["advice"], "price": v["price"],
                    "ret_1d": v["change_1d"], "yhat_5d": v["yhat_5d"], "_hist": h}
            elif i is not None:
                ret = (h[i]["price"] / h[i - 1]["price"] - 1) if i > 0 else 0.0
                payload[sym] = {
                    "regime": h[i]["regime"], "regime_conf": None,
                    "position": float(h[i]["position"]),
                    "prev_position": float(h[i - 1]["position"]) if i > 0 else 0.0,
                    "advice": None, "price": round(float(h[i]["price"]), 2),
                    "ret_1d": round(ret, 4), "yhat_5d": None, "_hist": h[:i + 1]}
            else:                                   # no quote that day — hold, no move
                pp = float(prev["paper"][sym]["pos"]) if prev and sym in prev.get("paper", {}) else 0.0
                payload[sym] = {"regime": None, "regime_conf": None, "position": pp,
                                "prev_position": pp, "advice": None, "price": None,
                                "ret_1d": 0.0, "yhat_5d": None, "_hist": h}
        rows.append(_step(payload, prev, d, backfilled=not live))
        prev = rows[-1]
    return rows


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


def gaps(rows: list[dict]) -> list[str]:
    """Trading days present in the tickers' history but missing from the ledger."""
    if not rows:
        return []
    try:
        with open(SIGNALS) as f:
            sig = json.load(f)
    except OSError:
        return []
    have = {r["date"] for r in rows}
    lo, hi = rows[0]["date"], rows[-1]["date"]
    allh = {p["date"] for v in sig["tickers"].values() for p in v["history"]}
    return sorted(d for d in allh if lo < d < hi and d not in have)


def repair() -> None:
    """Re-insert trading days that were skipped, recomputing NAV and hashes forward.

    Original rows keep their own forward-logged signal payload; only the recovered
    days are reconstructed from history, and they are flagged `backfilled`."""
    rows = read_ledger()
    missing = gaps(rows)
    if not missing:
        print("no internal gaps — nothing to repair.")
        return
    with open(SIGNALS) as f:
        sig = json.load(f)
    tickers = sig["tickers"]
    idx = {s: {p["date"]: i for i, p in enumerate(v["history"])} for s, v in tickers.items()}
    by_date = {r["date"]: r for r in rows}

    out, prev = [], None
    for d in sorted(set(by_date) | set(missing)):
        if d in by_date:                                  # keep the original payload
            old = by_date[d]
            payload = {}
            for sym, sg in old["signals"].items():
                payload[sym] = dict(sg)
                payload[sym]["ma_pos"] = old["paper"][sym].get("ma_pos")
                payload[sym]["_hist"] = tickers[sym]["history"]
            row = _step(payload, prev, d, backfilled=bool(old.get("backfilled")))
        else:                                             # reconstruct from history
            payload = {}
            for sym, v in tickers.items():
                h, i = v["history"], idx[sym].get(d)
                if i is None:
                    pp = float(prev["paper"][sym]["pos"]) if prev else 0.0
                    payload[sym] = {"regime": None, "regime_conf": None, "position": pp,
                                    "prev_position": pp, "advice": None, "price": None,
                                    "ret_1d": 0.0, "yhat_5d": None, "ma_pos": None, "_hist": h}
                else:
                    ret = (h[i]["price"] / h[i - 1]["price"] - 1) if i > 0 else 0.0
                    payload[sym] = {
                        "regime": h[i]["regime"], "regime_conf": None,
                        "position": float(h[i]["position"]),
                        "prev_position": float(h[i - 1]["position"]) if i > 0 else 0.0,
                        "advice": None, "price": round(float(h[i]["price"]), 2),
                        "ret_1d": round(ret, 4), "yhat_5d": None,
                        "ma_pos": None, "_hist": h[:i + 1]}
            row = _step(payload, prev, d, backfilled=True)
        out.append(row); prev = row

    with open(LEDGER, "w") as f:
        for r in out:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    write_summary(out)
    print(f"repaired: inserted {len(missing)} missing day(s) {missing}")
    print(f"ledger now {len(out)} rows, {out[0]['date']} -> {out[-1]['date']}")


def main() -> None:
    if "--verify" in sys.argv:
        rows = read_ledger()
        ok, msg = verify(rows)
        g = gaps(rows)
        print(("VERIFY OK — " if ok else "VERIFY FAILED — ") + msg)
        if g:
            print(f"WARNING: {len(g)} missing trading day(s): {g}")
        raise SystemExit(0 if ok and not g else 1)
    if "--repair" in sys.argv:
        repair()
        return

    with open(SIGNALS) as f:
        sig = json.load(f)
    date = next(iter(sig["tickers"].values()))["as_of"]

    rows = read_ledger()
    ok, msg = verify(rows)
    if not ok:
        raise SystemExit(f"refusing to append — existing chain is broken: {msg}")

    new_rows = build_rows(sig, rows[-1] if rows else None)
    if not new_rows:
        print(f"already logged {date} — no-op (idempotent).")
        return

    with open(LEDGER, "a") as f:
        for r in new_rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    rows.extend(new_rows)
    write_summary(rows)

    recovered = [r["date"] for r in new_rows if r.get("backfilled")]
    port = rows[-1]["paper"]["portfolio"]["nav"]
    print(f"appended {len(new_rows)} row(s): {[r['date'] for r in new_rows]}")
    if recovered:
        print(f"  self-healed {len(recovered)} previously-missed day(s): {recovered}")
    print(f"portfolio paper NAV {port:.4f} ({(port - 1) * 100:+.2f}%), {len(rows)} days tracked.")
    g = gaps(rows)
    if g:
        print(f"WARNING: still missing {len(g)} day(s): {g} — run --repair")


if __name__ == "__main__":
    main()

"""M1 data-flywheel runner for the Jackpot Carryover Harvester.

Two modes, meant to be driven by a scheduler (cron):

  capture [date] [venue]   poll the current/given meeting's jackpot pools + win
                           odds once; append to the capture stores. Run repeatedly
                           across the betting window (e.g. every few minutes).
  settle  <date> <venue>   after the meeting, join the captured pre-close snapshots
                           to the declared dividends -> append carryover ledger rows.

Empty date/venue let the server pick the current meeting. Stores live under
data_cache/ (gitignored). Values populate only while a meeting's pools are open.
"""
from __future__ import annotations

import sys

import polars as pl

from hkjc.data.carryover_ledger import (
    CarryoverLedger,
    build_carryover_rows,
    settled_jackpot_result,
)
from hkjc.data.historical import get_race_dividends
from hkjc.data.jackpot_ticks import JackpotCaptureStore
from hkjc.data.odds_ticks import OddsTickStore

JSTORE = JackpotCaptureStore("data_cache/jackpot_ticks")
OSTORE = OddsTickStore("data_cache/odds_ticks")
LEDGER = CarryoverLedger("data_cache/carryover_ledger.parquet")


def _leg_races(races_csv: str) -> list[int]:
    return sorted({int(x) for x in str(races_csv).split(",") if x.strip()})


def capture(date: str = "", venue: str = "") -> None:
    snap = JSTORE.capture(date, venue)
    if snap.height == 0:
        print("No jackpot pools open (future meeting / pools not yet open).")
        return
    d, v = snap["date"][0], snap["venue"][0]
    races = sorted({r for s in snap["races"].to_list() for r in _leg_races(s)})
    for rn in races:
        try:
            OSTORE.capture(d, v, rn, pools=("WIN",))
        except Exception as e:  # pragma: no cover - network
            print(f"  win-odds capture failed for race {rn}: {e}")
    print(f"Captured {snap.height} jackpot legs + win odds for races {races} @ {d} {v}")
    print(snap.select(["pool", "leg", "races", "jackpot", "investment", "estimated_dividend"]))


def settle(date: str, venue: str) -> None:
    cap = JSTORE.read(date=date, venue=venue)
    if cap.height == 0:
        print(f"No captures for {date} {venue}.")
        return
    settled = {}
    for pool in cap["pool"].unique().to_list():
        races_csv = cap.filter(pl.col("pool") == pool)["races"][0]
        final_race = _leg_races(races_csv)[-1]
        try:
            divs = get_race_dividends(date.replace("-", "/"), venue, final_race)
        except Exception:
            divs = pl.DataFrame()
        settled[pool] = settled_jackpot_result(pool, divs)
    rows = build_carryover_rows(cap, settled)
    total = LEDGER.append(rows)
    print(f"Appended {rows.height} ledger rows ({total.height} total).")
    print(rows.select(["pool", "carry_in", "gross_pool", "eff_takeout", "agg_edge", "won", "settled_dividend"]))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    if mode == "capture":
        capture(*sys.argv[2:])
    elif mode == "settle":
        if len(sys.argv) < 4:
            sys.exit("usage: settle <date YYYY-MM-DD> <venue ST|HV>")
        settle(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"unknown mode {mode!r} (use 'capture' or 'settle')")

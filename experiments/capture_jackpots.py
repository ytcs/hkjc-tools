"""M1 data-flywheel runner for the Jackpot Carryover Harvester.

Timezone discipline (HK racing is UTC+8, no DST):
- `captured_at` is stored in **UTC**; the partition `date` is the **server's HK
  meeting date** — both already correct in the stores, independent of host clock.
- This runner decides *when to poll* in **HK time** (computed internally via
  zoneinfo), so it is correct no matter what timezone the host machine is in.

Modes (drive with a scheduler):
  capture [date] [venue]   poll once; persist only if pools are actually open.
  loop [interval_s]        long-running self-gating poller (recommended). Polls
                           every interval during HK meeting hours, sleeps otherwise.
                           Run it under tmux/systemd; survives across meetings.
  settle <date> <venue>    after a meeting, join declared dividends -> ledger rows.

Deploy options:
  - Simplest: `python experiments/capture_jackpots.py loop` in tmux/systemd.
  - OS cron (UTC; self-gating covers holidays): poll across the HK windows, e.g.
      */5 3-16 * * * cd <repo> && uv run python experiments/capture_jackpots.py capture
    (HK day meetings ~05:00-10:00 UTC; HK night ~10:45-15:30 UTC.)
  - Claude CronCreate: session-bound, *local* tz, 7-day expiry -> only for ad-hoc
    "capture tonight" while a session is live, not the season-long pipeline.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl

from hkjc.data.carryover_ledger import (
    CarryoverLedger,
    build_carryover_rows,
    settled_jackpot_result,
)
from hkjc.data.historical import get_race_dividends
from hkjc.data.jackpot_ticks import JackpotCaptureStore
from hkjc.data.odds_ticks import OddsTickStore

HK = ZoneInfo("Asia/Hong_Kong")
JSTORE = JackpotCaptureStore("data_cache/jackpot_ticks")
OSTORE = OddsTickStore("data_cache/odds_ticks")
LEDGER = CarryoverLedger("data_cache/carryover_ledger.parquet")


def hk_now() -> datetime:
    return datetime.now(HK)


def in_capture_window(now_hk: datetime | None = None) -> bool:
    """Generous HK-time window when betting pools may be open.

    Deliberately broad (covers Wed night, Sat/Sun day, and holiday/midweek cards):
    HK 10:00–23:59. Off-window we skip the network entirely; in-window the
    self-gating capture still no-ops cheaply when no pools are open.
    """
    now_hk = now_hk or hk_now()
    return 10 <= now_hk.hour <= 23


def _leg_races(races_csv: str) -> list[int]:
    return sorted({int(x) for x in str(races_csv).split(",") if x.strip()})


def _pools_open(snapshot: pl.DataFrame) -> bool:
    """True only if at least one jackpot leg has a live (non-null) investment."""
    if snapshot is None or snapshot.height == 0:
        return False
    inv = snapshot["investment"]
    return bool((inv.is_not_null() & inv.is_not_nan()).any())


def capture(date: str = "", venue: str = "") -> bool:
    """Poll once; persist only when pools are genuinely open. Returns True if stored."""
    from hkjc.data.live import live_jackpots
    snap = live_jackpots(date, venue, jackpot_only=True)
    if not _pools_open(snap):
        print(f"[{hk_now():%Y-%m-%d %H:%M} HKT] no open jackpot pools — skipping.")
        return False
    captured_at = datetime.now(timezone.utc)
    JSTORE.record(captured_at, snap)
    d, v = snap["date"][0], snap["venue"][0]
    races = sorted({r for s in snap["races"].to_list() for r in _leg_races(s)})
    for rn in races:
        try:
            OSTORE.capture(d, v, rn, pools=("WIN",))
        except Exception as e:  # pragma: no cover - network
            print(f"  win-odds capture failed for race {rn}: {e}")
    print(f"[{hk_now():%Y-%m-%d %H:%M} HKT] captured {snap.height} legs + odds for "
          f"races {races} @ {d} {v}")
    return True


def loop(interval_s: int = 300) -> None:
    """Self-gating long-running poller. Polls in-window, sleeps out-of-window."""
    print(f"jackpot capture loop started (interval {interval_s}s); HK now {hk_now():%Y-%m-%d %H:%M}")
    while True:
        try:
            if in_capture_window():
                capture()
                time.sleep(interval_s)
            else:
                print(f"[{hk_now():%H:%M} HKT] outside capture window — sleeping 30m.")
                time.sleep(1800)
        except KeyboardInterrupt:
            print("loop stopped.")
            return
        except Exception as e:  # pragma: no cover - keep the loop alive
            print(f"  poll error: {e}; backing off 60s.")
            time.sleep(60)


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
    elif mode == "loop":
        loop(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif mode == "settle":
        if len(sys.argv) < 4:
            sys.exit("usage: settle <date YYYY-MM-DD> <venue ST|HV>")
        settle(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"unknown mode {mode!r} (use capture / loop / settle)")

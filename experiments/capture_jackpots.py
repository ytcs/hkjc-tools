"""M1 data-flywheel runner for the Jackpot Carryover Harvester (local).

Capture/settle logic lives in :mod:`hkjc.data.capture` (shared with the cloud
deployment in ``cloud/modal_app.py``); this script adds local store paths and the
HK-time-gated polling loop.

Timezone discipline (HK racing is UTC+8, no DST):
- `captured_at` is stored in UTC; the partition `date` is the server's HK meeting
  date — both already correct in the stores, independent of host clock.
- This runner decides *when to poll* in HK time (zoneinfo), correct on any host.

Modes:
  capture [date] [venue]   poll once; persist only if pools are open.
  loop [interval_s]        self-gating poller (poll in HK meeting hours, else sleep).
  settle  <date> <venue>   join declared dividends -> carryover ledger rows.

If your machine is not on 24/7, deploy the cloud version instead — see cloud/.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from hkjc.data.capture import capture_once, settle_meeting
from hkjc.data.carryover_ledger import CarryoverLedger
from hkjc.data.jackpot_ticks import JackpotCaptureStore
from hkjc.data.odds_ticks import OddsTickStore

HK = ZoneInfo("Asia/Hong_Kong")
JSTORE = JackpotCaptureStore("data_cache/jackpot_ticks")
OSTORE = OddsTickStore("data_cache/odds_ticks")
LEDGER = CarryoverLedger("data_cache/carryover_ledger.parquet")


def hk_now() -> datetime:
    return datetime.now(HK)


def in_capture_window(now_hk: datetime | None = None) -> bool:
    """Generous HK-time window when betting pools may be open (HK 10:00–23:59)."""
    now_hk = now_hk or hk_now()
    return 10 <= now_hk.hour <= 23


def capture(date: str = "", venue: str = "") -> bool:
    stored = capture_once(JSTORE, OSTORE, date, venue)
    print(f"[{hk_now():%Y-%m-%d %H:%M} HKT] "
          + ("captured jackpot snapshot + win odds." if stored else "no open jackpot pools — skipping."))
    return stored


def loop(interval_s: int = 300) -> None:
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
    rows = settle_meeting(JSTORE, LEDGER, date, venue)
    if rows.height == 0:
        print(f"No captures for {date} {venue}.")
        return
    print(f"Settled {rows.height} legs into the ledger.")
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

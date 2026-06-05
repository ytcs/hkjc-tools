"""Shared capture/settle operations for the carryover data flywheel.

The actual logic for one capture poll and for settling a meeting lives here so the
local runner (``experiments/capture_jackpots.py``) and any cloud deployment
(``cloud/modal_app.py``) share one implementation. Stores are passed in, so the
caller decides where data lives (local ``data_cache/`` or a cloud volume).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import polars as pl

from .carryover_ledger import (
    CarryoverLedger,
    build_carryover_rows,
    settled_jackpot_result,
)
from .historical import get_race_dividends
from .ticks import JackpotCaptureStore, OddsTickStore
from .live import live_jackpots


def _leg_races(races_csv: str) -> list[int]:
    return sorted({int(x) for x in str(races_csv).split(",") if x.strip()})


def pools_open(snapshot: pl.DataFrame) -> bool:
    """True only if at least one jackpot leg has a live (non-null) investment."""
    if snapshot is None or snapshot.height == 0:
        return False
    inv = snapshot["investment"]
    return bool((inv.is_not_null() & inv.is_not_nan()).any())


def capture_once(
    jackpot_store: JackpotCaptureStore,
    odds_store: Optional[OddsTickStore] = None,
    date: str = "",
    venue: str = "",
    now: Optional[datetime] = None,
) -> bool:
    """Poll once; persist a jackpot snapshot (+ win odds for its legs) iff pools open.

    Returns True if a snapshot was stored. Self-gating: when no jackpot pools are
    open (future meeting / pools not yet open), nothing is written.
    """
    snap = live_jackpots(date, venue, jackpot_only=True)
    if not pools_open(snap):
        return False
    captured_at = now or datetime.now(timezone.utc)
    jackpot_store.record(captured_at, snap)
    if odds_store is not None:
        d, v = snap["date"][0], snap["venue"][0]
        races = sorted({r for s in snap["races"].to_list() for r in _leg_races(s)})
        for rn in races:
            try:
                odds_store.capture(d, v, rn, pools=("WIN",))
            except Exception:  # pragma: no cover - network
                pass
    return True


def settle_meeting(
    jackpot_store: JackpotCaptureStore,
    ledger: CarryoverLedger,
    date: str,
    venue: str,
) -> pl.DataFrame:
    """Join a captured meeting's pre-close snapshots to declared dividends → ledger.

    Idempotent (the ledger upserts), and safe to run before results are final: if a
    pool's dividend isn't available yet, ``won`` stays null until a later settle.
    """
    cap = jackpot_store.read(date=date, venue=venue)
    if cap.height == 0:
        return pl.DataFrame()
    settled = {}
    for pool in cap["pool"].unique().to_list():
        races_csv = cap.filter(pl.col("pool") == pool)["races"][0]
        final_race = _leg_races(races_csv)[-1]
        try:
            divs = get_race_dividends(date.replace("-", "/"), venue, final_race)
        except Exception:  # pragma: no cover - network / not yet declared
            divs = pl.DataFrame()
        settled[pool] = settled_jackpot_result(pool, divs)
    rows = build_carryover_rows(cap, settled)
    ledger.append(rows)
    return rows


def settle_recent(jackpot_store: JackpotCaptureStore, ledger: CarryoverLedger) -> int:
    """Settle every captured meeting (idempotent). Returns number of meetings settled."""
    cap = jackpot_store.read()
    if cap.height == 0:
        return 0
    meetings = cap.select(["date", "venue"]).unique()
    for date, venue in meetings.iter_rows():
        settle_meeting(jackpot_store, ledger, date, venue)
    return meetings.height

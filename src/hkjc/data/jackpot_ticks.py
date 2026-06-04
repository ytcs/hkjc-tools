"""Jackpot pool capture — the carryover strategy's forward data flywheel (B1).

Result pages carry no carryover/pool-size data, so we accumulate it prospectively:
poll :func:`hkjc.data.live.live_jackpots` across a meeting's betting window and
persist every snapshot. The *trajectory* (how the jackpot ``C``, gross pool ``G``
and ``estimated_dividend`` evolve toward close) is what later lets us measure
late-money dilution — the crux research question (see CARRYOVER_RESEARCH_PLAN.md).

Unlike :class:`~hkjc.data.odds_ticks.OddsTickStore`, snapshots are **not** deduped:
the whole point is the time series. Append-only Parquet, one file per poll::

    {root}/date={YYYY-MM-DD}/venue={ST|HV}/{epoch_us}.parquet
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import polars as pl

from .live import live_jackpots

JACKPOT_TICK_SCHEMA = {
    "captured_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "date": pl.Utf8,
    "venue": pl.Utf8,
    "pool": pl.Utf8,
    "leg": pl.Int64,
    "races": pl.Utf8,
    "investment": pl.Float64,        # G: today's gross pool
    "jackpot": pl.Float64,           # C: carried-in jackpot
    "estimated_dividend": pl.Float64,
    "merged_pool_id": pl.Utf8,
}


class JackpotCaptureStore:
    """Append-only Parquet store for the live jackpot-pool time series."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def capture(self, date: str = "", venue: str = "", now: Optional[datetime] = None) -> pl.DataFrame:
        """Poll a meeting's jackpot pools once and persist the snapshot.

        Empty strings let the server pick the current meeting. Returns the snapshot
        (empty if no jackpot pools are open). Meant to be run repeatedly across the
        betting window by an orchestrator/cron.
        """
        snap = live_jackpots(date, venue, jackpot_only=True)
        captured_at = now or datetime.now(timezone.utc)
        return self.record(captured_at, snap)

    def record(self, captured_at: datetime, snapshot: pl.DataFrame) -> pl.DataFrame:
        """Persist an already-fetched ``live_jackpots`` snapshot (network-free)."""
        if snapshot is None or snapshot.height == 0:
            return self._empty()
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        df = snapshot.with_columns(
            pl.lit(captured_at).cast(JACKPOT_TICK_SCHEMA["captured_at"]).alias("captured_at")
        )
        df = df.select(list(JACKPOT_TICK_SCHEMA.keys())).cast(JACKPOT_TICK_SCHEMA)
        date = df["date"][0]
        venue = df["venue"][0]
        self._write(df, date, venue, captured_at)
        return df

    def read(self, date: Optional[str] = None, venue: Optional[str] = None,
             pool: Optional[str] = None) -> pl.DataFrame:
        pd_ = f"date={date}" if date else "date=*"
        pv = f"venue={venue}" if venue else "venue=*"
        files = sorted(self.root.glob(f"{pd_}/{pv}/*.parquet"))
        if not files:
            return self._empty()
        df = pl.read_parquet(files)
        if pool is not None:
            df = df.filter(pl.col("pool") == pool)
        return df.sort("captured_at")

    def latest(self, date: Optional[str] = None, venue: Optional[str] = None) -> pl.DataFrame:
        """The most recent snapshot per (date, venue, pool, leg) — the close-proxy."""
        df = self.read(date, venue)
        if df.height == 0:
            return df
        return (df.sort("captured_at")
                .group_by(["date", "venue", "pool", "leg"], maintain_order=True)
                .last())

    def _write(self, df: pl.DataFrame, date: str, venue: str, captured_at: datetime) -> Path:
        d = self.root / f"date={date}" / f"venue={venue}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{int(captured_at.timestamp() * 1_000_000)}.parquet"
        df.write_parquet(path)
        return path

    def _empty(self) -> pl.DataFrame:
        return pl.DataFrame(schema=JACKPOT_TICK_SCHEMA)

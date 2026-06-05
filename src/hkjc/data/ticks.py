"""Time-series tick stores — odds and jackpot pools.

Both the live-odds and jackpot-pool feeds are turned into durable time series the same
way: every poll is normalized to tidy rows and written as an immutable Parquet file
under a meeting partition (append-only, concurrency-safe — each poll is a fresh file).
The two stores share that mechanic (:func:`_write_parquet`, :func:`_empty`) but differ
in partition granularity and schema, so they stay as two classes in one module.

Separation of concerns: this module **only acquires and persists**. It does not compute
edge or evaluate bets — closing-line semantics and CLV live in :mod:`hkjc.betting.clv`,
which reads from here. The GraphQL transport lives in :mod:`hkjc.data.live`.

Layouts::

    odds:     {root}/date={YYYY-MM-DD}/venue={ST|HV}/race={n}/{epoch_us}.parquet
    jackpot:  {root}/date={YYYY-MM-DD}/venue={ST|HV}/{epoch_us}.parquet
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import polars as pl

from .live import _fetch_live_odds_uncached, _fetch_live_races, live_jackpots

DEFAULT_POOLS: tuple[str, ...] = ("WIN", "PLA", "QIN", "QPL")

TICK_SCHEMA = {
    "captured_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "date": pl.Utf8,
    "venue": pl.Utf8,
    "race": pl.Int64,
    "pool": pl.Utf8,
    "selection": pl.Utf8,
    "odds": pl.Float64,
    "last_update": pl.Utf8,           # server-side pool lastUpdateTime (dedup key)
}

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


# ----------------------------------------------------------------- shared helpers
def _ensure_utc(captured_at: datetime) -> datetime:
    return captured_at.replace(tzinfo=timezone.utc) if captured_at.tzinfo is None else captured_at


def _write_parquet(df: pl.DataFrame, partition_dir: Path, captured_at: datetime) -> Path:
    """Write one immutable poll file named by capture epoch-microseconds."""
    partition_dir.mkdir(parents=True, exist_ok=True)
    path = partition_dir / f"{int(captured_at.timestamp() * 1_000_000)}.parquet"
    df.write_parquet(path)
    return path


def _empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


# ----------------------------------------------------------------- odds tick store
class OddsTickStore:
    """Append-only Parquet store for the live-odds time series.

    Parameters
    ----------
    root:
        Directory under which meeting partitions are written. Created on demand.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    # ----------------------------------------------------------------- capture
    def capture(
        self,
        date: str = "",
        venue: str = "",
        race: int = 1,
        pools: Sequence[str] = DEFAULT_POOLS,
        now: Optional[datetime] = None,
        dedupe: bool = True,
    ) -> pl.DataFrame:
        """Poll one race's odds and persist them.

        Empty ``date``/``venue`` let the server pick the current meeting; the
        returned/persisted ``date``/``venue`` are the server's reconciled values.
        Returns the captured snapshot (whether or not it was written — see
        ``dedupe``).
        """
        info = _fetch_live_races(date, venue)
        rdate, rvenue = info["Date"], info["Venue"]
        rows = _fetch_live_odds_uncached(rdate, rvenue, race, odds_type=tuple(pools))
        captured_at = now or datetime.now(timezone.utc)
        return self.record(captured_at, rdate, rvenue, race, rows, dedupe=dedupe)

    def capture_meeting(
        self,
        date: str = "",
        venue: str = "",
        pools: Sequence[str] = DEFAULT_POOLS,
        delay: float = 0.15,
        now: Optional[datetime] = None,
        dedupe: bool = True,
    ) -> pl.DataFrame:
        """Poll every race of a meeting once, politely.

        ``delay`` is a courtesy pause between per-race requests (scraping
        discipline). The *cadence* of repeated meeting captures is the caller's
        concern (an orchestrator / cron), not the store's.
        """
        info = _fetch_live_races(date, venue)
        rdate, rvenue = info["Date"], info["Venue"]
        frames: List[pl.DataFrame] = []
        for i, race in enumerate(sorted(info["Races"])):
            captured_at = now or datetime.now(timezone.utc)
            rows = _fetch_live_odds_uncached(rdate, rvenue, race, odds_type=tuple(pools))
            frames.append(self.record(captured_at, rdate, rvenue, race, rows, dedupe=dedupe))
            if delay and i < len(info["Races"]) - 1:
                time.sleep(delay)
        return pl.concat(frames) if frames else _empty(TICK_SCHEMA)

    def record(
        self,
        captured_at: datetime,
        date: str,
        venue: str,
        race: int,
        rows: Iterable[dict],
        dedupe: bool = True,
    ) -> pl.DataFrame:
        """Persist already-fetched odds rows (network-free).

        ``rows`` are dicts as returned by
        :func:`hkjc.data.live._fetch_live_odds_uncached`
        (``HorseID``/``Type``/``Odds``/``LastUpdate``). When ``dedupe`` is set and
        the market is unchanged from the latest stored snapshot for this race,
        nothing is written. Returns the snapshot frame regardless.
        """
        df = self._rows_to_frame(captured_at, date, venue, race, rows)
        if df.is_empty():
            return df
        if dedupe:
            prev = self._latest_frame(date, venue, race)
            if prev is not None and self._same_market(prev, df):
                return df
        _write_parquet(df, self._partition_dir(date, venue, race), _ensure_utc(captured_at))
        return df

    # -------------------------------------------------------------------- read
    def read(
        self,
        date: Optional[str] = None,
        venue: Optional[str] = None,
        race: Optional[int] = None,
        pool: Optional[Union[str, Sequence[str]]] = None,
    ) -> pl.DataFrame:
        """Read stored ticks, optionally filtered by partition and pool.

        Returns a tidy frame (``TICK_SCHEMA``) sorted by ``captured_at``; empty
        with the correct schema when nothing matches.
        """
        pd_ = f"date={date}" if date else "date=*"
        pv = f"venue={venue}" if venue else "venue=*"
        pr = f"race={race}" if race is not None else "race=*"
        files = sorted(self.root.glob(f"{pd_}/{pv}/{pr}/*.parquet"))
        if not files:
            return _empty(TICK_SCHEMA)
        df = pl.read_parquet(files)
        if pool is not None:
            pools = [pool] if isinstance(pool, str) else list(pool)
            df = df.filter(pl.col("pool").is_in(pools))
        return df.sort("captured_at")

    # ---------------------------------------------------------------- internal
    def _partition_dir(self, date: str, venue: str, race: int) -> Path:
        return self.root / f"date={date}" / f"venue={venue}" / f"race={int(race)}"

    def _rows_to_frame(
        self, captured_at: datetime, date: str, venue: str, race: int, rows: Iterable[dict]
    ) -> pl.DataFrame:
        clean = [r for r in rows if r.get("Odds") is not None]
        n = len(clean)
        captured_at = _ensure_utc(captured_at)
        return pl.DataFrame(
            {
                "captured_at": [captured_at] * n,
                "date": [date] * n,
                "venue": [venue] * n,
                "race": [int(race)] * n,
                "pool": [r["Type"] for r in clean],
                "selection": [str(r["HorseID"]) for r in clean],
                "odds": [float(r["Odds"]) for r in clean],
                "last_update": [r.get("LastUpdate") for r in clean],
            },
            schema=TICK_SCHEMA,
        )

    def _latest_frame(self, date: str, venue: str, race: int) -> Optional[pl.DataFrame]:
        d = self._partition_dir(date, venue, race)
        if not d.exists():
            return None
        files = sorted(d.glob("*.parquet"), key=lambda p: int(p.stem))
        if not files:
            return None
        return pl.read_parquet(files[-1])

    @staticmethod
    def _same_market(a: pl.DataFrame, b: pl.DataFrame) -> bool:
        ka = set(zip(a["pool"].to_list(), a["selection"].to_list(), a["odds"].round(3).to_list()))
        kb = set(zip(b["pool"].to_list(), b["selection"].to_list(), b["odds"].round(3).to_list()))
        return ka == kb


# -------------------------------------------------------------- jackpot tick store
class JackpotCaptureStore:
    """Append-only Parquet store for the live jackpot-pool time series.

    Unlike :class:`OddsTickStore`, snapshots are **not** deduped: the whole point is
    the trajectory of the jackpot ``C``, gross pool ``G`` and estimated dividend toward
    close (the late-money-dilution signal the carryover strategy is built on).
    """

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
            return _empty(JACKPOT_TICK_SCHEMA)
        captured_at = _ensure_utc(captured_at)
        df = snapshot.with_columns(
            pl.lit(captured_at).cast(JACKPOT_TICK_SCHEMA["captured_at"]).alias("captured_at")
        )
        df = df.select(list(JACKPOT_TICK_SCHEMA.keys())).cast(JACKPOT_TICK_SCHEMA)
        _write_parquet(df, self.root / f"date={df['date'][0]}" / f"venue={df['venue'][0]}", captured_at)
        return df

    def read(self, date: Optional[str] = None, venue: Optional[str] = None,
             pool: Optional[str] = None) -> pl.DataFrame:
        pd_ = f"date={date}" if date else "date=*"
        pv = f"venue={venue}" if venue else "venue=*"
        files = sorted(self.root.glob(f"{pd_}/{pv}/*.parquet"))
        if not files:
            return _empty(JACKPOT_TICK_SCHEMA)
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

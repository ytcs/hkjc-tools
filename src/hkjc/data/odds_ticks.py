"""Odds time-series capture and persistence.

The tick store turns the live odds endpoint into a durable time series: every
poll is normalized to tidy rows and written as an immutable Parquet file under a
meeting/venue/race partition. This is the densest feedback signal the research
loop has — it underpins closing-line value (see :mod:`hkjc.eval.clv`),
odds-drift / steam detection, and any timing study.

Separation of concerns: this module **only acquires and persists**. It does not
compute edge or evaluate bets — closing-line semantics and CLV live in
:mod:`hkjc.eval.clv`, which reads from here. The GraphQL transport lives in
:mod:`hkjc.data.live`; the store calls the *uncached* fetcher so it records genuine
market readings rather than cache-repeated snapshots.

Storage layout (append-only, concurrency-safe — each poll is a fresh file)::

    {root}/date={YYYY-MM-DD}/venue={ST|HV}/race={n}/{epoch_us}.parquet

Each row is one (pool, selection) odds reading at one poll:

    captured_at  pl.Datetime(UTC)  our poll time
    date         str               meeting date, reconciled with the server
    venue        str               ST | HV
    race         int               race number
    pool         str               WIN | PLA | QIN | QPL | ...
    selection    str               combString: "3" (win/place) or "3,5" (quinella)
    odds         float             decimal odds
    last_update  str               server-side pool lastUpdateTime (dedup key)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import polars as pl

from .live import _fetch_live_races, _fetch_live_odds_uncached

DEFAULT_POOLS: tuple[str, ...] = ("WIN", "PLA", "QIN", "QPL")

TICK_SCHEMA = {
    "captured_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "date": pl.Utf8,
    "venue": pl.Utf8,
    "race": pl.Int64,
    "pool": pl.Utf8,
    "selection": pl.Utf8,
    "odds": pl.Float64,
    "last_update": pl.Utf8,
}


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
        return pl.concat(frames) if frames else self._empty()

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
        self._write(df, date, venue, race, captured_at)
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
            return self._empty()
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
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
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

    def _write(self, df: pl.DataFrame, date: str, venue: str, race: int, captured_at: datetime) -> Path:
        d = self._partition_dir(date, venue, race)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{int(captured_at.timestamp() * 1_000_000)}.parquet"
        df.write_parquet(path)
        return path

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

    def _empty(self) -> pl.DataFrame:
        return pl.DataFrame(schema=TICK_SCHEMA)

"""Operational log — high-volume paper-trade fills and live-race events (parquet).

The markdown :class:`~hkjc.agents.journal.store.Journal` is for durable narrative
(findings, decisions, handoffs). The trading agent also produces a *lot* of low-narrative
operational rows — intended paper-trade fills, per-poll odds snapshots, decisions — that
would drown the journal. Those go here: an append-only parquet store partitioned by date,
cheap to write every tick and to query later. Only *notable* ops events get promoted to a
markdown ``observation`` entry for another agent.

Paper-trade discipline: this records *intended* fills (and the closing line for CLV). It
sends nothing to any book.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = Path(os.environ.get("HKJC_JOURNAL", str(_REPO_ROOT / "journal"))) / "ops"


class OpsLog:
    """Append-only parquet log of operational rows, partitioned ``{root}/date=YYYY-MM-DD/``."""

    def __init__(self, root: Union[str, Path] = DEFAULT_ROOT, *, stream: str = "events"):
        self.root = Path(root)
        self.stream = stream

    def append(self, rows: Sequence[dict] | pl.DataFrame, *, now: Optional[datetime] = None) -> pl.DataFrame:
        """Persist ops rows. Stamps ``logged_at`` (UTC) and partitions by today's date."""
        df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(list(rows))
        if df.height == 0:
            return df
        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        df = df.with_columns(pl.lit(ts).cast(pl.Datetime("us", "UTC")).alias("logged_at"))
        d = self.root / self.stream / f"date={ts.date().isoformat()}"
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{int(ts.timestamp() * 1_000_000)}.parquet")
        return df

    def read(self, *, date: Optional[str] = None) -> pl.DataFrame:
        """Read logged rows, optionally for one ``date`` (YYYY-MM-DD). Empty frame if none."""
        pat = f"date={date}" if date else "date=*"
        files = sorted((self.root / self.stream).glob(f"{pat}/*.parquet"))
        return pl.read_parquet(files) if files else pl.DataFrame()

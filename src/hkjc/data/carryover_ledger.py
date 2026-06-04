"""Carryover ledger — join pre-close jackpot captures to settled results (B2+B3).

After a meeting, the jackpot pool either paid (the result page shows a DT/TT/6UP
dividend on the final leg) or was not won (no row → carried forward). We join that
settled outcome to the last pre-close capture (:mod:`hkjc.data.jackpot_ticks`) to
produce one durable ledger row per jackpot event: the carried-in ``C``, the gross
pool ``G``, the estimated dividend, the structural ``effective_takeout``/edge, and
whether it was won at what dividend. This accumulating ledger is the dataset behind
every carryover research question.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import polars as pl

from ..bets import carryover as _co

CARRYOVER_LEDGER_SCHEMA = {
    "date": pl.Utf8,
    "venue": pl.Utf8,
    "pool": pl.Utf8,
    "leg": pl.Int64,
    "races": pl.Utf8,
    "n_snapshots": pl.Int64,
    "carry_in": pl.Float64,            # C
    "gross_pool": pl.Float64,          # G
    "est_dividend": pl.Float64,
    "eff_takeout": pl.Float64,
    "agg_edge": pl.Float64,
    "won": pl.Int64,                   # 1 / 0 / null(unknown)
    "settled_dividend": pl.Float64,    # decimal payout if won
    "winning_combo": pl.Utf8,
    "captured_close_at": pl.Datetime(time_unit="us", time_zone="UTC"),
}

_LEG_KEYS = ["date", "venue", "pool", "leg"]


def settled_jackpot_result(pool: str, dividends: pl.DataFrame) -> Dict:
    """Extract a jackpot pool's settled outcome from a final-leg dividend frame.

    A row present ⇒ won (with decimal payout + combo); absent ⇒ not won / carried
    forward (the pool is known to be offered, since it was captured live).
    """
    if dividends is None or dividends.height == 0:
        return {"won": None, "settled_dividend": None, "winning_combo": None}
    sub = dividends.filter(pl.col("pool") == pool)
    if sub.height == 0:
        return {"won": 0, "settled_dividend": None, "winning_combo": None}
    row = sub.row(0, named=True)
    return {"won": 1, "settled_dividend": float(row["decimal"]), "winning_combo": row["combination"]}


def build_carryover_rows(
    capture: pl.DataFrame,
    settled_by_pool: Optional[Dict[str, Dict]] = None,
    net_fraction: float = _co.DEFAULT_NET_FRACTION,
) -> pl.DataFrame:
    """Collapse a meeting's capture time series + settled results into ledger rows.

    ``capture`` is the full :class:`JackpotCaptureStore` read for one meeting;
    ``settled_by_pool`` maps pool code → :func:`settled_jackpot_result` output
    (omit for an as-yet-unsettled meeting).
    """
    if capture is None or capture.height == 0:
        return pl.DataFrame(schema=CARRYOVER_LEDGER_SCHEMA)
    settled_by_pool = settled_by_pool or {}
    latest = (
        capture.sort("captured_at")
        .group_by(_LEG_KEYS, maintain_order=True)
        .agg(
            pl.col("races").last(),
            pl.col("jackpot").last().alias("carry_in"),
            pl.col("investment").last().alias("gross_pool"),
            pl.col("estimated_dividend").last().alias("est_dividend"),
            pl.len().alias("n_snapshots"),
            pl.col("captured_at").max().alias("captured_close_at"),
        )
    )
    rows = []
    for r in latest.iter_rows(named=True):
        c, g = r["carry_in"], r["gross_pool"]
        has_cg = c is not None and g is not None and g > 0
        eff = _co.effective_takeout(c, g, net_fraction) if has_cg else None
        s = settled_by_pool.get(r["pool"], {"won": None, "settled_dividend": None, "winning_combo": None})
        rows.append({
            "date": r["date"], "venue": r["venue"], "pool": r["pool"], "leg": r["leg"],
            "races": r["races"], "n_snapshots": r["n_snapshots"],
            "carry_in": c, "gross_pool": g, "est_dividend": r["est_dividend"],
            "eff_takeout": eff, "agg_edge": (-eff if eff is not None else None),
            "won": s["won"], "settled_dividend": s["settled_dividend"],
            "winning_combo": s["winning_combo"], "captured_close_at": r["captured_close_at"],
        })
    return pl.DataFrame(rows, schema=CARRYOVER_LEDGER_SCHEMA)


class CarryoverLedger:
    """Durable, accumulating ledger of jackpot events (single Parquet file)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def read(self) -> pl.DataFrame:
        if self.path.exists():
            return pl.read_parquet(self.path)
        return pl.DataFrame(schema=CARRYOVER_LEDGER_SCHEMA)

    def append(self, rows: pl.DataFrame) -> pl.DataFrame:
        """Upsert ledger rows, keyed by (date, venue, pool, leg); last write wins."""
        if rows is None or rows.height == 0:
            return self.read()
        combined = pl.concat([self.read(), rows], how="diagonal_relaxed")
        combined = combined.unique(subset=_LEG_KEYS, keep="last").sort(["date", "venue", "pool", "leg"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(self.path)
        return combined

"""Closing-line value (CLV) and market-implied probabilities.

CLV is the single most useful leading indicator in an efficient pari-mutuel
market: beating the *closing* line is strong evidence of genuine edge, and it is
observable for **every** race whether or not we bet — which partly rescues the
sparse-feedback problem of ~800 races/year. This module turns a tick time series
(see :mod:`hkjc.data.odds_ticks`) into the quantities the harness needs.

Conventions
-----------
- Odds are **decimal** (HKJC win/place dividends per unit stake).
- **CLV** of a bet = ``entry_odds / closing_odds - 1``: positive when you secured
  better (larger) odds than the close.
- **Drift** of a runner = ``close_odds / open_odds - 1``: *negative* drift means
  the odds shortened (money arrived) — a steam/informed-money signal.

This module is pure: tick/odds frames in, scored frames out. No I/O, no network.
"""
from __future__ import annotations


import numpy as np
import polars as pl

from ..core.hkjc_facts import DEFAULT_TAKEOUT

_KEYS = ["date", "venue", "race", "pool", "selection"]


# --------------------------------------------------------------------- implied
def implied_probabilities(
    odds, takeout: float = DEFAULT_TAKEOUT, normalize: bool = True
):
    """Convert decimal odds to implied probabilities.

    Parameters
    ----------
    odds:
        Array-like of decimal odds for the runners in a single pool/race.
    takeout:
        Pool takeout. Only affects the result when ``normalize`` is ``False``
        (then ``p_i = (1 - takeout) / odds_i``, an EV-style pool fraction). When
        ``normalize`` is ``True`` the takeout cancels and you get the standard
        **de-vigged** probabilities ``(1/odds_i) / sum_j (1/odds_j)`` that sum to
        1 — what you almost always want for calibration / CLV.
    normalize:
        Force the probabilities to sum to 1 (removing the overround).
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (1.0 - takeout) / odds
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    if normalize:
        s = np.nansum(p)
        if s > 0:
            p = p / s
    return p


def market_overround(odds) -> float:
    """Sum of inverse decimal odds (``1/(1-takeout)`` in an efficient pool)."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / odds
    return float(np.nansum(np.where(np.isfinite(inv), inv, np.nan)))


# ------------------------------------------------------------------- snapshots
def _edge_snapshot(ticks: pl.DataFrame, which: str, cutoff=None) -> pl.DataFrame:
    """Latest (``which='close'``) or earliest (``which='open'``) tick per key."""
    df = ticks
    if cutoff is not None:
        df = df.filter(pl.col("captured_at") <= cutoff)
    agg = pl.col("odds").last() if which == "close" else pl.col("odds").first()
    at = pl.col("captured_at").last() if which == "close" else pl.col("captured_at").first()
    lu = pl.col("last_update").last() if which == "close" else pl.col("last_update").first()
    return (
        df.sort("captured_at")
        .group_by(_KEYS, maintain_order=True)
        .agg(agg.alias("odds"), at.alias("captured_at"), lu.alias("last_update"))
    )


def closing_snapshot(ticks: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """Last odds per (date, venue, race, pool, selection), at or before ``cutoff``.

    With no ``cutoff`` this is the last odds we recorded — use a post-time cutoff
    to approximate the true closing line and avoid any post-race contamination.
    """
    return _edge_snapshot(ticks, "close", cutoff)


def opening_snapshot(ticks: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """First odds per (date, venue, race, pool, selection)."""
    return _edge_snapshot(ticks, "open", cutoff)


def odds_drift(ticks: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """Open→close drift per runner.

    Returns the keys plus ``open_odds``, ``close_odds`` and ``drift``
    (``close/open - 1``; negative = shortened = steam).
    """
    op = opening_snapshot(ticks).select(_KEYS + ["odds"]).rename({"odds": "open_odds"})
    cl = closing_snapshot(ticks, cutoff).select(_KEYS + ["odds"]).rename({"odds": "close_odds"})
    return op.join(cl, on=_KEYS, how="inner").with_columns(
        (pl.col("close_odds") / pl.col("open_odds") - 1.0).alias("drift")
    )


# -------------------------------------------------------------------------- CLV
def clv(entry_odds, closing_odds):
    """Closing-line value: ``entry_odds / closing_odds - 1`` (decimal odds)."""
    entry_odds = np.asarray(entry_odds, dtype=float)
    closing_odds = np.asarray(closing_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = entry_odds / closing_odds - 1.0
    return out


def clv_for_bets(bets: pl.DataFrame, ticks: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """Attach closing odds and CLV to a set of placed bets.

    ``bets`` must carry the key columns (``date, venue, race, pool, selection``)
    plus ``entry_odds``. Returns ``bets`` with ``close_odds`` and ``clv`` added;
    bets whose selection has no recorded closing tick get null ``close_odds``/``clv``.
    """
    close = closing_snapshot(ticks, cutoff).select(_KEYS + ["odds"]).rename({"odds": "close_odds"})
    out = bets.join(close, on=_KEYS, how="left")
    return out.with_columns(
        (pl.col("entry_odds") / pl.col("close_odds") - 1.0).alias("clv")
    )

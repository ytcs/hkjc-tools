"""Tests for closing-line value primitives."""
import datetime as dt

import numpy as np
import polars as pl

from hkjc.eval import clv


def test_implied_probabilities_sum_to_one_and_devig():
    p = clv.implied_probabilities([2.0, 4.0, 4.0])
    assert abs(p.sum() - 1.0) < 1e-9
    # normalized de-vig is independent of the takeout assumption
    p2 = clv.implied_probabilities([2.0, 4.0, 4.0], takeout=0.30)
    assert np.allclose(p, p2)
    # unnormalized applies takeout as a pool fraction
    raw = clv.implied_probabilities([2.0, 4.0, 4.0], takeout=0.175, normalize=False)
    assert abs(raw[0] - (1 - 0.175) / 2.0) < 1e-9


def test_overround():
    assert abs(clv.market_overround([2.0, 4.0, 4.0]) - 1.0) < 1e-9


def _ticks() -> pl.DataFrame:
    rows = []

    def add(minute, sel, odds):
        rows.append(dict(
            captured_at=dt.datetime(2026, 6, 3, 12, minute, tzinfo=dt.timezone.utc),
            date="2026-06-03", venue="HV", race=1, pool="WIN",
            selection=sel, odds=odds, last_update="x"))

    add(0, "1", 6.0); add(30, "1", 4.0)   # shortened (steam)
    add(0, "2", 3.0); add(30, "2", 3.5)   # drifted out
    return pl.DataFrame(rows)


def test_closing_and_opening_snapshots():
    t = _ticks()
    cl = clv.closing_snapshot(t)
    assert cl.filter(pl.col("selection") == "1")["odds"].item() == 4.0
    op = clv.opening_snapshot(t)
    assert op.filter(pl.col("selection") == "1")["odds"].item() == 6.0


def test_odds_drift_sign():
    d = clv.odds_drift(_ticks())
    assert d.filter(pl.col("selection") == "1")["drift"].item() < 0   # shortened
    assert d.filter(pl.col("selection") == "2")["drift"].item() > 0   # drifted


def test_clv_for_bets():
    bets = pl.DataFrame(dict(
        date=["2026-06-03"], venue=["HV"], race=[1], pool=["WIN"],
        selection=["1"], entry_odds=[6.0]))
    out = clv.clv_for_bets(bets, _ticks())
    assert out["close_odds"].item() == 4.0
    assert abs(out["clv"].item() - 0.5) < 1e-9   # 6.0 / 4.0 - 1


def test_clv_scalar():
    assert abs(float(clv.clv(6.0, 4.0)) - 0.5) < 1e-9

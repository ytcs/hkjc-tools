"""Lateral in metres (HKJC_LAT_METRES=1): new columns lat_m / sd_lat_m only inside the validated scope (Sha Tin home
straight, the reserved test's dates), NaN elsewhere; every existing column unchanged."""

import numpy as np
import polars as pl

from hkjc.data import Mask, MaskedRace, RaceInfo, Runner
from hkjc.recon import fusion
from hkjc.recon.pipeline import RaceRun


def _run(tmp_path=None, date="20260913", venue="ST", distance=1200.0, course="B"):
    runners = (Runner(1, "H1", "X001", 1),)
    info = RaceInfo(date, venue, 1, distance, course, "turf", "GOOD", "Class 4", runners, pl.DataFrame(),
                    pl.DataFrame())
    if tmp_path is None:
        run = RaceRun.__new__(RaceRun)
        run.info = MaskedRace(info, Mask())
        return run
    return RaceRun(info, root=tmp_path, log=lambda *a: None)


def test_scope_is_the_sha_tin_home_straight_in_the_tested_dates():
    d = np.array([900.0, 431.0, 430.0, 429.9, 200.0, 0.0, -0.1, np.nan])
    assert _run().lat_m_scope(d).tolist() == [False, False, False, True, True, True, False, False]
    assert _run(date="20260107").lat_m_scope(d).any() and _run(date="20261123").lat_m_scope(d).any()
    for r in (_run(date="20251123"), _run(date="20260104"), _run(date="20261124"),  # untested seasons / dates
              _run(venue="HV", course="C", distance=1650.0), _run(distance=1000.0)):  # Happy Valley, the straight
        assert not r.lat_m_scope(d).any()


def _path(n=300):
    t = np.arange(n) * 0.1 + 50.0
    s = 1200.0 - 600.0 + 17.0 * (t - 50.0)  # d from 600 m to about 90 m
    return fusion.Path(1, t, s, np.linspace(2.0, 12.0, n), np.full(n, 0.2), np.linspace(0.05, 0.3, n), np.ones(n))


def test_metre_columns_only_with_the_switch_and_nothing_else_changes(tmp_path, monkeypatch):
    run = _run(tmp_path)
    p = _path()
    monkeypatch.delenv("HKJC_LAT_METRES", raising=False)
    run._write_paths({1: p}, "off.parquet")
    monkeypatch.setenv("HKJC_LAT_METRES", "1")
    run._write_paths({1: p}, "on.parquet")
    A, B = pl.read_parquet(run.p("off.parquet")), pl.read_parquet(run.p("on.parquet"))
    assert B.columns == A.columns + ["lat_m", "sd_lat_m"]
    assert B.select(A.columns).equals(A)  # every existing column bitwise as before
    a, b = RaceRun.LAT_MAP
    m = (B["d"] >= 0) & (B["d"] < 430.0)
    assert 0 < m.sum() < B.height
    In, Out = B.filter(m), B.filter(~m)
    assert np.array_equal(In["lat_m"].to_numpy(), a + b * In["lat"].to_numpy())
    assert np.array_equal(In["sd_lat_m"].to_numpy(), b * In["sd_lat"].to_numpy())
    assert Out["lat_m"].is_nan().all() and Out["sd_lat_m"].is_nan().all()


def test_metre_columns_are_nan_outside_the_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HKJC_LAT_METRES", "1")
    run = _run(tmp_path, venue="HV", course="C", distance=1650.0)
    run._write_paths({1: _path()}, "hv.parquet")
    B = pl.read_parquet(run.p("hv.parquet"))
    assert B["lat_m"].is_nan().all() and B["sd_lat_m"].is_nan().all()

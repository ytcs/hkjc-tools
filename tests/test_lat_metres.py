"""Lateral in metres (HKJC_LAT_METRES=1): new columns lat_m / sd_lat_m, one map continuous in d, defined only in the
segments where it passed its drone test (drone-era dates), NaN elsewhere; every existing column unchanged."""

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
    run = RaceRun(info, root=tmp_path, log=lambda *a: None)
    np.savez(run.p("tracks.npz"), t=np.array([11.0, 11.04, 30.0]))
    return run


def test_slope_is_continuous_in_d_with_the_home_straight_map_exactly_in_the_straight():
    r = _run()
    a, bh = RaceRun.LAT_MAP
    x0, x1 = RaceRun.LAT_RAMP
    d = np.arange(-10.0, 1500.0, 0.05)
    b = r.lat_b(d)
    assert np.all(b[d - 430.0 <= x0] == bh) and np.all(b[d - 430.0 >= x1] == RaceRun.LAT_B_FAR)
    # no step: the largest change between nodes 0.05 m apart is the ramp's slope x 0.05 m
    assert np.abs(np.diff(b)).max() <= abs(bh - RaceRun.LAT_B_FAR) / (x1 - x0) * 0.05 * (1 + 1e-9)


def test_scope_follows_the_tested_segments_per_venue_and_style():
    t_r0 = 11.0
    d = np.array([1100.0, 1000.0, 700.0, 431.0, 429.9, 200.0, 0.0, -0.1])
    t = np.array([12.0, 20.0, 40.0, 55.0, 56.0, 62.0, 70.0, 71.0])
    # ST disc: start fails -> NaN there; straight, bend, home in scope; past the post out
    assert _run().lat_m_scope(d, t, t_r0).tolist() == [False, True, True, True, True, True, True, False]
    # ST silk 2026: start in scope too
    assert _run(date="20260426").lat_m_scope(d, t, t_r0).tolist() == [True, True, True, True, True, True, True, False]
    # before the first render frame: out
    assert not _run().lat_m_scope(np.array([1150.0]), np.array([10.9]), t_r0).any()
    hv = _run(venue="HV", course="C", distance=1650.0)
    seg = hv.lat_m_segment(d, t, t_r0)
    m = hv.lat_m_scope(d, t, t_r0)
    assert set(seg[m]) <= {"start", "straight"} and m.any()  # HV disc: bend and home straight failed
    for r in (_run(date="20251123"), _run(date="20260104"), _run(date="20261124"), _run(distance=1000.0)):
        assert not r.lat_m_scope(d, t, t_r0).any()  # untested seasons / dates, the 1000 m straight


def _path(n=600):
    t = np.arange(n) * 0.1 + 30.0
    s = 1200.0 - 1000.0 + 17.0 * (t - 30.0)  # d from 1000 m to past the post
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
    a, bh = RaceRun.LAT_MAP
    d, lat, sd = (B[c].to_numpy() for c in ("d", "lat", "sd_lat"))
    m = np.isfinite(B["lat_m"].to_numpy())
    assert np.array_equal(m, d >= 0)  # ST disc 1200 m, t >= 30 s (past the start): straight, bend, home to the post
    b = run.lat_b(d)
    assert np.array_equal(B["lat_m"].to_numpy()[m], a + b[m] * lat[m])
    assert np.array_equal(B["sd_lat_m"].to_numpy()[m], b[m] * sd[m])
    home = m & (d < 430.0)  # the home straight: the Sha Tin home-straight map exactly
    assert np.array_equal(B["lat_m"].to_numpy()[home], a + bh * lat[home])
    assert B["sd_lat_m"].to_numpy()[~m].tolist() == [] or np.isnan(B["sd_lat_m"].to_numpy()[~m]).all()
    # one unit along the path: no jump between consecutive defined nodes beyond the map's own change
    lm = B["lat_m"].to_numpy()
    step = np.abs(np.diff(lm) - b[1:] * np.diff(lat))[m[1:] & m[:-1]]
    assert step.max() < 0.01


def test_metre_columns_are_nan_outside_the_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HKJC_LAT_METRES", "1")
    run = _run(tmp_path, date="20251123")
    run._write_paths({1: _path()}, "old.parquet")
    B = pl.read_parquet(run.p("old.parquet"))
    assert B["lat_m"].is_nan().all() and B["sd_lat_m"].is_nan().all()

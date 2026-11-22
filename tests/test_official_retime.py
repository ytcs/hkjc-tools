"""Final product: paths are re-timed onto the official boundary times."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from hkjc.data import Mask, MaskedRace
from hkjc.recon.fusion import Path
from hkjc.recon.pipeline import RaceRun
from hkjc.recon.validate import crossing


def test_retime_puts_every_official_boundary_exactly_on_time():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info)
    out, warp = run._official_retime({1: p})
    q = out[1]
    for b, tc in zip((800.0, 400.0, 0.0), np.cumsum([24.0, 23.0, 23.5])):
        assert abs(crossing(q.t, 1200.0 - q.s, None, b) - tc) < 5e-3  # grid resampling (0.05 s nodes)
    assert np.all(np.diff(q.s) >= 0) and warp[1] > 0


def test_equal_official_times_keep_the_official_placing():
    secs = pl.DataFrame({"runner": [1, 2], "section": [1, 1], "section_time": [70.5, 70.5]})
    res = pl.DataFrame({"HorseNo": [2, 1], "Pla": [1, 2], "FinishTime": [70.5, 70.5]})  # #2 won on the photo
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)

    def mk(h, v):
        return Path(h, t, v * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))

    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info)
    out, _w = run._official_retime({1: mk(1, 17.1), 2: mk(2, 16.9)})
    c = {h: crossing(q.t, 1200.0 - q.s, None, 0.0) for h, q in out.items()}
    assert c[2] < c[1] and c[1] - c[2] < 0.01


def test_a_withheld_boundary_is_not_retimed():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info, Mask(boundary=1))  # boundary 1 = 400 m to go, official 47.0 s
    q = run._official_retime({1: p})[0][1]
    assert abs(crossing(q.t, 1200.0 - q.s, None, 400.0) - 47.0) > 0.05  # not snapped onto the official time
    assert abs(crossing(q.t, 1200.0 - q.s, None, 0.0) - 70.5) < 5e-3  # the others are


def test_retime_adds_no_speed_jump_at_the_boundaries():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.round(np.arange(0, 80.0001, 0.05), 2)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info)
    q = run._official_retime({1: p})[0][1]
    v = np.gradient(q.s, q.t)
    k = q.t <= 71.0  # beyond: the warped path runs past the input's end
    # a piecewise-linear warp jumps ~0.7 m/s at 800 m / 400 m (section rates 0.930 / 0.891 / 0.911);
    # a smooth one changes speed by < 0.005 m/s per 0.05 s node (the warp's curvature), boundaries included
    assert np.max(np.abs(np.diff(v[k]))) < 5e-3


def test_monotone_warp_is_strictly_increasing_and_c1():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(2, 7)
        kt = np.cumsum(rng.uniform(0.05, 30.0, n))
        kw = kt[0] + np.concatenate([[0.0], np.cumsum(rng.uniform(0.01, 40.0, n - 1))])
        t = np.linspace(kt[0] - 5, kt[-1] + 5, 20001)
        w = RaceRun._monotone_warp(kt, kw, t)
        assert np.all(np.diff(w) > 0)
        assert np.allclose(RaceRun._monotone_warp(kt, kw, kt), kw)
        dw = np.diff(w) / np.diff(t)
        assert np.max(np.abs(np.diff(dw))) < 0.05 * np.max(dw)  # no slope jump at a knot

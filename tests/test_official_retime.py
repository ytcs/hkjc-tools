"""Final product: paths are re-timed onto the official finish (interior official times are soft, in the fusion)."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from hkjc.data import Mask, MaskedRace
from hkjc.recon.fusion import Path
from hkjc.recon.pipeline import RaceRun
from hkjc.recon.validate import crossing


def test_retime_hits_the_official_finish_and_keeps_interior_crossings_soft():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info)
    out, warp = run._official_retime({1: p})
    q = out[1]
    assert abs(crossing(q.t, 1200.0 - q.s, None, 0.0) - 70.5) < 5e-3  # the finish: exact (0.05 s node resampling)
    before = q.t <= 800.0 / 15.5  # up to the last section's start the path is untouched
    assert np.allclose(q.s[before], p.s[before])
    for b in (800.0, 400.0):  # interior crossings stay the fused path's own (soft official times)
        assert abs(crossing(q.t, 1200.0 - q.s, None, b) - (1200.0 - b) / 15.5) < 1e-4  # 400 m: at the segment start
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


def test_a_withheld_finish_means_no_retiming():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.linspace(0, 80, 1601)
    p = Path(1, t, 15.5 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info, Mask(boundary=2))  # boundary 2 = the finish
    out, warp = run._official_retime({1: p})
    assert warp == {} and out[1] is p


def test_retime_adds_no_speed_jump_at_the_boundaries():
    secs = pl.DataFrame({"runner": [1, 1, 1], "section": [1, 2, 3], "section_time": [24.0, 23.0, 23.5]})
    res = pl.DataFrame({"HorseNo": [1], "Pla": [1], "FinishTime": [70.5]})
    info = SimpleNamespace(sectionals=secs, distance=1200.0, results=res)
    t = np.round(np.arange(0, 80.0001, 0.05), 2)
    p = Path(1, t, 17.0 * t, np.full_like(t, 3.0), np.full_like(t, 0.2), np.full_like(t, 0.3), np.ones_like(t))
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info)
    q = run._official_retime({1: p})[0][1]
    v = np.gradient(q.s, q.t)
    k = q.t <= 71.0  # beyond: the warped path runs past the input's end
    # the finish correction (70.59 -> 70.5 s) is spread over the last section by a C1 segment: no speed jump at its
    # start, and < 0.005 m/s change per 0.05 s node anywhere
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


def test_finish_warp_is_identity_before_the_last_section_and_c1():
    t = np.linspace(0.0, 90.0, 18001)
    for tf, wf in ((70.0, 70.4), (70.0, 68.5), (70.0, 72.0)):
        w = RaceRun._finish_warp(50.0, tf, wf, t)
        assert np.array_equal(w[t <= 50.0], t[t <= 50.0])
        assert abs(np.interp(tf, t, w) - wf) < 1e-9 and np.all(np.diff(w) > 0)
        dw = np.diff(w) / np.diff(t)
        assert np.max(np.abs(np.diff(dw))) < 1e-3  # no slope jump at 50 s or at the finish

"""The product's metric line (METRIC_W, RaceRun.metric_w / _shape_bias): Happy Valley turf ovals only; the bias table
keeps its values at the visible boundaries and takes the line metric's shape between them."""

import numpy as np
import polars as pl

from hkjc.data import RaceInfo, Runner
from hkjc.recon import pipeline
from hkjc.recon.pipeline import RaceRun


def _run(tmp_path, venue="HV", surface="turf", straight=False):
    info = RaceInfo("20260603", venue, 8, 1650.0, "C", surface, "GOOD", "Class 4", (Runner(1, "H1", "X001", 1),),
                    pl.DataFrame(), pl.DataFrame())
    run = RaceRun(info, root=tmp_path, log=lambda *a: None)
    np.savez(run.p("rail.npz"), rail=np.zeros((3, 2)), **({"lat0_px": 4.0} if straight else {}))
    return run


def test_metric_w_only_on_the_listed_venues_turf_ovals(tmp_path):
    assert pipeline.METRIC_W == {"HV": 3.10}
    assert _run(tmp_path / "a").metric_w() == 3.10
    assert _run(tmp_path / "b", venue="ST").metric_w() == 0.0
    assert _run(tmp_path / "c", straight=True).metric_w() == 0.0
    assert _run(tmp_path / "d", surface="awt").metric_w() == 0.0


def test_shape_bias_keeps_the_boundaries_and_takes_the_line_metric_shape_between(tmp_path, monkeypatch):
    run = _run(tmp_path)
    s = np.arange(0.0, 1650.0, 0.5)
    d0 = 1650.0 - s
    dw = d0 - 2.0 * np.sin(s / 300.0)  # the line metric puts render s ahead by 2 sin(s / 300)
    monkeypatch.setattr(run, "_observed_dets", lambda: (np.zeros((len(s), 6)), np.zeros((len(s), 2)), None))
    monkeypatch.setattr(run, "_rail_projection", lambda rows, X: (None, None, None, None, None))
    monkeypatch.setattr(run, "tick_metric", lambda rows, s_px, rf, m, w=0.0: {"d": dw if w else d0})
    bias = (np.array([450.0, 850.0, 1250.0, 1650.0]), np.array([1.0, -0.5, 0.3, 0.0]))
    assert run._shape_bias(None) is None
    assert _run(tmp_path / "st", venue="ST")._shape_bias(bias) is bias  # no line at Sha Tin: unchanged
    grid, shift = run._shape_bias(bias)
    assert np.allclose(np.interp(bias[0], grid, shift), bias[1], atol=1e-9)  # the visible boundaries keep b
    mid = 650.0  # between boundaries: b's chord plus the line metric's shape less its chord

    def ds(x):
        return 2.0 * np.sin(np.round(x) / 300.0)  # median per metre of render s

    want = np.interp(mid, bias[0], bias[1]) + ds(mid) - np.interp(mid, bias[0][:2], ds(bias[0][:2]))
    assert abs(np.interp(mid, grid, shift) - want) < 0.02


def test_shaped_moves_raw_biased_render_observations_onto_the_shape(tmp_path, monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Ob:
        feed: str
        s: object

    run = _run(tmp_path)
    raw = (np.array([450.0, 1650.0]), np.array([1.0, 0.0]))
    grid = np.arange(0.0, 1700.0, 1.0)
    shape = (grid, np.interp(grid, *raw) + 0.5 * np.sin(grid / 200.0))
    monkeypatch.setattr(run, "_shape_bias", lambda b: shape if b is raw else b)
    s_raw = np.array([500.0, 900.0, 1300.0])
    obs = [Ob("render", s_raw + np.interp(s_raw, *raw)), Ob("official", np.array([1.0]))]
    out = run._shaped(obs, raw)
    assert np.allclose(out[0].s, s_raw + np.interp(s_raw, *shape), atol=1e-3)  # labels untouched, s on the shape
    assert out[1] is obs[1]
    monkeypatch.setattr(run, "_shape_bias", lambda b: b)
    assert run._shaped(obs, raw) is obs

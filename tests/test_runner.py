"""Stage runner: atomic artifact writes, forced-stage propagation, background-stage cleanup."""

import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hkjc.recon import video
from hkjc.recon.pipeline import DEPS, OUTPUTS, STAGES, RaceRun, upstream


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    target = tmp_path / "a.npz"
    with pytest.raises(RuntimeError):
        with video.atomic(target) as fp:
            np.savez(fp, x=np.arange(3))
            raise RuntimeError("killed mid-stage")
    assert list(tmp_path.iterdir()) == []
    with video.atomic(target) as fp:
        np.savez(fp, x=np.arange(3))
    assert [p.name for p in tmp_path.iterdir()] == ["a.npz"]
    assert np.load(target)["x"].tolist() == [0, 1, 2]


def test_deps_table_consistent():
    assert set(DEPS) == set(STAGES) == set(OUTPUTS)
    for st in STAGES:  # every dependency runs earlier in STAGES order
        assert all(STAGES.index(d) < STAGES.index(st) for d in DEPS[st]), st
    assert upstream("film") == {"frames", "hud", "start"}
    assert "finish" in upstream("validate") and "finish" not in upstream("observe")


def _run(tmp_path, cls=RaceRun):
    return cls(SimpleNamespace(key="race"), root=tmp_path, log=lambda *_: None)


def test_force_reruns_consumers(tmp_path):
    R = _run(tmp_path)
    for outs in OUTPUTS.values():
        for o in outs:
            R.p(o).touch()
    assert R.plan() == []
    assert R.plan(force=("rails",)) == [
        "rails", "observe", "identify", "fuse_video", "fuse_final", "validate"
    ]
    assert R.plan(force=("finish",)) == ["finish", "identify", "fuse_video", "fuse_final", "validate"]
    # a missing artifact is regenerated (deterministic) without invalidating its consumers
    R.p("film.npz").unlink()
    assert R.plan() == ["film"]


class _Failing(RaceRun):
    def frames(self):
        pass

    def hud(self):  # background stage that would outlive the failure
        time.sleep(60)

    def features(self):
        raise RuntimeError("main-chain failure")


def test_failed_stage_terminates_background(tmp_path):
    t0 = time.time()
    with pytest.raises(RuntimeError, match="main-chain failure"):
        _run(tmp_path, _Failing).run(stages=("frames", "features", "hud"))
    assert time.time() - t0 < 20
    assert mp.active_children() == []


def test_yolo_backfilled_from_yolo_only_dets(tmp_path):
    R = _run(tmp_path)
    R.p("feats.npz").touch()
    rows, hist = np.arange(12.0).reshape(2, 6), np.ones((2, 14))
    np.savez_compressed(R.p("dets.npz"), rows=rows, hist=hist, box=np.zeros(4), source="yolo")
    assert R.done("features")
    z = np.load(R.p("yolo.npz"))
    assert np.array_equal(z["rows"], rows) and np.array_equal(z["hist"], hist)
    R.p("yolo.npz").unlink()
    np.savez_compressed(R.p("dets.npz"), rows=rows, hist=hist, box=np.zeros(4), source="yolo+bgsub")
    assert not R.done("features")  # map-difference blobs are mixed in: features must re-run


def test_is_straight_survives_a_fold_back():
    from hkjc.recon import rails

    def tangents(theta):
        return np.c_[np.cos(theta), np.sin(theta)]

    bend = np.linspace(0, 3.2, 400)  # an oval's turning
    assert rails.is_straight(tangents(np.zeros(400)))
    assert not rails.is_straight(tangents(bend))
    folded = np.r_[bend, np.linspace(3.2, 0.1, 40)]  # parked frames fold the end back
    assert not rails.is_straight(tangents(folded))


def test_final_holdout_withholds_the_boundary_from_metric_bias(monkeypatch):
    """HKJC_FINAL_HOLDOUT=j removes boundary j from the shared render-metric bias too (a held-out
    boundary must not set the bias it is scored against)."""
    from types import SimpleNamespace

    import numpy as np
    import polars as pl

    from hkjc.recon.pipeline import RaceRun

    secs = pl.DataFrame(
        {
            "runner": [r for r in (1, 2, 3) for _ in range(3)],
            "section": [1, 2, 3] * 3,
            "section_time": [24.0, 24.0, 24.0] * 3,
        }
    )
    info = SimpleNamespace(sectionals=secs, distance=1200.0)
    t = np.linspace(0, 80, 801)
    paths = {h: SimpleNamespace(t=t, s=15.0 * t, sd_s=np.full_like(t, 0.2)) for h in (1, 2, 3)}
    run = SimpleNamespace(info=info, log=lambda *a: None)
    monkeypatch.delenv("HKJC_FINAL_HOLDOUT", raising=False)
    sb, _b = RaceRun._metric_bias(run, paths)
    assert len(sb) == 3
    monkeypatch.setenv("HKJC_FINAL_HOLDOUT", "1")
    sb1, _b1 = RaceRun._metric_bias(run, paths)
    assert len(sb1) == 2 and 1200.0 - 400.0 not in sb1.tolist()

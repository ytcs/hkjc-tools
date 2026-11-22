"""Stage runner: atomic artifact writes, forced-stage propagation, background-stage cleanup."""

import json
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
    assert upstream("film") == {"frames", "hud"}
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
    # a large artifact deleted for space is regenerated (deterministic) without invalidating its consumers
    R.p("frames.bin").unlink()
    assert R.plan() == ["frames"]
    R.p("frames.bin").touch()
    # a stage that never ran (e.g. start) re-runs every consumer: their artifacts predate it
    R.p("start.npz").unlink()
    assert R.plan() == ["start", "observe", "identify", "fuse_video", "fuse_final", "validate"]
    # ... but only when it is requested
    assert R.plan(stages=("identify", "fuse_video")) == []


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


def test_withheld_boundary_is_withheld_from_metric_bias():
    """Mask(boundary=j) removes boundary j from the shared render-metric bias too (a held-out
    boundary must not set the bias it is scored against)."""
    import polars as pl

    from hkjc.data import Mask, MaskedRace

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
    run = SimpleNamespace(info=MaskedRace(info), log=lambda *a: None)
    sb, _b = RaceRun._metric_bias(run, paths)
    assert len(sb) == 3
    run.info = MaskedRace(info, Mask(boundary=1))
    sb1, _b1 = RaceRun._metric_bias(run, paths)
    assert len(sb1) == 2 and 1200.0 - 400.0 not in sb1.tolist()


def test_status_records_the_failed_stage(tmp_path):
    R = _run(tmp_path, _Failing)
    with pytest.raises(RuntimeError):
        R.run(stages=("frames", "features"))
    st = json.load(open(R.p("status.json")))
    assert st["status"] == "failed" and st["stage"] == "features" and "main-chain failure" in st["error"]


def test_health_flags_an_unreadable_render(tmp_path):
    R = RaceRun(SimpleNamespace(key="race", distance=1200, n=12, venue="HV"), root=tmp_path)
    np.savez(R.p("frames_idx.npz"), idx=np.arange(150))  # 150 render ticks for a ~73 s race
    np.savez(R.p("yolo.npz"), rows=np.c_[np.repeat(np.arange(150), 1), np.zeros((150, 5))])  # 1 marker/frame
    rows = np.c_[np.arange(10.0), np.arange(10.0), np.arange(10.0)]
    np.savez(R.p("hud.npz"), rows=rows)
    np.savez(R.p("rail.npz"), rail=np.zeros((3, 2)), lat0_px=1.0)  # 'straight' on a Happy Valley oval
    R.write_status()
    st = json.load(open(R.p("status.json")))
    assert st["status"] == "unhealthy"
    assert set(st["unhealthy"]) == {"render_ticks_per_s", "markers_per_runner", "course_shape"}
    assert st["checks"]["clock_reads"]["ok"]


def test_official_clock_offset_is_a_robust_median():
    off = RaceRun.official_clock_offset
    # three horses 0.48 s late against the official finish, one camera misread far off
    cam = {1: 109.16, 2: 109.40, 3: 109.90, 4: 120.0}
    official = {1: 109.64, 2: 109.88, 3: 110.38, 4: 110.50, 5: 111.0}
    assert abs(off(cam, official) - 0.48) < 1e-9
    assert off({1: 10.0, 2: 10.5}, {1: 10.5, 2: 11.0}) is None  # fewer than 3 matched horses
    assert off({1: 10.0, 2: 10.5, 3: 11.0}, {1: 10.5, 2: float("nan"), 3: 11.5}) is None
    assert off(cam, {}) is None  # finish withheld by the mask


def test_camera_crossings_leave_fuse_final_where_the_official_finish_is_visible(tmp_path, monkeypatch):
    from hkjc.recon import fusion

    R = RaceRun(SimpleNamespace(key="race", distance=1200.0, n=3, venue="HV", width=25.0), root=tmp_path)
    cam = [fusion.Obs(np.array([70.0 + 0.1 * h]), np.array([1200.0]), None, 0.3, 1.0, "finishcam", horse=h, tid=10_000 + h)
           for h in (1, 2, 3)]
    monkeypatch.setattr(RaceRun, "finish_camera", lambda self, paths: list(cam))
    assert [o.horse for o in R.finish_observations({})] == [1, 2, 3]  # fuse_video: every camera crossing
    R._official_clock = {1: 70.5, 2: float("nan")}  # fuse_final: #1 visible, #2 withheld, #3 without a time
    assert [o.horse for o in R.finish_observations({})] == [2, 3]
    R._official_clock = {}  # finish withheld by the mask: unchanged
    assert [o.horse for o in R.finish_observations({})] == [1, 2, 3]
    # seeds keep every camera-read path-less horse; a visible official finish times the seed (sd OFFICIAL_SD)
    R._official_clock = {1: 70.5}
    p = fusion.Path(9, np.arange(60.0, 72.0, 0.1), np.linspace(1000.0, 1210.0, 120), np.full(120, 3.0),
                    np.full(120, 0.2), np.full(120, 0.2), np.ones(120), {})
    seeds = R._finish_seeds({9: p}, fusion.Prior(width=25.0))
    assert sorted(seeds) == [1, 2, 3]
    s1 = seeds[1]
    assert abs(np.interp(1200.0, s1.s, s1.t) - 70.5) < 1e-6 and abs(s1.sd_s.min() - R.OFFICIAL_SD) < 0.02
    assert abs(np.interp(1200.0, seeds[2].s, seeds[2].t) - 70.2) < 1e-6


def test_registration_shift_puts_both_finish_clocks_in_the_shifted_frame(tmp_path, monkeypatch):
    from hkjc.recon import finishcam

    bias = (np.array([400.0, 800.0]), np.array([-2.0, -9.0]))  # held flat past the last visible boundary
    assert abs(RaceRun.registration_shift(bias, 1200.0, 15.0) - 0.6) < 1e-12  # render +9 m short -> 0.6 s later
    R = RaceRun(SimpleNamespace(key="race", distance=1200.0, n=4, venue="ST", width=25.0), root=tmp_path,
                log=lambda *_: None)
    np.savez(R.p("finish.npz"), line_x=100.0, reads=np.zeros((0, 4)), ppm=100.0)
    gaps = {1: 0.0, 2: 300.0, 3: 600.0, 4: 900.0}
    monkeypatch.setattr(finishcam, "finish_gaps", lambda sc, with_sd=False: ([1, 2, 3, 4], gaps, None, {}))
    # the anchor fits the (bias-shifted) fused crossings: winner at 70.6 s; the label-free registration of the
    # unshifted tracklets: 70.0 s
    monkeypatch.setattr(finishcam, "anchor", lambda g, t, **kw: (70.6, 1500.0, 4, np.eye(2) * 1e-4))
    monkeypatch.setattr(RaceRun, "_finish_identity", lambda self, *a: ({}, {"t_w": 70.0, "C": 1500.0, "edge": False,
                                                                               "n_close": 4}))
    t = np.arange(60.0, 72.0, 0.1)
    p = SimpleNamespace(t=t, s=1200.0 + 15.0 * (t - 70.6), sd_s=np.full(len(t), 0.2))
    paths = {h: p for h in gaps}
    t_win = lambda obs: min(float(o.t[0]) for o in obs)  # noqa: E731
    assert abs(t_win(R.finish_camera(paths)) - 70.0) < 1e-9  # frames compared unshifted: anchor rejected
    R._render_bias = bias
    assert abs(t_win(R.finish_camera(paths)) - 70.6) < 1e-9  # one frame: the anchor stands

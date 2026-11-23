"""Opt-in runtime switches: HKJC_MATCH_HELPER (per-race matcher helper + detector feeding the feature workers) and
HKJC_CUDA_BLOCKING_SYNC (CUDA primary context in blocking-sync mode). Both are off by default."""

import numpy as np
import pytest


def test_match_helper_is_off_by_default_and_routes_the_cuda_tables(monkeypatch):
    from hkjc.recon import bfmatch

    calls = []
    monkeypatch.setattr(bfmatch, "_tables_oneshot", lambda dev, des, jobs: calls.append("oneshot") or {})
    monkeypatch.setattr(bfmatch, "_tables_subprocess", lambda dev, des, jobs: calls.append("kept") or {})
    monkeypatch.setattr(bfmatch, "_helper", lambda dev: calls.append("warm"))
    monkeypatch.setenv("HKJC_MATCH_DEVICE", "cuda")
    des = [np.zeros((5, 32), np.uint8)] * 2
    monkeypatch.delenv("HKJC_MATCH_HELPER", raising=False)
    assert not bfmatch.helper_on()
    bfmatch.warm()
    bfmatch.tables(des, [(0, 1, True)])
    assert calls == ["oneshot"]  # no helper started, a fresh interpreter per call (as before the switch)
    calls.clear()
    monkeypatch.setenv("HKJC_MATCH_HELPER", "1")
    bfmatch.warm()
    bfmatch.tables(des, [(0, 1, True)])
    assert calls == ["warm", "kept"]


def test_features_job_detects_in_the_worker_unless_given_the_detector_pool_boxes(monkeypatch):
    from hkjc.recon import pipeline

    class Det:
        def raw(self, p):
            return np.array([[10.0, 20.0, 30.0, 44.0], [5.0, 400.0, 25.0, 420.0]]), np.array([0.9, 0.8])

        def panel(self, p, lt):
            from hkjc.recon.detect import AvatarDetector

            return AvatarDetector.drop_legend(self.raw(p), lt)

    class Cache:
        idx = np.array([7])

        def panel(self, k):
            rng = np.random.default_rng(0)
            return rng.integers(0, 255, (535, 945, 3), np.uint8)

    monkeypatch.setattr(pipeline, "_CACHE", Cache())
    monkeypatch.setattr(pipeline, "_STYLE", "silk")
    monkeypatch.setattr(pipeline, "_DET", Det())
    here = pipeline._feat_job((0, 500, pipeline._DETECT_HERE))
    fed = pipeline._feat_job((0, 500, Det().raw(None)))
    assert here[0] == fed[0] == 7 and here[1] == fed[1] and np.array_equal(here[4], fed[4])
    assert np.array_equal(here[2].pts, fed[2].pts) and np.array_equal(here[2].des, fed[2].des)
    assert all(np.array_equal(a, b) for a, b in zip(here[5], fed[5]))


def test_blocking_sync_is_a_no_op_unless_switched_on_for_cuda(monkeypatch):
    import ctypes

    from hkjc.recon import cudasched

    def boom(*a, **k):
        raise AssertionError("libcuda touched")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    monkeypatch.setattr(cudasched, "_DONE", False)
    monkeypatch.delenv("HKJC_CUDA_BLOCKING_SYNC", raising=False)
    cudasched.blocking_sync("cuda")
    monkeypatch.setenv("HKJC_CUDA_BLOCKING_SYNC", "1")
    cudasched.blocking_sync("cpu")
    cudasched.blocking_sync("intel:gpu")


def test_blocking_sync_sets_the_primary_context_flag(monkeypatch):
    import ctypes

    from hkjc.recon import cudasched

    seen = []

    class FakeCuda:
        def cuInit(self, f):  # noqa: N802 (the driver API names)
            seen.append(("init", f))
            return 0

        def cuDeviceGet(self, ref, ordinal):  # noqa: N802
            seen.append(("get", ordinal))
            return 0

        def cuDevicePrimaryCtxSetFlags_v2(self, d, flags):  # noqa: N802
            seen.append(("flags", flags))
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeCuda())
    monkeypatch.setattr(cudasched, "_DONE", False)
    monkeypatch.setenv("HKJC_CUDA_BLOCKING_SYNC", "1")
    cudasched.blocking_sync("cuda:1")
    assert seen == [("init", 0), ("get", 1), ("flags", cudasched.CU_CTX_SCHED_BLOCKING_SYNC)]
    cudasched.blocking_sync("cuda")  # once per process
    assert len(seen) == 3

    class Refuses(FakeCuda):
        def cuDevicePrimaryCtxSetFlags_v2(self, d, flags):  # noqa: N802
            return 708  # CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Refuses())
    monkeypatch.setattr(cudasched, "_DONE", False)
    with pytest.raises(RuntimeError, match="CUresult 708"):
        cudasched.blocking_sync("cuda")

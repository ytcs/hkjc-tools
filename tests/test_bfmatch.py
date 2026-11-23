"""bfmatch: brute-force Hamming tables index-identical to cv2.BFMatcher (knn k=4 and cross-check)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _des(rng, n, dup):
    d = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    d[rng.integers(0, n, dup)] = d[rng.integers(0, n, dup)]  # duplicates: distance ties
    d[:, 8:] &= 0x0F  # shared high bits: many equal distances
    return d


def test_tables_match_opencv_indices_and_tie_order(monkeypatch):
    import cv2

    from hkjc.recon import bfmatch

    monkeypatch.setenv("HKJC_MATCH_DEVICE", "cpu")
    rng = np.random.default_rng(0)
    des = [_des(rng, n, 40) for n in (300, 280, 3, 1)]
    jobs = [(0, 1, True), (0, 1, False), (1, 0, False), (0, 2, False), (2, 3, True), (0, 3, False)]
    T = bfmatch.tables(des, jobs)
    for i, j, cross in jobs:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross)
        if cross:
            ms = bf.match(des[i], des[j])
            ia, ib = T[(i, j, True)]
            assert ia.tolist() == [m.queryIdx for m in ms] and ib.tolist() == [m.trainIdx for m in ms]
        else:
            ref = [[m.trainIdx for m in q] for q in bf.knnMatch(des[i], des[j], k=4)]
            assert T[(i, j, False)].tolist() == ref


def test_tables_empty_when_not_opted_in(monkeypatch):
    from hkjc.recon import bfmatch

    monkeypatch.delenv("HKJC_MATCH_DEVICE", raising=False)
    assert bfmatch.tables([np.zeros((5, 32), np.uint8)] * 2, [(0, 1, True)]) == {}


def test_helper_interpreter_gives_the_in_process_tables():
    from hkjc.recon import bfmatch

    rng = np.random.default_rng(1)
    des = [_des(rng, 200, 30), _des(rng, 180, 30)]
    jobs = [(0, 1, True), (0, 1, False)]
    a, b = bfmatch._tables("cpu", des, jobs), bfmatch._tables_subprocess("cpu", des, jobs)
    assert a.keys() == b.keys()
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
    # the helper is kept: a second request is served by the same interpreter, with the same tables
    pid = bfmatch._HELPER[0].pid
    c = bfmatch._tables_subprocess("cpu", des[::-1], jobs)
    assert bfmatch._HELPER[0].pid == pid
    d = bfmatch._tables("cpu", des[::-1], jobs)
    assert all(np.array_equal(np.asarray(c[k]), np.asarray(d[k])) for k in d)
    bfmatch._close()
    assert bfmatch._HELPER is None

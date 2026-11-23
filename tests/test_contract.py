"""Output contract (HKJC_CONTRACT=1, opt-in): free-running flag, E-step margins."""

import numpy as np
import polars as pl

from hkjc.recon import assoc, contract, fusion


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HKJC_CONTRACT", raising=False)
    assert not contract.contract_on()
    monkeypatch.setenv("HKJC_CONTRACT", "1")
    assert contract.contract_on()


def test_node_free_uses_one_length_and_the_abreast_spacing():
    t = np.array([0.0, 0.1])
    P = pl.DataFrame({"horse_no": [1, 1, 2, 2, 3, 3], "t": np.tile(t, 3),
                      "s": [10.0, 11.0, 11.0, 20.0, 30.0, 30.0], "lat": [1.0, 1.0, 3.5, 3.5, 1.0, 9.0]})
    f = contract.node_free(P)
    # 1 and 2 at t = 0: 1 m along and 2.5 render units across (abreast) -> neither is free; 9 m apart at t = 0.1
    assert f[1][1].tolist() == [False, True] and f[2][1].tolist() == [False, True]
    assert f[3][1].tolist() == [True, True]


def test_margins_are_read_only_and_rank_the_ambiguous_detection_lowest():
    tg = np.arange(0, 10.01, 0.1)
    paths = {
        1: fusion.Path(1, tg, 15.0 * tg, np.full_like(tg, 2.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
        2: fusion.Path(2, tg, 15.0 * tg - 3.0, np.full_like(tg, 4.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
    }
    t = np.array([5.0, 5.0, 5.0, 6.0])
    s = np.array([75.0, 72.0, 40.0, 88.5])  # horse 1, horse 2, clutter; at t = 6 one detection between the two
    lat = np.array([2.0, 4.0, 3.0, 3.0])
    fr = np.array([7, 7, 7, 8])
    lab0, chi0 = assoc.assign_frames(fr, t, s, lat, paths, np.zeros((0, 9), int))
    m = np.full(4, np.nan)
    lab, chi = assoc.assign_frames(fr, t, s, lat, paths, np.zeros((0, 9), int), margin_out=m)
    assert np.array_equal(lab, lab0) and np.array_equal(chi, chi0)
    assert lab.tolist()[:3] == [1, 2, 0]
    assert np.isnan(m[2]) and (m[[0, 1]] > 0).all()
    assert m[3] < m[0] and m[3] < m[1]


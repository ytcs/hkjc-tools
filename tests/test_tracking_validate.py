import numpy as np

from hkjc.recon import tracking
from hkjc.recon.validate import crossing


def test_two_horses_tracked_through_a_gap():
    t = np.arange(0, 10, 0.1)
    S, L, F = [], [], []
    for k, tk in enumerate(t):
        if 4.0 < tk < 4.6:  # detection gap
            continue
        for lat in (2.0, 4.0):
            S.append(17 * tk)
            L.append(lat)
            F.append(k)
    trs = tracking.stitch(tracking.track(t, np.array(S), np.array(L), np.array(F)))
    assert len(trs) == 2 and all(len(tr.t) > 90 for tr in trs)


def test_crossing_interpolates_and_respects_sd():
    t = np.arange(0, 10, 0.1)
    d = 100 - 17 * t
    assert abs(crossing(t, d, None, 50.0) - 50 / 17) < 1e-6
    assert np.isnan(crossing(t, d, np.full(len(t), 5.0), 50.0))

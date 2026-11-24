import numpy as np

from hkjc.recon import fusion as F


def test_lateral_bound_enforced():
    """The core's lateral smoother keeps a horse inside the course: [l_min, width - l_min]."""
    from types import SimpleNamespace

    from hkjc.recon import core

    t = np.arange(1.0, 10, 0.1)
    m = core.Model(SimpleNamespace(horses=np.array([1])), core.list_rows(None, None, [1]), t, 20.0)
    lat, _J, _A = m.solve_l(core.Obs(np.zeros(len(t), int), t, np.full(len(t), -0.5), np.full(len(t), 0.3)))
    assert lat.min() >= core.L_MIN - 1e-6


def test_with_tail_appends_only():
    t = np.arange(0.0, 2.0 + 1e-9, F.DT)
    p = F.Path(
        1,
        t,
        16.0 * t,
        np.full(len(t), 3.0),
        np.full(len(t), 0.2),
        np.full(len(t), 0.1),
        np.ones(len(t), int),
    )
    q = F.with_tail(p, F.Prior())
    n = len(t)
    assert len(q.t) == n + int(round(F.TAIL / F.DT))
    for a in ("t", "s", "lat", "sd_s", "sd_l", "n_obs"):
        assert np.array_equal(getattr(q, a)[:n], getattr(p, a))
    assert np.allclose(np.diff(q.s[n - 1 :]), 16.0 * F.DT)
    assert np.all(q.lat[n:] == 3.0) and np.all(q.n_obs[n:] == 0)
    assert np.all(np.diff(q.sd_s[n - 1 :]) > 0)

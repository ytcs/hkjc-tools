import numpy as np

from hkjc.recon import fusion as F


def _truth():
    t = np.arange(0, 60, 0.1)
    return t, 17 * t + 0.02 * t**2, 2 + 1.5 * np.sin(t / 10)


def test_fuse_bridges_gaps_and_associates():
    rng = np.random.default_rng(0)
    t, s, lat = _truth()
    a = (t < 20) | (t > 35)
    b = (t > 15) & (t < 45)
    A = F.Obs(
        t[a],
        s[a] + rng.normal(0, 0.5, a.sum()),
        lat[a] + rng.normal(0, 0.3, a.sum()),
        0.5,
        0.3,
        "render",
        horse=1,
        tid=1,
    )
    B = F.Obs(
        t[b],
        s[b] + rng.normal(0, 1, b.sum()),
        lat[b] + rng.normal(0, 0.6, b.sum()),
        1.0,
        0.6,
        "drone",
        tid=2,
    )
    C = F.Obs(
        t[b],
        s[b] + 5 + rng.normal(0, 1, b.sum()),
        lat[b] + 2 + rng.normal(0, 0.6, b.sum()),
        1.0,
        0.6,
        "drone",
        horse=2,
        tid=3,
    )
    paths, assign = F.fuse_race([A, B, C], [1, 2], F.Prior(width=22.5))
    assert assign[2] == 1  # anonymous drone tracklet -> horse 1
    p = paths[1]
    assert np.sqrt(np.mean((np.interp(t, p.t, p.s) - s) ** 2)) < 0.3
    assert np.sqrt(np.mean((np.interp(t, p.t, p.lat) - lat) ** 2)) < 0.15
    assert p.feeds == {"render": a.sum(), "drone": b.sum()}


def test_lateral_bound_enforced():
    t = np.arange(0, 10, 0.1)
    o = F.Obs(t, 17 * t, np.full(len(t), -0.5), 0.5, 0.3, "render", horse=1, tid=1)
    s, _ss, lat, _sl, _c = F.smooth([o], F.Prior(width=20.0, l_min=0.3), t)
    assert lat.min() >= 0.3 - 1e-6


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

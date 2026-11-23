import numpy as np
import pytest

from hkjc.recon import pose, rails, register


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    A = rng.uniform(0, 900, (200, 2))
    S = register.sim_matrix(1.03, 0.2, 30.0, -12.0)
    B = A @ S[:2, :2].T + S[:2, 2]
    assert np.allclose(register.umeyama(A, B), S, atol=1e-9)


def test_pose_graph_closes_a_loop():
    K = 40
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    G_true = np.stack(
        [register.sim_matrix(1.0, a, 1000 * np.cos(a), 1000 * np.sin(a)) for a in ang]
    )
    rel = lambda i, j: np.linalg.inv(G_true[j]) @ G_true[i]  # noqa: E731
    rng = np.random.default_rng(2)
    noisy = [np.eye(3)] + [
        rel(k - 1, k) @ register.sim_matrix(1.0, rng.normal(0, 2e-3), *rng.normal(0, 1.0, 2))
        for k in range(1, K)
    ]
    G0 = pose.chain(noisy)
    G0 = np.einsum("ij,kjl->kil", G_true[0], G0)
    # several loop closures (a real revisit yields dozens; one lone edge is Huber-downweighted)
    edges = [pose.Edge(k - 1, k, noisy[k]) for k in range(1, K)] + [
        pose.Edge(K - 1 - j, i, rel(K - 1 - j, i), 3.0, "loop") for j in range(4) for i in range(4)
    ]
    pg0 = pose.PoseGraph((535, 945), G0, edges)
    pg = pose.solve_linear(pg0)
    loop = np.array([e.kind == "loop" for e in edges])
    assert pose.edge_residuals(pg0)[loop].mean() > 10 * pose.edge_residuals(pg)[loop].mean()
    err = lambda G: np.abs(G[:, :2, 2] - G_true[:, :2, 2]).mean()  # noqa: E731
    assert err(pg.G) < 0.8 * err(G0)


def test_railframe_projection_on_circle():
    th = np.linspace(0, np.pi, 2000)
    R = np.c_[1000 * np.cos(th), 1000 * np.sin(th)]  # counter-clockwise in y-down image
    rf = rails.RailFrame.from_points(R, inner=-1)
    X = np.array([[1100 * np.cos(1.0), 1100 * np.sin(1.0)]])
    s, lat = rf.project(X)
    assert abs(s[0] - 1000 * 1.0) < 3 and abs(abs(lat[0]) - 100) < 1


def test_marker_grid_fit():
    ppm, sF = 18.5, 30000.0
    ss = np.arange(20000, 31000, 1.0)
    resp = np.zeros_like(ss)
    for k in (0, 1, 2, 3, 4, 6):
        resp[np.abs(ss - (sF - k * 50 * ppm)) < 2] = 1.0
    fit = rails.fit_marker_grid(ss, resp)
    assert fit is not None and abs(fit[0] - ppm) < 0.02 and abs(fit[1] - sF) < 2


def test_odometer_tick_calibration_recovers_scale_drift():
    ppm0, sF = 18.0, 30000.0
    s = np.linspace(1000, sF, 900)
    d_true = (sF - s) / ppm0 * (1 + 0.01 * np.sin((s - 1000) / 5000))  # smooth ~1% scale drift
    odo = np.ceil(d_true / 10) * 10  # round-up display
    # wrong marker guesses (ppm off 15%, finish off 400 px) must not matter: ticks carry the metric
    d_of_s, c, _ppm, _r, (ppm1, sF1, _pin, _cm, _nsh) = rails.calibrate_distance_ticks(
        np.arange(len(s)), s, odo, 1.15 * ppm0, sF + 400
    )
    assert abs(float(d_of_s(np.array([sF]))[0])) < 3.0
    assert np.abs(d_of_s(s) - d_true).max() < 3.0


def _rail_map(lines, h=200, w=2000):
    """Synthetic mosaic: turf with horizontal white rail lines; ``lines`` = [(y, brightness(x))]."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (60, 120, 60)
    x = np.arange(w)
    for y, bright in lines:
        img[y - 1 : y + 2, :] = np.repeat(bright(x)[None, :, None], 3, axis=2)
    return img, np.ones((h, w), np.float32), (lambda W: np.asarray(W, float))


def test_refine_rail_keeps_the_dp_line_between_parallel_rails():
    pytest.importorskip("cv2")
    # field side y=100 (rail), the render's fixed rail 16 px inside (y=116, +y inward); the brighter
    # one alternates every 300 px, so the strongest-ridge snap switches lines and the nearest does not
    alt = lambda x: np.where((x // 300) % 2 == 0, 255, 200)  # noqa: E731
    img, cnt, to_px = _rail_map([(100, alt), (116, lambda x: 455 - alt(x))])
    Rail = np.c_[np.arange(100, 1900, 2.0), np.full(900, 101.0)]  # coarse rail (DP) 1 px off the field-side line
    ok = np.ones(len(Rail), bool)
    near, _ = rails.refine_rail(img, cnt, to_px, Rail, 1, ok, nearest=True)
    assert np.abs(near[:, 1] - 100).max() < 1.0
    strong, _ = rails.refine_rail(img, cnt, to_px, Rail, 1, ok)
    assert np.abs(strong[:, 1] - 116).min() < 1.0  # the old snap does switch


def test_refine_rail_nearest_equals_strongest_on_a_single_rail():
    pytest.importorskip("cv2")
    img, cnt, to_px = _rail_map([(100, lambda x: np.full(len(x), 230))])
    Rail = np.c_[np.arange(100, 1900, 2.0), 103.0 + 2.0 * np.sin(np.arange(900) / 50)]
    ok = np.ones(len(Rail), bool)
    a = rails.refine_rail(img, cnt, to_px, Rail, 1, ok, nearest=True)
    b = rails.refine_rail(img, cnt, to_px, Rail, 1, ok)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.abs(a[0][:, 1] - 100).max() < 0.5

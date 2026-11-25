import numpy as np

from hkjc.recon import appearance as A


def _app(n, horses, read, p=0.9):
    """n detections, each reading runner ``read`` with probability p (rest uniform)."""
    H = len(horses)
    q = np.full((n, H), (1.0 - p) / (H - 1))
    q[:, horses.index(read)] = p
    return dict(logp=np.log(q).astype(np.float32), ok=np.ones(n, bool), nos=np.array(horses),
                has_gif=np.ones(H, bool))


def test_cost_is_a_likelihood_ratio_against_the_best_matching_runner():
    horses = [1, 2, 3, 4]
    app = _app(20, horses, 3)
    seg = np.zeros(20, int)
    C = A.costs(app, np.arange(20), seg, 1, horses, np.full(20, 0.97), [np.arange(20)])
    assert abs(C[0, 2]) < 1e-9 and C[0, 0] > 0  # appearance never makes labelling a box cheaper than clutter
    q = np.exp(-C[0] / 2)
    q /= q.sum()  # the tracklet's runner distribution
    assert int(np.argmax(q)) == 2 and q.max() <= 1.0 - A.EPS + A.EPS / len(horses) + 1e-9
    assert C.max() <= 2 * np.log((1 - A.EPS + A.EPS / 4) / (A.EPS / 4)) + 1e-9


def test_repeated_detections_of_one_tracklet_saturate():
    horses = [1, 2, 3, 4]
    one = A.costs(_app(10, horses, 3), np.arange(10), np.zeros(10, int), 1, horses, np.full(10, 0.97),
                  [np.arange(10)])
    many = A.costs(_app(1000, horses, 3), np.arange(1000), np.zeros(1000, int), 1, horses, np.full(1000, 0.97),
                   [np.arange(1000)])
    assert many[0, 2] - one[0, 2] > -1.0  # 100x the detections add about nothing
    # the same 1000 detections cut into segments of one tracklet: one label over all costs what the tracklet costs
    seg = np.minimum(np.arange(1000) // 150, 5)
    cut = A.costs(_app(1000, horses, 3), np.arange(1000), seg, 6, horses, np.full(1000, 0.97), [np.arange(1000)])
    assert np.allclose(cut.sum(0), many[0])


def test_segments_of_a_tracklet_that_changes_avatar_keep_their_own_reads():
    horses = [1, 2, 3, 4]
    app = _app(400, horses, 3)
    app["logp"][200:] = _app(200, horses, 1)["logp"]
    seg = (np.arange(400) >= 200).astype(int)
    C = A.costs(app, np.arange(400), seg, 2, horses, np.full(400, 0.97), [np.arange(400)])
    assert int(np.argmin(C[0])) == 2 and int(np.argmin(C[1])) == 0


def test_a_box_that_is_not_a_horse_or_unscored_says_nothing():
    horses = [1, 2, 3, 4]
    app = _app(20, horses, 3)
    C = A.costs(app, np.arange(20), np.zeros(20, int), 1, horses, np.zeros(20), [np.arange(20)])
    assert np.allclose(C, 0.0)
    app["ok"][:] = False
    C = A.costs(app, np.arange(20), np.zeros(20, int), 1, horses, np.full(20, 0.97), [np.arange(20)])
    assert np.allclose(C, 0.0)


def test_runner_columns_follow_the_cache_numbers_and_rows_follow_det_idx():
    horses = [2, 5, 7]
    app = _app(30, [7, 2, 5], 5)  # cache columns in another order
    app["logp"][:10] = np.log(np.array([0.9, 0.05, 0.05]))  # rows 0-9 read #7
    C = A.costs(app, np.arange(10, 30), np.zeros(20, int), 1, horses, np.full(20, 0.97), [np.arange(20)])
    assert int(np.argmin(C[0])) == horses.index(5)
    C = A.costs(app, np.arange(0, 10), np.zeros(10, int), 1, horses, np.full(10, 0.97), [np.arange(10)])
    assert int(np.argmin(C[0])) == horses.index(7)

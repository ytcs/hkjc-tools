"""JAX traffic kernel — matches the numpy model in distribution; grid behaves.

Runs only with the bayes stack: ``uv run --group bayes pytest``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

from hkjc.model.traffic import TrafficRaceModel  # noqa: E402
from hkjc.model.traffic_jax import sample_orders_jax, win_prob_grid  # noqa: E402
from hkjc.track import HAPPY_VALLEY  # noqa: E402


def _pwin(orders, n):
    return np.array([(orders[:, 0] == i).mean() for i in range(n)])


def test_jax_matches_numpy_draw_curve():
    n = 12
    v_c, w, draw = np.full(n, 16.0), np.full(n, 60.0), np.arange(1, n + 1)
    onp = TrafficRaceModel(v_c, w, draw, 1200, track=HAPPY_VALLEY).sample_orders(
        5000, np.random.default_rng(0))
    oj = sample_orders_jax(v_c, w, draw, 1200, track=HAPPY_VALLEY, size=5000, seed=0)
    pn, pj = _pwin(onp, n), _pwin(oj, n)
    assert pj.argmax() == 0                                  # inner draw favoured
    assert np.abs(pn - pj).max() < 0.03                      # same distribution


def test_win_prob_grid_responds_to_bend():
    n = 10
    v_c, w, draw = np.full(n, 16.0), np.full(n, 55.0), np.arange(1, n + 1)
    grid = np.linspace(0.001, 0.04, 6)
    p_inner = win_prob_grid(v_c, w, draw, 1200, 0, grid, track=HAPPY_VALLEY, size=1500)
    p_outer = win_prob_grid(v_c, w, draw, 1200, n - 1, grid, track=HAPPY_VALLEY, size=1500)
    # stronger bend -> inner draw wins more, outer draw wins less
    assert p_inner[-1] > p_inner[0]
    assert p_outer[-1] < p_outer[0]

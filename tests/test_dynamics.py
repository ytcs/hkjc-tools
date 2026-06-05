"""Tests for the power-balance equation-of-motion core (mechanistic successor).

The EOM integrates under JAX (the single canonical implementation), so these run only
with the bayes stack: ``uv run --group bayes pytest``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

from hkjc.model.dynamics import G, K_LOCO, DynamicsRaceModel  # noqa: E402
from hkjc.core.simulation import RaceSimulator  # noqa: E402
from hkjc.core.track import SHA_TIN_TURF  # noqa: E402


def _rng():
    return np.random.default_rng(0)


def test_reduces_to_kinematic_with_no_drag():
    # The acceptance test: drag off + even pace ⇒ T = (D − w_prime)/v_c (the old core).
    n = 6
    v_c = np.array([16.0, 16.2, 15.8, 16.5, 15.5, 16.1])
    w = np.full(n, 55.0)
    dist = 1600.0
    m = DynamicsRaceModel.from_kinematic(v_c, w, np.full(n, 500.0), np.arange(1, n + 1), dist,
                                         track=SHA_TIN_TURF, cda=0.0, speed_noise=0.0,
                                         bend_curvature=0.0, draft=0.0)
    ft = m.finish_times(1, _rng())[0]
    assert np.abs(ft - (dist - w) / v_c).max() / ((dist - w) / v_c).mean() < 0.01


def test_mass_slows_at_fixed_physiology():
    # Same intrinsic aerobic power P_c, more mass ⇒ lower v_c = P_c/(K_LOCO·m·g) ⇒ slower.
    # This is the mechanistic weight coupling — mass is a term, not a tacked-on penalty.
    p_c = K_LOCO * 500.0 * G * 16.0
    w = K_LOCO * 500.0 * G * 55.0
    m = DynamicsRaceModel(p_c=[p_c, p_c], w_reserve=[w, w], mass=[470.0, 560.0],
                          draw=[1, 2], distance=1600.0, track=SHA_TIN_TURF, cda=0.0,
                          speed_noise=0.0, bend_curvature=0.0, draft=0.0)
    ft = m.finish_times(1, _rng())[0]
    assert m.v_c[0] > m.v_c[1]          # lighter sustains a higher velocity
    assert ft[1] > ft[0]                # heavier finishes later


def test_anaerobic_reserve_speeds_up_then_bounds():
    dist = 1600.0
    none = DynamicsRaceModel.from_kinematic([16.0], [0.0], [500.0], [1], dist,
                                            cda=0.0, speed_noise=0.0).finish_times(1, _rng())[0, 0]
    some = DynamicsRaceModel.from_kinematic([16.0], [80.0], [500.0], [1], dist,
                                            cda=0.0, speed_noise=0.0).finish_times(1, _rng())[0, 0]
    assert abs(none - dist / 16.0) < 1.0   # no reserve ⇒ runs at the aerobic critical velocity
    assert some < none                  # spending W′ above P_c is faster


def test_drag_slows_and_is_neutral_at_zero():
    dist = 1600.0
    lo = DynamicsRaceModel.from_kinematic([16.0], [55.0], [500.0], [1], dist,
                                          cda=0.0, speed_noise=0.0).finish_times(1, _rng())[0, 0]
    hi = DynamicsRaceModel.from_kinematic([16.0], [55.0], [500.0], [1], dist,
                                          cda=1.5, speed_noise=0.0).finish_times(1, _rng())[0, 0]
    assert hi > lo


def test_draw_bias_emerges_and_orders_are_permutations():
    n = 12
    m = DynamicsRaceModel.from_kinematic(np.full(n, 16.0), np.full(n, 55.0), np.full(n, 500.0),
                                         np.arange(1, n + 1), 1200.0, track=SHA_TIN_TURF, course="A")
    sim = RaceSimulator(m, n_samples=4000)
    pw = np.array([sim.prob(sim.wins(i)) for i in range(n)])
    assert abs(pw.sum() - 1.0) < 1e-9
    assert pw[:3].mean() > pw[-3:].mean()     # inner-draw advantage emerges from the geometry
    # every row is a valid finishing-order permutation
    assert (np.sort(m.sample_orders(50, _rng()), axis=1) == np.arange(n)).all()

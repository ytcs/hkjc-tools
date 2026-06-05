"""Tests for the Tier-A interacting (traffic-flow) race model."""
import numpy as np

from hkjc.model.traffic import TrafficRaceModel
from hkjc.core.simulation import RaceSimulator


def test_draw_bias_emerges_from_geometry():
    n = 12
    m = TrafficRaceModel(np.full(n, 16.0), np.full(n, 60.0), np.arange(1, n + 1),
                         distance=1200, course="A")
    sim = RaceSimulator(m, n_samples=8000)
    pw = np.array([sim.prob(sim.wins(i)) for i in range(n)])
    assert abs(pw.sum() - 1.0) < 1e-9
    # equal ability: the inner-draw advantage emerges from the bend geometry alone
    assert pw.argmax() == 0
    assert pw[:3].mean() > 3 * pw[-3:].mean()


def test_ability_overcomes_draw():
    n = 8
    v_c = np.full(n, 15.5)
    v_c[7] = 17.0                       # a much faster horse drawn widest
    m = TrafficRaceModel(v_c, np.full(n, 60.0), np.arange(1, n + 1), distance=1200, course="A")
    sim = RaceSimulator(m, n_samples=8000)
    pw = np.array([sim.prob(sim.wins(i)) for i in range(n)])
    assert pw[7] > pw[3]                # superior engine beats a mid draw despite the wide gate

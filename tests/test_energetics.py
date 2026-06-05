"""Tests for the critical-velocity / W' energetics estimation."""
from hkjc.model.energetics import estimate_run_energetics, section_distances


def test_section_distances_convention():
    assert list(section_distances(1800, 5)) == [200, 400, 400, 400, 400]
    assert list(section_distances(1650, 4)) == [450, 400, 400, 400]
    assert list(section_distances(1200, 3)) == [400, 400, 400]


def test_estimate_run_energetics_profile():
    times = [13.74, 22.76, 23.78, 24.01, 25.14]            # the real 1800m winner
    e = estimate_run_energetics(times, section_distances(1800, 5))
    # velocities ~ 14.56, 17.57, 16.82, 16.66, 15.91
    assert 15.5 < e.v_c < 16.0           # settles to ~15.9 (slower of last two)
    assert e.v_peak > 17.0               # anaerobic peak ~17.6
    assert e.w_prime > 0                 # spent work above critical
    assert 16.0 < e.v_mean < 16.7        # 1800/109.43 = 16.45
    assert e.n_sections == 5


def test_energetics_race_model_ranks_by_predicted_time():
    import numpy as np

    from hkjc.model.energetics import EnergeticsRaceModel
    from hkjc.core.simulation import RaceSimulator

    v_c = np.array([16.5, 16.0, 15.5])      # A fastest sustainable
    w_prime = np.array([100.0, 80.0, 60.0])
    m = EnergeticsRaceModel(v_c, w_prime, distance=1600, time_noise=3.0)
    # T = (1600 - W')/v_c -> 90.9, 95.0, 99.4  (A < B < C)
    assert m.pred_time[0] < m.pred_time[1] < m.pred_time[2]
    sim = RaceSimulator(m, n_samples=60000)
    pw = [sim.prob(sim.wins(i)) for i in range(3)]
    assert pw[0] > pw[1] > pw[2] and abs(sum(pw) - 1.0) < 1e-9

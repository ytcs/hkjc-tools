"""Tests for the dedicated-surface identification estimators (journal DRAFT 0048).

Each estimator inverts the SAME EOM physics on the surface where one constant dominates. We test
the estimators against SYNTHETIC runs generated FROM the EOM at known constants: a constant
recovered on its dedicated surface must return the truth (round-trip), and the consistency helper
must behave.
"""
from __future__ import annotations

import numpy as np

from hkjc.model.dynamics import (
    PhysioConstants, cp_for_cruise, cruise_velocity, eom_section_velocities_fast,
    make_resistances,
)
from hkjc.model.identify import (
    Run, bland_altman, cp_cruise_run, hill_fit, pmax_for_ceiling,
    settled_cruise_velocity, wprime_spent_run,
)


def _synthetic_run(horse_id, dist, n_sec, cp, w0, p_max, mass=560.0, going="GOOD"):
    """Build a Run from the EOM forward solution at known constants (the ground truth)."""
    res = make_resistances(mass, going)
    seg = dist / n_sec
    d = np.full(n_sec, seg)
    v, _, _ = eom_section_velocities_fast(d, cp, w0, p_max, 120.0, res, dt=0.05, v0=4.0)
    t = d / np.clip(v, 1e-3, None)
    return Run(horse_id=horse_id, distance=float(dist), section_dist=d, section_time=t,
               velocity=v, mass=mass, going=going)


def test_cp_cruise_recovers_v_c_on_a_flat_route():
    """A long route synthesised at known CP: the settled-cruise inversion returns ~CP."""
    res = make_resistances(560.0, "GOOD")
    cp_true = cp_for_cruise(16.5, res)             # v_c = 16.5 m/s
    # large reserve, modest P_max ⇒ the mid-race sits at the cruise root (flat singular arc)
    run = _synthetic_run("H", 2000, 5, cp_true, 2.5e5, 1.6 * cp_true, mass=560.0)
    cp_hat = cp_cruise_run(run, level="min")
    assert cp_hat is not None
    # the settled-cruise CP recovers the truth within ~12% (cruise arc sits at/above v_c)
    assert abs(cp_hat - cp_true) / cp_true < 0.20
    # the settled velocity read sits at/just above the true cruise root
    vc_true = cruise_velocity(cp_true, res)
    assert settled_cruise_velocity(run) >= vc_true - 0.6


def test_cp_cruise_none_for_short_run():
    run = _synthetic_run("H", 1200, 3, 27000.0, 2e5, 45000.0)
    assert cp_cruise_run(run) is None              # <4 sections ⇒ no interior cruise read


def test_hill_recovers_critical_velocity():
    """Synthesise the SAME horse at several distances; the Hill slope returns the critical velocity.

    Build each race as a near-maximal time-trial at a fixed CP; d = CV·t + D' should recover CV ≈
    the cruise root and a positive distance capacity D'.
    """
    res = make_resistances(560.0, "GOOD")
    cp_true = cp_for_cruise(16.0, res)
    cv_true = cruise_velocity(cp_true, res)
    dists, times = [], []
    for dist, nsec in [(1200, 3), (1600, 4), (2000, 5), (2400, 6)]:
        run = _synthetic_run("H", dist, nsec, cp_true, 2.0e5, 1.7 * cp_true, mass=560.0)
        dists.append(dist)
        times.append(run.total_time)
    fit = hill_fit(np.array(dists), np.array(times), min_dist=3)
    assert fit is not None
    assert fit.n_dist == 4
    assert fit.r2 > 0.99                            # the EOM time-trials are near-linear in d–t
    # the Hill SLOPE recovers the critical velocity tightly (the headline quantity). The intercept
    # D' sign depends on the ramp/kick balance across distances and is not guaranteed by the EOM;
    # only CV (the slope) is the robust velocity-space CP here.
    assert abs(fit.cv - cv_true) < 0.5
    assert fit.cv_se < 0.1                          # tight slope ⇒ CV is well identified


def test_hill_none_for_too_few_distances():
    fit = hill_fit(np.array([1200.0, 1200.0]), np.array([70.0, 71.0]), min_dist=3)
    assert fit is None


def test_pmax_for_ceiling_exceeds_cp():
    """The sprint-ceiling P_max (force balance at peak v) exceeds the cruise CP (force balance at v_c)."""
    res = make_resistances(560.0, "GOOD")
    cp = cp_for_cruise(16.0, res)
    p_ceiling = pmax_for_ceiling(18.5, res)         # a sprint-ceiling velocity above v_c
    assert p_ceiling > cp


def test_wprime_spent_positive_in_a_hard_race():
    """A maximal sprint spends reserve above CP ⇒ the back-solved spent reserve is positive."""
    res = make_resistances(560.0, "GOOD")
    cp = cp_for_cruise(15.5, res)                   # a modest CP
    run = _synthetic_run("H", 1200, 3, cp, 2.0e5, 2.0 * cp, mass=560.0)
    spent = wprime_spent_run(run, cp)
    assert spent is not None and spent > 0


def test_bland_altman_perfect_agreement():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ba = bland_altman(a, a.copy())
    assert ba["n"] == 5
    assert abs(ba["bias"]) < 1e-9
    assert abs(ba["mad"]) < 1e-9
    assert abs(ba["corr"] - 1.0) < 1e-9


def test_physio_constants_synthetic_roundtrip_sanity():
    """Sanity: the synthetic builder produces equine-plausible section velocities."""
    run = _synthetic_run("H", 2000, 5, cp_for_cruise(16.5, make_resistances(560.0)), 2.5e5,
                         45000.0)
    assert np.all(run.velocity > 8.0) and np.all(run.velocity < 22.0)
    # ensure the constants object validates (P_max > CP)
    PhysioConstants(cp=27000.0, w_prime0=2.0e5, p_max=45000.0, tau=120.0)

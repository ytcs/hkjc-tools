"""Unit tests for the first-principles EOM generative core (``hkjc.model.dynamics``).

These pin the PHYSICS, not a fit: the steady-state cruise velocity is the force-balance root,
the reserve W′ is conserved / non-negative, and the bang–CP–bang policy degenerates sanely
(no kick when the tank cannot refill; pure cruise at v_c). No regression, no fitted constants.
"""
from __future__ import annotations

import numpy as np
import pytest

from hkjc.model.dynamics import (
    ETA, PhysioConstants, body_mass_kg, cp_for_cruise, cruise_velocity,
    eom_section_velocities_fast, expected_profile_eom, make_resistances, section_velocities,
    solve_eom,
)


def _res():
    m = body_mass_kg(1122, 125)          # median ST horse + carried weight
    return make_resistances(m, "GOOD")


def test_cruise_root_is_force_balance():
    """v_c must be the ROOT of η·CP = ½ρC_dA·v³ + C_rr·m·g·v (dv/dt = 0 at P=CP, v=v_c)."""
    res = _res()
    cp = cp_for_cruise(16.5, res)
    v_c = cruise_velocity(cp, res)
    assert v_c == pytest.approx(16.5, abs=1e-3)
    # the residual of the power balance at the root is ~0
    assert float(res.resist_power(v_c)) == pytest.approx(ETA * cp, rel=1e-4)
    # and dv/dt = (η·CP/v_c − F_resist)/m ≈ 0
    dvdt = (ETA * cp / v_c - float(res.resist_force(v_c))) / res.m
    assert abs(dvdt) < 1e-6


def test_cp_vc_roundtrip_monotone():
    """CP↔v_c is an exact invertible bijection (monotone cubic); higher CP ⇒ higher v_c."""
    res = _res()
    for vc in (15.0, 16.0, 17.0):
        assert cruise_velocity(cp_for_cruise(vc, res), res) == pytest.approx(vc, abs=1e-3)
    assert cp_for_cruise(17.0, res) > cp_for_cruise(15.0, res)


def test_physio_bounds_enforced():
    """The physical bounds (CP>0, W′_0>0, P_max>CP, τ>0) are hard."""
    with pytest.raises(ValueError):
        PhysioConstants(cp=1000.0, w_prime0=1e5, p_max=500.0)   # P_max < CP
    with pytest.raises(ValueError):
        PhysioConstants(cp=1000.0, w_prime0=-1.0, p_max=2000.0)  # W′_0 < 0


def test_wprime_nonnegative_and_conserved():
    """Reserve stays in [0, W′_0] throughout, and the total anaerobic spend ≤ W′_0."""
    res = _res()
    cp = cp_for_cruise(15.5, res)
    phys = PhysioConstants(cp=cp, w_prime0=4.0e5, p_max=1.7 * cp, tau=120.0)
    sol = solve_eom(phys, res, 1600.0, v0=4.0)
    assert sol.w_prime.min() >= -1e-6
    assert sol.w_prime.max() <= phys.w_prime0 + 1e-6
    spent = phys.w_prime0 - float(sol.w_prime[-1])
    assert -1e-6 <= spent <= phys.w_prime0 + 1e-6


def test_kick_empties_tank_at_line():
    """When a terminal kick fires, the policy times it so W′ ≈ 0 at the finish (full extension)."""
    res = _res()
    cp = cp_for_cruise(15.5, res)
    phys = PhysioConstants(cp=cp, w_prime0=4.0e5, p_max=1.7 * cp, tau=120.0)
    sol = solve_eom(phys, res, 1600.0, v0=4.0)
    assert sol.t_burn < 1600.0                       # a kick actually fired
    assert float(sol.w_prime[-1]) == pytest.approx(0.0, abs=1.0)   # tank empty at the line
    sv = section_velocities(sol, np.array([400.0, 400, 400, 400]))
    assert sv[-1] > sol.v_c                           # the close sits ABOVE bare v_c (the kick)


def test_no_kick_when_ramp_exhausts_reserve():
    """A small reserve fully spent reaching cruise ⇒ no kick; the close fades to bare v_c."""
    res = _res()
    cp = cp_for_cruise(16.5, res)
    phys = PhysioConstants(cp=cp, w_prime0=5.0e4, p_max=1.8 * cp, tau=120.0)  # tiny reserve
    sol = solve_eom(phys, res, 1600.0, v0=4.0)
    sv = section_velocities(sol, np.array([400.0, 400, 400, 400]))
    assert sol.t_burn == pytest.approx(1600.0, abs=1.0)   # no terminal kick
    assert sv[-1] == pytest.approx(sol.v_c, abs=0.05)     # close ≈ v_c (pinned by exhaustion)


def test_cruise_arc_sits_at_vc():
    """The singular cruise arc holds v at the force-balance root v_c (P=CP equilibrium)."""
    res = _res()
    cp = cp_for_cruise(16.0, res)
    phys = PhysioConstants(cp=cp, w_prime0=2.5e5, p_max=1.7 * cp, tau=120.0)
    sol = solve_eom(phys, res, 1600.0, v0=4.0)
    sv = section_velocities(sol, np.array([400.0, 400, 400, 400]))
    # the middle (cruise) sections sit at v_c to within a small tolerance
    assert sv[1] == pytest.approx(sol.v_c, abs=0.1)


def test_softer_going_slows_cruise():
    """Softer going raises C_rr ⇒ a lower cruise root at fixed CP (surface physics, not a fit)."""
    m = body_mass_kg(1122, 125)
    cp = cp_for_cruise(16.5, make_resistances(m, "GOOD"))
    v_good = cruise_velocity(cp, make_resistances(m, "GOOD"))
    v_soft = cruise_velocity(cp, make_resistances(m, "SOFT"))
    assert v_soft < v_good


def test_fast_core_matches_reference():
    """The numba fast core reproduces the pure-Python reference integrator (same physics)."""
    res = _res()
    cp = cp_for_cruise(15.5, res)
    d = np.array([400.0, 400, 400, 400])
    sv_fast, v_c, _ = eom_section_velocities_fast(d, cp, 4.0e5, 1.7 * cp, 120.0, res,
                                                  dt=0.02, v0=4.0)
    phys = PhysioConstants(cp=cp, w_prime0=4.0e5, p_max=1.7 * cp, tau=120.0)
    sol = solve_eom(phys, res, 1600.0, dt=0.02, v0=4.0)
    sv_ref = section_velocities(sol, d)
    assert np.max(np.abs(sv_fast - sv_ref)) < 0.01
    assert v_c == pytest.approx(sol.v_c, abs=1e-3)


def test_expected_profile_shape():
    """The generative profile returns one velocity per section, all positive and finite."""
    res = _res()
    cp = cp_for_cruise(16.0, res)
    phys = PhysioConstants(cp=cp, w_prime0=3.0e5, p_max=1.7 * cp, tau=120.0)
    prof = expected_profile_eom(np.array([400.0, 400, 400, 400]), phys, res)
    assert prof.shape == (4,)
    assert np.all(prof > 0) and np.all(np.isfinite(prof))

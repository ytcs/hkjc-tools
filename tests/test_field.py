"""Unit tests for the mean-field pace/draft coupling (``hkjc.model.field``, journal 0049).

These pin the PHYSICS of the coupling, not a fit:
  * draft reduces the effective aero drag by a physically-bounded fraction (leader gets 0);
  * pace-following REDUCES TO the isolated EOM when the field pace ≤ the horse's own v_c (the
    required limit — no field pull below the sustainable level);
  * a HOT field pace lifts the cruise above v_c (the horse runs on reserve), spending W′;
  * the draft de-confounding LOWERS a drafted cruise read toward the clean-air aerobic v_c, and
    is the identity when there is no cover.
No regression, no fitted constants, no market/win/pool.
"""
from __future__ import annotations

import numpy as np
import pytest

from hkjc.model.dynamics import (
    PhysioConstants, body_mass_kg, cp_for_cruise, cruise_velocity, make_resistances,
)
from hkjc.model.field import (
    DRAFT_FRAC, FRONT_STYLE_CUTOFF, FRONT_WAR_THRESHOLD, cover_fraction,
    deconfounded_cruise_velocity, draft_aero_factor, equilibrium_pace, field_section_velocities,
    front_count, front_group_pace, front_war_gate, pace_war_burn_per_runner, pace_war_gradient,
    solve_field_coupled, v_max_at_pmax,
)


def _res():
    return make_resistances(body_mass_kg(1122, 125), "GOOD")


def _phys(v_c=16.0, w0=2.0e5, pr=1.9):
    res = _res()
    cp = cp_for_cruise(v_c, res)
    return PhysioConstants(cp=cp, w_prime0=w0, p_max=pr * cp, tau=120.0), res


def test_draft_factor_bounds():
    """The leader (cover 0) gets clear air (factor 1); a fully-covered follower gets 1−DRAFT_FRAC."""
    assert draft_aero_factor(0.0) == pytest.approx(1.0)
    assert draft_aero_factor(1.0) == pytest.approx(1.0 - DRAFT_FRAC)
    # monotone decreasing in cover, clamped
    assert draft_aero_factor(0.5) == pytest.approx(1.0 - 0.5 * DRAFT_FRAC)
    assert draft_aero_factor(2.0) == pytest.approx(1.0 - DRAFT_FRAC)   # clamp at 1


def test_cover_fraction_leader_zero():
    """Cover is 0 for the leader (rank 1) and 1 for the backmarker."""
    assert cover_fraction(1, 12) == pytest.approx(0.0)
    assert cover_fraction(12, 12) == pytest.approx(1.0)
    assert 0.0 < cover_fraction(6, 12) < 1.0


def test_v_max_above_v_c():
    """The pace-follow ceiling v_max(P_max) must exceed the sustainable v_c (P_max > CP)."""
    phys, res = _phys()
    v_c = cruise_velocity(phys.cp, res)
    assert v_max_at_pmax(phys, res) > v_c


def test_reduces_to_isolated_when_field_slow():
    """With v_field ≤ v_c everywhere and no cover, the coupled cruise must settle at v_c (isolated)."""
    phys, res = _phys(v_c=16.0)
    v_c = cruise_velocity(phys.cp, res)
    d = np.full(5, 400.0)
    v_field = np.full(5, v_c - 2.0)        # field slower than the horse's own sustainable level
    cover = np.zeros(5)                    # leader / clear air
    sv = field_section_velocities(phys, res, d, v_field, cover, dt=0.05)
    # the interior (settled) sections should sit at v_c (no field pull, no draft), not above
    assert sv[2] == pytest.approx(v_c, abs=0.25)
    assert sv[2] <= v_c + 0.25


def test_hot_field_lifts_cruise_above_vc():
    """A field pace ABOVE v_c pulls the settled cruise above v_c (the horse runs on reserve).

    Uses a realistic reserve (the small default empties in the opening ramp); the EARLY cruise is
    lifted to the field pace, then fades back through the close (the early-burn / late-fade hump).
    """
    phys, res = _phys(v_c=16.0, w0=6.0e5)
    v_c = cruise_velocity(phys.cp, res)
    d = np.full(5, 400.0)
    v_field = np.full(5, v_c + 1.0)        # hot field
    cover = np.zeros(5)
    sv = field_section_velocities(phys, res, d, v_field, cover, dt=0.05)
    assert sv[1] > v_c + 0.2               # early cruise lifted above the sustainable level


def test_pace_following_spends_reserve():
    """Holding above v_c on the field draws down W′ (the spent-pace reserve is positive)."""
    phys, res = _phys(v_c=16.0, w0=6.0e5)
    v_c = cruise_velocity(phys.cp, res)
    d = np.full(5, 400.0)
    sol_hot = solve_field_coupled(phys, res, float(d.sum()), d,
                                  np.full(5, v_c + 1.0), np.zeros(5), dt=0.05)
    sol_cold = solve_field_coupled(phys, res, float(d.sum()), d,
                                   np.full(5, v_c - 2.0), np.zeros(5), dt=0.05)
    assert sol_hot.w_spent_pace > sol_cold.w_spent_pace
    assert sol_cold.w_spent_pace == pytest.approx(0.0, abs=1.0)   # no pull below v_c


def test_draft_lifts_cruise():
    """Drafting (reduced drag) lets the same CP hold a HIGHER cruise than in clean air."""
    phys, res = _phys(v_c=16.0)
    v_c = cruise_velocity(phys.cp, res)
    d = np.full(5, 400.0)
    v_field = np.full(5, v_c - 2.0)        # slow field so pace-following is OFF — isolate draft
    sv_clear = field_section_velocities(phys, res, d, v_field, np.zeros(5), dt=0.05)
    sv_draft = field_section_velocities(phys, res, d, v_field, np.full(5, 1.0), dt=0.05)
    assert sv_draft[2] > sv_clear[2]       # covered horse cruises faster on the same CP


def test_deconfound_identity_no_cover():
    """With no cover the de-confounding is the identity (clean air already)."""
    res = _res()
    assert deconfounded_cruise_velocity(16.5, 0.0, res) == pytest.approx(16.5, abs=1e-3)


def test_deconfound_lowers_drafted_read():
    """A drafted cruise read (cover>0) de-confounds DOWN to the lower clean-air aerobic v_c."""
    res = _res()
    v_obs = 16.5
    v_dec = deconfounded_cruise_velocity(v_obs, 1.0, res)
    assert v_dec < v_obs                   # the draft inflation is removed
    # the implied CP must be consistent: clean-air power at v_dec == drafted power at v_obs
    from hkjc.model.dynamics import Resistances
    draft = 1.0 - DRAFT_FRAC
    res_draft = Resistances(m=res.m, c_da=res.c_da * draft, c_rr=res.c_rr,
                            rho_air=res.rho_air, g=res.g)
    assert float(res.resist_power(v_dec)) == pytest.approx(
        float(res_draft.resist_power(v_obs)), rel=1e-3)


# ---------------------------------------------------------- the mean-field EQUILIBRIUM PACE (S7)
def test_equilibrium_reduces_to_mean_when_style_uniform():
    """Style-uniform field ⇒ the equilibrium is the plain field-mean of the bases (the null limit)."""
    base = np.array([15.0, 16.0, 17.0])
    style = np.full(3, 0.7)                 # everyone equally forward
    # front pace = weighted mean = plain mean = 16; closers below 16 pulled toward it, fast held
    assert equilibrium_pace(base, style) == pytest.approx(
        float(np.mean(base + 0.7 * np.maximum(16.0 - base, 0.0))))
    # and it lies in [mean(base), max(base)]
    assert np.mean(base) <= equilibrium_pace(base, style) <= np.max(base)


def test_equilibrium_hot_front_lifts_pace_above_mean():
    """When the FORWARD horses are the FAST horses, the equilibrium runs ABOVE the plain field-mean."""
    base = np.array([15.0, 15.5, 17.5])     # the fast horse (17.5) is the front-runner below
    style = np.array([0.2, 0.3, 1.0])       # forwardness concentrated on the fast horse
    naive = float(np.mean(base))
    v_eq = equilibrium_pace(base, style)
    assert v_eq > naive                     # the hot front pulls the closers up
    assert v_eq <= float(np.max(base))      # but never beyond the fastest base


def test_equilibrium_bounded_by_fastest_base():
    """The equilibrium can lift the field-mean toward a fast front but NEVER beyond the fastest base."""
    base = np.array([14.0, 17.0, 17.5])
    style = np.array([1.0, 0.2, 0.2])       # forwardness on the slow horse
    v_eq = equilibrium_pace(base, style)
    assert float(np.mean(base)) <= v_eq <= float(np.max(base))   # always in [mean, max] of base
    # explicit: front pace pulls the slow leader up; faster horses (above L) hold their base
    lead = front_group_pace(base, style)
    expect = float(np.mean(base + style * np.maximum(lead - base, 0.0)))
    assert v_eq == pytest.approx(expect)


def test_front_group_pace_weights_forward_horses():
    """The front-group pace is the forwardness-weighted mean (a lone fast front-runner dominates it)."""
    base = np.array([14.0, 18.0])
    style = np.array([0.1, 1.0])
    assert front_group_pace(base, style) == pytest.approx((0.1 * 14.0 + 1.0 * 18.0) / 1.1)
    # all-zero style ⇒ falls back to the plain mean
    assert front_group_pace(base, np.zeros(2)) == pytest.approx(16.0)


# ---------------------------------------------------------------- the pace-war W′-burn gradient
def test_front_war_gate_threshold_null_and_saturation():
    """The pace-war gate is 0 at/below the threshold (no war) and saturates toward 1 above it."""
    assert front_war_gate(FRONT_WAR_THRESHOLD) == 0.0          # at the threshold: no pace war
    assert front_war_gate(FRONT_WAR_THRESHOLD - 1.0) == 0.0    # below: a lone leader is uncontested
    g3 = front_war_gate(FRONT_WAR_THRESHOLD + 1.0)             # one over: x=1 → 1/(1+1)=0.5
    assert g3 == pytest.approx(0.5)
    g_big = front_war_gate(FRONT_WAR_THRESHOLD + 100.0)        # far above: saturates → ~1
    assert 0.9 < g_big < 1.0


def test_front_count_only_counts_genuine_front_runners():
    """Only horses with forwardness ≥ cutoff are counted as front-runners contesting the lead."""
    style = np.array([0.1, 0.2, FRONT_STYLE_CUTOFF, 0.9])      # two clear the cutoff
    assert front_count(style) == pytest.approx(2.0)
    assert front_count(np.zeros(5)) == 0.0                     # all closers ⇒ no front-runners


def test_pace_war_gradient_null_below_threshold():
    """≤ threshold genuine front-runners ⇒ no pace war ⇒ zero induced gradient (isolated limit)."""
    base = np.array([16.0, 16.5, 17.5, 15.0])
    style = np.array([0.9, 0.1, 0.1, 0.1])                     # ONE front-runner (< threshold=2)
    assert pace_war_gradient(base, style) == 0.0


def test_pace_war_gradient_null_when_front_not_held_above_base():
    """A contested front whose members are all the FASTEST ⇒ none held above its cruise ⇒ 0."""
    # three front-runners, but they are also the fastest horses → L = max(front base) ≤ each? No:
    # the fastest front-runner sets L; the others ARE below L, so this fires. To get the NULL we
    # need a single-speed front group (all equal) → L equals every front base → no over-spend.
    base = np.array([17.0, 17.0, 17.0, 14.0])
    style = np.array([0.9, 0.9, 0.9, 0.1])                     # 3 front-runners, identical pace
    assert pace_war_gradient(base, style) == pytest.approx(0.0)


def test_pace_war_gradient_fires_and_is_negative_when_front_contested():
    """A crowded front with a fast leader pulls slower front-runners above their cruise ⇒ burn < 0."""
    base = np.array([17.5, 16.0, 15.5, 14.0])                  # fast leader + two slower front-runners
    style = np.array([0.9, 0.9, 0.9, 0.1])                     # 3 genuine front-runners (> threshold)
    g = pace_war_gradient(base, style)
    assert g < 0.0                                             # an early over-spend perturbation
    # steeper when the front is faster (more horses dragged further above their own cruise)
    base_hot = np.array([19.0, 16.0, 15.5, 14.0])
    assert pace_war_gradient(base_hot, style) < g             # more burn ⇒ more negative


# ----------------------------------------------- the PER-RUNNER pace-war W′-burn (win-drift wiring)
def test_pace_war_burn_per_runner_null_below_threshold():
    """≤ threshold genuine front-runners ⇒ no pace war ⇒ every per-runner burn is 0."""
    base = np.array([16.0, 16.5, 17.5, 15.0])
    style = np.array([0.9, 0.1, 0.1, 0.1])                    # ONE front-runner (< threshold=2)
    burn = pace_war_burn_per_runner(base, style)
    assert burn.shape == base.shape
    assert np.allclose(burn, 0.0)


def test_pace_war_burn_per_runner_only_front_runners_held_up_burn():
    """Only contested front-runners held ABOVE their own cruise burn reserve; closers/leader = 0."""
    base = np.array([17.5, 16.0, 15.5, 14.0])                 # fast leader + two slower front-runners
    style = np.array([0.9, 0.9, 0.9, 0.1])                    # 3 genuine front-runners (> threshold)
    burn = pace_war_burn_per_runner(base, style)
    assert burn.shape == base.shape
    assert np.all(burn >= 0.0)
    assert burn[0] == pytest.approx(0.0)                      # the fastest front-runner sets L → not held up
    assert burn[1] > 0.0 and burn[2] > 0.0                    # slower front-runners burn reserve
    assert burn[3] == pytest.approx(0.0)                      # the closer is not in the war
    # a HOTTER front (faster leader) burns MORE reserve from the slower front-runners
    base_hot = np.array([19.0, 16.0, 15.5, 14.0])
    burn_hot = pace_war_burn_per_runner(base_hot, style)
    assert burn_hot[1] > burn[1] and burn_hot[2] > burn[2]


def test_pace_war_burn_per_runner_single_speed_front_null():
    """A single-speed front group (all equal) ⇒ none held above its cruise ⇒ zero burn."""
    base = np.array([17.0, 17.0, 17.0, 14.0])
    style = np.array([0.9, 0.9, 0.9, 0.1])                    # 3 front-runners, identical pace
    assert np.allclose(pace_war_burn_per_runner(base, style), 0.0)

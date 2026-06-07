"""MEAN-FIELD PACE / DRAFT COUPLING of the EOM (journal DRAFT 0049).

The single-horse EOM (:mod:`hkjc.model.dynamics`, journal 0047) integrates each runner in
ISOLATION under the bang–CP–bang optimal policy. Two independent signatures (journal 0047's
closing fade-hump + fast-pace closing residual; journal 0048's +1.4 m/s cruise-vs-Hill
critical-velocity gap) converge on the SAME missing physics: real horses cruise ABOVE their own
physiological critical velocity because the FIELD's pace pulls them along. This module adds that
coupling — the THEORY §III.4(a) mean-field game of pace — in its tractable, OBSERVED-pace form.

We do NOT solve a game equilibrium. The field's per-section pace ``v_field(section)`` is a REAL
observable from the data (the field-mean and the leader velocity of the runners in the same race
and section), so we CONDITION on it (it is contemporaneous public info — a legitimate boundary
condition) rather than solving for a Nash fixed point. That is what makes the coupling tractable
now; the per-horse PHYSIOLOGICAL constants (CP from the Hill anchor, W′_0, P_max) stay FIXED —
the pace game perturbs DEPLOYMENT, not the constants (0047/0048 established the constants are right).

================================================================================================
TWO PHYSICAL MECHANISMS (both first-principles, physically bounded — no fitted coefficients)
================================================================================================
**(1) Longitudinal slipstream / draft.** THEORY §III.2 writes the aero resistance as
``½ρ·C_dA·v²·(1 − draft_i)``. A horse running WITHIN the field (not on the lead, with bodies
ahead) sits in a partial slipstream → reduced effective drag area. The magnitude comes from
drafting aerodynamics, NOT a fit:

  * Close in-line drafting saves ~ 30–40 % of aero drag (cyclist/runner wind-tunnel; the wake of
    a bluff body roughly halves stagnation pressure for a follower one body-length back).
  * In a loose pack the saving is smaller, ~ 15–25 %.

We take a single physically-anchored slipstream fraction ``DRAFT_FRAC = 0.30`` (30 % aero-drag
reduction) applied only when the horse is BEHIND the lead, scaled by how covered it is (its field-
relative running position: a deep mid-pack horse is more covered than one racing 2nd). The LEADER
gets no draft (clear air). Aero is ~40–47 % of total resistance at racing speed (0047), so a
30 % aero cut is a ~12–14 % cut on the aero term ⇒ a few-tenths-m/s cruise lift — the right order
to explain part of the cruise-vs-Hill gap. The FULL positional draft (who is laterally behind
whom) needs CV and is DEFERRED (note the boundary in the journal); this is the coarse longitudinal
version (covered-fraction from the longitudinal running-position rank).

**(2) Competitive pace-matching.** In the mean-field game each horse tends to run NEAR the field
pace where physically feasible — a front-runner among closers steals an uncontested lead, but a
horse in a fast-pace field holds a velocity ABOVE its own v_c, spending reserve to keep contact.
We make the cruise TARGET velocity the field pace where the field is running hot, bounded by what
P_max can physically sustain and by W′ ≥ 0:

      v_target = clip( v_field ,  v_c ,  v_max(P_max) )         (only ABOVE the horse's own v_c)

The horse holds ``v_target`` on power ``P_match`` (the force-balance power at v_target, drag now
draft-reduced), which is ABOVE CP whenever ``v_target > v_c`` ⇒ it DRAWS DOWN W′ at rate
``(P_match − CP)⁺``. When the reserve is gone it falls back to the sustainable level (the
draft-reduced v_c). This is the mechanism that makes the settled cruise sit ABOVE the physiological
v_c: the horse is pace-following on reserve, exactly as 0048 inferred (cruise reads ~1.4 m/s too
hot vs the Hill CV). **It reduces to the isolated EOM when ``v_field ≤ v_c``** (no pull above the
sustainable level → v_target = v_c → P_match = CP → no reserve drawn → bang–CP–bang) — the required
limit.

Both mechanisms have referents in the EOM force balance / energy law and physically-bounded
magnitudes; neither is a free regression coefficient on the outcome. Field pace is OBSERVED data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.model.dynamics import (
    ETA, M_REF, PhysioConstants, Resistances, cruise_velocity, make_resistances,
)

# ---------------------------------------------------------------- physical coupling magnitudes
# Slipstream aero-drag reduction for a fully-covered follower (drafting aerodynamics, not fitted):
# a bluff-body wake ~halves the follower's stagnation drag one body-length back; in racing packs the
# realised saving is ~0.30. The leader (clear air) gets 0; a horse only partly covered gets less.
DRAFT_FRAC = 0.30          # fraction of AERO drag removed for a fully-covered follower
# Pace-following is bounded by the instantaneous sustainable ceiling at P_max (force-balance root at
# P_max) and by W′ ≥ 0 — no free knob. The horse follows the field only ABOVE its own v_c.


def v_max_at_pmax(phys: PhysioConstants, res: Resistances) -> float:
    """The sustainable velocity ceiling at P = P_max (force-balance root) — the pace-follow cap.

    The fastest the horse could hold if it poured P_max into cruise: the root of
    ``η·P_max = resist_power(v)``. Pace-matching cannot pull the horse past this physical ceiling.
    """
    return cruise_velocity(phys.p_max, res)


def draft_aero_factor(cover_frac: float) -> float:
    """The aero multiplier ``(1 − draft)`` for a horse covered to fraction ``cover_frac`` ∈ [0,1].

    ``cover_frac`` = how shielded the runner is (0 = leader / clear air, 1 = fully covered deep in
    the pack). The realised slipstream saving scales linearly with cover up to ``DRAFT_FRAC``:
        draft = DRAFT_FRAC · cover_frac ,   factor = 1 − draft .
    """
    cf = float(np.clip(cover_frac, 0.0, 1.0))
    return 1.0 - DRAFT_FRAC * cf


def cover_fraction(position: float, field_size: float) -> float:
    """Longitudinal cover fraction from the running-position rank (coarse, longitudinal only).

    ``position`` = the running-position rank (1 = leader). The leader has clear air (cover 0); the
    further back, the more bodies ahead shield it. We map rank linearly to cover ∈ [0,1]:
        cover = (position − 1) / max(field_size − 1, 1) .
    This is the COARSE longitudinal proxy — the full positional draft (who is laterally behind whom)
    needs CV and is deferred. Returns 0 when position/field is degenerate.
    """
    fs = max(float(field_size), 2.0)
    return float(np.clip((float(position) - 1.0) / (fs - 1.0), 0.0, 1.0))


# ====================================================================== the coupled integrator
@dataclass(frozen=True)
class FieldCoupledSolution:
    """The field-coupled velocity path + derived section quantities (mirror of EOMSolution)."""
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    w_prime: np.ndarray
    finish_time: float
    v_c: float
    w_spent_pace: float        # J drawn down specifically by pace-following above v_c


def _field_v_at_x(x: float, dist: float, bounds: np.ndarray, v_field: np.ndarray,
                  cover: np.ndarray) -> tuple[float, float]:
    """Look up the observed field pace + cover at cumulative distance x (piecewise-constant/section).

    ``bounds`` are the cumulative section boundaries (len K+1); ``v_field``/``cover`` are the K
    per-section field-pace and cover values. Returns ``(v_field_here, cover_here)``.
    """
    k = v_field.shape[0]
    # find the section index containing x
    idx = int(np.searchsorted(bounds, x, side="right") - 1)
    if idx < 0:
        idx = 0
    elif idx >= k:
        idx = k - 1
    return float(v_field[idx]), float(cover[idx])


def solve_field_coupled(phys: PhysioConstants, res: Resistances, dist: float,
                        section_dists: np.ndarray, v_field: np.ndarray, cover: np.ndarray, *,
                        dt: float = 0.05, v0: float = 4.0) -> FieldCoupledSolution:
    """Integrate the field-COUPLED EOM: draft-reduced drag + pace-following deployment of W′.

    Same plant (eq. 2) and energy law (eq. 4) as the isolated EOM — the ONLY changes are physical:
      * the aero drag is multiplied by ``(1 − draft)`` from the per-section cover (mechanism 1);
      * the cruise control TARGETS the observed field pace where it exceeds the horse's own v_c,
        holding it on the force-balance power (which is > CP ⇒ draws W′), capped at v_max(P_max)
        and by W′ ≥ 0 (mechanism 2). Below v_c the field exerts no pull ⇒ this reduces to the
        isolated bang–CP–bang singular arc.

    The opening RAMP (P = P_max until the draft-reduced cruise root is reached) and the terminal
    behaviour (fall back to the sustainable level once W′ = 0) are UNCHANGED in structure — we do
    not re-solve the kick bisection here; the kick emerges naturally because pace-following spends
    reserve through the body of the race and the horse fades when it empties, which is exactly the
    early-burn / late-fade HUMP the isolated policy could not make.
    """
    m = res.m
    half_rho_cda = 0.5 * res.rho_air * res.c_da
    crr_mg = res.c_rr * m * res.g
    cp, w0, p_max, tau = phys.cp, phys.w_prime0, phys.p_max, phys.tau
    v_cap = v_max_at_pmax(phys, res)
    bounds = np.concatenate([[0.0], np.cumsum(np.asarray(section_dists, float))])

    n_max = int(dist / (max(v0, 1.0) * dt) * 6.0) + 20000
    ts = np.empty(n_max)
    xs = np.empty(n_max)
    vs = np.empty(n_max)
    ws = np.empty(n_max)
    t, x, v, w = 0.0, 0.0, max(v0, 1e-3), w0
    ramp_done = False
    w_spent_pace = 0.0
    # the ISOLATED draft-free cruise root (the sustainable level when not pulled by the field)
    i = 0
    while x < dist and i < n_max:
        ts[i], xs[i], vs[i], ws[i] = t, x, v, w
        vf, cov = _field_v_at_x(x, dist, bounds, v_field, cover)
        draft = 1.0 - DRAFT_FRAC * cov                          # mechanism 1: aero multiplier
        # the draft-reduced sustainable cruise root (P = CP with reduced drag): root of
        #   η·CP = draft·half_rho_cda·v³ + crr_mg·v.  Reuse cruise_velocity via a scaled c_da.
        res_draft = Resistances(m=m, c_da=res.c_da * draft, c_rr=res.c_rr, f_geom=res.f_geom,
                                rho_air=res.rho_air, g=res.g)
        v_c_draft = cruise_velocity(cp, res_draft)
        if (not ramp_done) and v >= v_c_draft - 1e-3:
            ramp_done = True
        # ---- power policy (field-coupled deployment) --------------------------------------
        if w <= 0.0:
            p = cp                                              # tank empty: sustainable level only
        elif not ramp_done:
            p = p_max                                           # opening ramp transient
        else:
            # CRUISE: target the field pace where it is HOT (above the horse's own draft cruise),
            # bounded by the P_max ceiling. Below v_c → target = v_c → P = CP (isolated limit).
            v_target = min(max(vf, v_c_draft), v_cap)
            # force-balance power to hold v_target against the draft-reduced drag (mechanism 2)
            f_resist = draft * half_rho_cda * v_target * v_target + crr_mg + res.f_geom
            p = (f_resist * v_target) / ETA
            if p > p_max:
                p = p_max
        # ---- integrate v (RK4) with the draft-reduced drag --------------------------------
        def dvdt(vv: float, pp: float = p) -> float:
            return (ETA * pp / max(vv, 1e-3)
                    - (draft * half_rho_cda * vv * vv + crr_mg + res.f_geom)) / m
        k1 = dvdt(v)
        k2 = dvdt(v + 0.5 * dt * k1)
        k3 = dvdt(v + 0.5 * dt * k2)
        k4 = dvdt(v + dt * k3)
        v_new = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        over = max(p - cp, 0.0)
        under = max(cp - p, 0.0)
        w_new = w + dt * (-over + under / tau * (1.0 - w / w0))
        if over > 0.0 and ramp_done:
            w_spent_pace += over * dt                           # reserve drawn by pace-following
        w_new = min(max(w_new, 0.0), w0)
        x_new = x + 0.5 * (v + v_new) * dt
        t, x, v, w = t + dt, x_new, max(v_new, 1e-3), w_new
        i += 1
    if i < n_max:
        ts[i], xs[i], vs[i], ws[i] = t, x, v, w
        i += 1
    sl = slice(0, i)
    v_c_iso = cruise_velocity(cp, res)
    return FieldCoupledSolution(t=ts[sl], x=xs[sl], v=vs[sl], w_prime=ws[sl], finish_time=t,
                                v_c=v_c_iso, w_spent_pace=float(w_spent_pace))


def field_section_velocities(phys: PhysioConstants, res: Resistances, section_dists, v_field,
                             cover, *, dt: float = 0.05, v0: float = 4.0) -> np.ndarray:
    """K-section velocity profile of the field-coupled EOM (drop-in for ``expected_profile_eom``).

    Given the per-horse physiology, the resistances (KNOWN mass + going), and the OBSERVED per-
    section field pace + cover, integrate the coupled EOM and project onto the section boundaries.
    """
    d = np.asarray(section_dists, float)
    vfa = np.asarray(v_field, float)
    cva = np.asarray(cover, float)
    dist = float(d.sum())
    sol = solve_field_coupled(phys, res, dist, d, vfa, cva, dt=dt, v0=v0)
    bounds = np.concatenate([[0.0], np.cumsum(d)])
    cross = np.interp(bounds, sol.x, sol.t)
    seg_t = np.diff(cross)
    seg_t = np.where(seg_t <= 0, 1e-6, seg_t)
    return d / seg_t


# ============================================================ pace-de-confounded cruise CP read
def deconfounded_cruise_velocity(v_settled: float, cover: float, res: Resistances) -> float:
    """Remove the pace inflation from a settled cruise read to recover the AEROBIC critical velocity.

    The settled cruise ``v_settled`` is observed hot for TWO physical reasons (0048): the horse is
    (a) pace-following ABOVE its v_c on reserve, and (b) drafting (reduced drag lets the same CP buy
    a higher velocity). The DRAFT part is exactly invertible: a horse cruising at v_settled with
    draft-reduced drag (cover ``cover``) is dissipating ``draft·½ρC_dA·v³ + C_rr·m·g·v`` of power;
    the AEROBIC critical velocity is the speed that SAME sustainable power buys in CLEAN air —

        η·CP = draft·½ρC_dA·v_settled³ + C_rr·m·g·v_settled        (power at the drafted cruise)
        η·CP =        ½ρC_dA·v_c³      + C_rr·m·g·v_c               (same CP, clean-air root)

    so we evaluate the drafted resistive power at v_settled, treat it as η·CP, and invert the
    CLEAN-air cruise root. This collapses the draft-driven part of the cruise–Hill gap; the
    pace-following part is handled by the coupled integrator (it never reaches steady state above
    v_c without spending W′, so a horse's MINIMUM settled section already approximates the
    sustainable floor). Returns the de-confounded critical velocity [m/s].
    """
    cf = float(np.clip(cover, 0.0, 1.0))
    draft = 1.0 - DRAFT_FRAC * cf
    res_draft = Resistances(m=res.m, c_da=res.c_da * draft, c_rr=res.c_rr, f_geom=res.f_geom,
                            rho_air=res.rho_air, g=res.g)
    # the sustainable power the horse is actually producing at the drafted cruise:
    eta_cp = float(res_draft.resist_power(v_settled))
    # invert the CLEAN-air root: find v_c with res.resist_power(v_c) == eta_cp
    from scipy.optimize import brentq

    def excess(v: float) -> float:
        return float(res.resist_power(v)) - eta_cp

    if excess(40.0) < 0:
        return 40.0
    if excess(1e-6) > 0:
        return float(v_settled)
    return float(brentq(excess, 1e-6, 40.0, xtol=1e-6, rtol=1e-8))


# ============================================================ the MEAN-FIELD EQUILIBRIUM PACE
# 0049 conditioned the EOM on the OBSERVED field pace (a contemporaneous boundary condition, fine
# for path RECONSTRUCTION but NOT usable to PREDICT a held-out race pre-race). The next structural
# step (THEORY §III.4a; 0049 §5; 0060 the FIELD/DEPLOYMENT cause) is to SOLVE the pace as a
# self-consistent fixed point of the field's deployment interaction, so the equilibrium pace is
# available from PRIOR-race constants alone — a genuine mean-field game, not a regression.
#
# THE GAME (large-field limit, rivals enter only through the aggregate FRONT pace):
#   Each runner i has, strictly-prior, an own opening-pace tendency ``base_i`` (the velocity it runs
#   to its OWN preferred deployment, uncontested) and a forwardness ``style_i`` ∈ [0,1] (1 = a
#   front-runner that refuses to be buried, 0 = a closer content to settle off the pace). NOTE
#   (journal 0063/0065 orientation fix): callers derive forwardness as ``1 − trail_early_pos_norm``
#   (the trailing position RANK is high=closer); this module always takes the CORRECTED forwardness.
#   THE FRONT-GROUP PACE is the forwardness-weighted mean of the bases — the LEADERS set the pace:
#       L = Σ style_j·base_j / Σ style_j .
#   BEST RESPONSE: a horse holds its field-relative position per its style. Where the FRONT is faster
#   than its own base it gets pulled UP toward L, in proportion to its OWN forwardness (a forward
#   horse contests a hot front; a deep closer is barely dragged along); a horse already faster than
#   the front holds its base (it is not slowed by chasing):
#       v_i = base_i + style_i · max(L − base_i, 0).
#   PACE FORMATION — the realised early FIELD pace is the field-mean of those best responses:
#       V* = mean_i v_i .
#   This is the mean-field game in its tractable, self-consistent form (THEORY §III.4a; 0049 §5):
#   relational, not scalar — a field whose FORWARD horses are also its FAST horses runs ABOVE the
#   plain field-mean (the hot front pulls the closers up), a field of slow front-runners does not.
#   The same horse, a different equilibrium. No coefficient is fitted to any outcome; ``base`` and
#   ``style`` are strictly-prior per-horse observables, the front pace is the "leaders set the pace"
#   mechanism, and the chase is bounded (closers cannot exceed the front, leaders are not slowed).
#   It REDUCES to the plain field-mean of ``base`` when the field is style-uniform or no front horse
#   is faster than the rest — the required null limit.


def front_group_pace(base: np.ndarray, style: np.ndarray) -> float:
    """The pace the FRONT GROUP sets — the forwardness-weighted mean of the bases (THEORY §III.4a).

    The leaders set the pace, so the field's contested front velocity is ``L = Σ s_j b_j / Σ s_j``.
    Falls back to the plain mean of ``base`` for a degenerate (all-zero-style) field.
    """
    b = np.asarray(base, dtype=float)
    s = np.clip(np.asarray(style, dtype=float), 0.0, 1.0)
    ssum = float(s.sum())
    if ssum <= 0.0:
        return float(b.mean())
    return float((s * b).sum() / ssum)


def equilibrium_pace(base: np.ndarray, style: np.ndarray) -> float:
    """Self-consistent mean-field equilibrium early FIELD pace of a field (THEORY §III.4a).

    Parameters
    ----------
    base : per-horse strictly-prior own opening-pace tendency [m/s] (uncontested deployment pace).
    style : per-horse forwardness ∈ [0,1] (1 = front-runner that refuses to be buried, 0 = closer).

    Returns ``V* = mean_i [ base_i + style_i·max(L − base_i, 0) ]`` where ``L`` is the
    forwardness-weighted front-group pace. The closers are pulled UP toward a hot front in
    proportion to their own forwardness; horses already faster than the front hold their base. The
    equilibrium is bounded in ``[mean(base), max(base)]`` — it can only LIFT the field-mean toward a
    fast front, never beyond the fastest base (chasing does not create speed) — and reduces to the
    plain field-mean of ``base`` for a style-uniform field (the null limit). No outcome-fitted knob.
    """
    b = np.asarray(base, dtype=float)
    s = np.clip(np.asarray(style, dtype=float), 0.0, 1.0)
    if b.size == 0:
        return float("nan")
    lead = front_group_pace(b, s)                       # the leaders set the pace
    v_resp = b + s * np.maximum(lead - b, 0.0)          # closers pulled up toward a hot front
    return float(np.mean(v_resp))


# ====================================================== the FRONT-GROUP PACE-WAR W′-BURN GRADIENT
# 0061's mean-field equilibrium is a section-1 LEVEL (how fast the field opens). The S7b ceiling
# diagnostic (journal 0062) showed that the field/deployment COMPOSITION signal does NOT live in the
# opening level — on the big galloping ST track that is a TRUE NULL — but in the early pace GRADIENT
# (section-2 − section-1 field-mean velocity): a front-heavy field that contests the lead runs the
# front group ABOVE its members' own optima early, burning W′, which perturbs how the pace evolves
# into section 2 (de-confounded composition signal over a no-composition conditions+pace ceiling:
# +0.030 ST / +0.057 HV, both seed-stable). 0061's level-chase form is structurally blind to it. This
# mechanism re-specifies the observable (LEVEL → GRADIENT) and the coupling (smooth chase-the-front →
# a W′-burn GATED BY A FRONT-RUNNER-COUNT THRESHOLD), as PHYSICS.
#
# THE PACE-WAR MECHANISM (first-principles; no coefficient fitted to any outcome) ----------------
# THEORY §III.4a: among GENUINE FRONT-RUNNERS (forwardness ``style ≥ cutoff``) the lead is
# CONTESTED — the front group is dragged to the contested front pace ``L`` (the fastest front-runner
# sets it; the others must match to hold position) ABOVE each member's own sustainable opening cruise
# ``base_i`` (which we read as its CP-level opening pace). A front-runner forced to hold ``L`` above
# its own ``base_i`` runs ABOVE CP and SPENDS anaerobic reserve W′ early. The excess work-rate above
# its sustainable level is the EOM force-balance power difference (no fit — pure resistance physics,
# eq. 3):
#       ΔP_i = [ resist_power(L) − resist_power(base_i) ]⁺      (W, the over-CP work-rate at L)
# converted to the velocity perturbation it represents via the LOCAL pace sensitivity dP/dv at the
# contested speed ``L`` (the same force-balance: holding ΔP above CP corresponds to a speed
# increment ΔP/(dP/dv); the spent reserve is the magnitude of the early over-pace):
#       δv_i = ΔP_i / (dP/dv|_L)        [m/s].
#
# THE THRESHOLD (a pace WAR, not a lone leader). A single uncontested front-runner sets the pace it
# wants — no over-spend. The contested over-spend needs MULTIPLE genuine front-runners crowding the
# lead. We gate the burn on the COUNT of genuine front-runners ``n_front = #{i: style_i ≥ cutoff}``
# exceeding a threshold ``FRONT_WAR_THRESHOLD`` via a smooth Michaelis–Menten saturation
#       g(n) = (n − n*)⁺ / [ (n − n*)⁺ + 1 ]   ∈ [0,1)
# (a contested front turns the over-spend ON above the threshold and SATURATES — a duel escalates but
# cannot run away). At/below the threshold g = 0 ⇒ no pace war ⇒ no induced gradient term (the
# required null limit: a field with ≤ n* front-runners reproduces the isolated deployment).
#
# THE GRADIENT TERM. The pace-war over-spend is the field-mean of the contesting front group's early
# over-pace, scaled by the pace-war gate:
#       pace_war_burn = g(n_front) · mean_i[ δv_i ]   ( ≥ 0, the magnitude of the early over-spend ).
# It is ZERO when no front-runner is held above its base (L ≤ every front base) or below the
# front-count threshold. Returned as a SIGNED gradient perturbation ``−pace_war_burn`` (the early
# over-spend pushes the pace ABOVE the sustainable opener and the field must give it back relative to
# a steady field — the perturbation to the section-1→2 evolution). The VALIDATION (journal 0062 /
# studies/field_gradient_coupling.py) scores it by CORRELATION to the de-confounded realised gradient
# residual, so only its monotone direction (not its absolute sign convention) is load-bearing. No
# outcome is regressed; ``base``/``style`` are strictly-prior per-horse observables, the resistance
# physics is the fixed EOM (eq. 3), and the threshold + contested-pace are physical structure.

FRONT_WAR_THRESHOLD = 2.0   # # of genuine front-runners above which a pace WAR contests the lead (a
                            # lone/uncontested leader sets its own pace → no over-spend → no burn).
FRONT_STYLE_CUTOFF = 0.34   # forwardness ∈[0,1] above which a horse is a genuine front-runner that
                            # contests the lead. forwardness = front-third early position: racing
                            # ahead of the field median (a position-rank reading, not an outcome fit).

# The reference resistance model (a documented ~500 kg racehorse on good turf, dynamics.M_REF) sets
# the PHYSICAL pace→power scale of the burn. It is the EOM's own reference constant — NOT a fit; the
# burn is a ratio of resistance powers so it is robust to the exact mass (the dP/dv in the δv
# conversion divides out most of the absolute scale).
_REF_RES = make_resistances(M_REF, going=None)


def front_war_gate(n_front: float, threshold: float = FRONT_WAR_THRESHOLD) -> float:
    """Pace-war saturation gate ∈ [0,1) of the genuine-front-runner count ``n_front``.

    A lone or sub-threshold leader runs uncontested → 0 (no over-spend). Above ``threshold`` the
    contested over-spend turns on and SATURATES via Michaelis–Menten ``x/(x+1)`` with
    ``x = (n_front − threshold)⁺`` — a duel escalates but cannot run away. Returns 0 at/below the
    threshold (the required null limit), → 1 as the front crowd grows.
    """
    x = max(float(n_front) - float(threshold), 0.0)
    return x / (x + 1.0)


def _dp_dv(v: float, res: Resistances) -> float:
    """Local pace sensitivity dP/dv of the resistive power at speed v (eq. 3 differentiated).

    P(v) = ½ρ·C_dA·v³ + C_rr·m·g·v + F_geom·v ⇒ dP/dv = (3/2)ρ·C_dA·v² + C_rr·m·g + F_geom.
    """
    return float(1.5 * res.rho_air * res.c_da * v * v + res.c_rr * res.m * res.g + res.f_geom)


def front_count(style: np.ndarray, cutoff: float = FRONT_STYLE_CUTOFF) -> float:
    """Count of GENUINE front-runners ``#{i: style_i ≥ cutoff}`` (forward enough to contest the lead).

    A front-runner is a horse whose forwardness clears ``cutoff`` (races in the front third); the
    pace war is driven by HOW MANY such horses crowd the lead, not by a smooth style mean.
    """
    s = np.clip(np.asarray(style, dtype=float), 0.0, 1.0)
    return float((s >= cutoff).sum())


# The opening pace-war is contested over the early phase (the first ~two sections, journal 0062's
# section-1→2 gradient window). The over-pace δv_i [m/s] is held over that early duration; the
# anaerobic reserve it depletes is δv_i·t_early [m] — the EXTRA distance the front-runner "borrows"
# early by contesting the war (the same Σ(v−v_c)·t distance-reserve units the model's W′ carries,
# `energetics.recover_energetics`). The early-phase distance is a fixed physical opening window (not a
# fit): the model's W′ already integrates over-pace over the whole run, so the burn is the part of
# that integral the pace war forces in the OPENING — it is subtracted from the reserve carried to the
# finish. The window is a documented physical constant (the opening straight before the field settles).
PACE_WAR_EARLY_DIST = 400.0   # m, the opening phase over which the contested front pace is held (≈ the
                              # first two ~200 m sectionals — the section-1→2 gradient window of 0062).


def pace_war_burn_per_runner(base: np.ndarray, style: np.ndarray, *,
                             res: Resistances | None = None,
                             threshold: float = FRONT_WAR_THRESHOLD,
                             cutoff: float = FRONT_STYLE_CUTOFF,
                             early_dist: float = PACE_WAR_EARLY_DIST) -> np.ndarray:
    """PER-RUNNER anaerobic reserve [m] depleted EARLY by the front-group pace war (leak-free).

    The per-runner companion of :func:`pace_war_gradient` for the WIN-drift wiring (journal 0065). A
    genuine front-runner (``style ≥ cutoff``) forced to hold the contested front pace ``L`` (the
    fastest front-runner's own cruise) above its OWN sustainable opening cruise ``base_i`` runs ABOVE
    CP and spends reserve early. The over-pace it holds is ``δv_i = ΔP_i⁺/(dP/dv|_L)`` [m/s]
    (``ΔP_i⁺ = [P(L) − P(base_i)]⁺``, the EOM force-balance over-spend at L — pure resistance physics,
    eq. 3); the RESERVE it depletes over the opening phase is ``δv_i · t_early`` [m] where the early
    duration ``t_early = early_dist/L`` comes from holding ``L`` over the opening window. This is the
    same distance-reserve unit the model's W′ carries (``Σ(v−v_c)·t``, ``energetics``), so it
    subtracts directly off W′_eff. Gated on the front-runner-count pace-WAR threshold
    (:func:`front_war_gate`) — only a CONTESTED front burns reserve; a lone leader (or ≤ threshold
    front-runners) returns all-zeros.

    Returns a non-negative per-runner vector aligned with ``base``/``style`` (closers and uncontested
    leaders get 0). No coefficient fitted to any outcome: ``base``/``style`` are strictly-prior, the
    resistance physics is the fixed EOM, the threshold + contested-pace + early window are physical.
    """
    b = np.asarray(base, dtype=float)
    s = np.clip(np.asarray(style, dtype=float), 0.0, 1.0)
    burn = np.zeros_like(b)
    if b.size == 0:
        return burn
    rr = _REF_RES if res is None else res
    gate = front_war_gate(front_count(s, cutoff), threshold)
    if gate <= 0.0:
        return burn                                         # no pace war → no reserve burned
    front = s >= cutoff
    if not front.any():
        return burn
    lead = float(b[front].max())                            # contested front pace = fastest front-runner
    if lead <= 0.0:
        return burn
    p_lead = float(rr.resist_power(lead))
    dpdv = _dp_dv(lead, rr)
    if dpdv <= 0.0:
        return burn
    t_early = float(early_dist) / lead                      # opening-phase duration holding L [s]
    for i in range(b.size):
        if not front[i] or lead <= b[i]:
            continue                                        # closer / leader / not held above its cruise
        delta_p = p_lead - float(rr.resist_power(b[i]))     # ΔP_i⁺ at the contested pace [W]
        if delta_p > 0.0:
            dv = delta_p / dpdv                             # early over-pace [m/s]
            burn[i] = gate * dv * t_early                   # reserve depleted early [m] (≥0)
    return burn


def pace_war_gradient(base: np.ndarray, style: np.ndarray, *,
                      res: Resistances | None = None,
                      threshold: float = FRONT_WAR_THRESHOLD,
                      cutoff: float = FRONT_STYLE_CUTOFF) -> float:
    """Deployment-induced early pace GRADIENT perturbation from the front-group pace-war W′-burn.

    The first-principles re-specification of the field/deployment coupling for the early pace
    GRADIENT (journal 0062). Among genuine front-runners (``style ≥ cutoff``) the lead is contested
    at the front pace ``L`` (the fastest front-runner sets it); each front-runner held above its own
    sustainable cruise ``base_i`` spends excess over-CP reserve ``ΔP_i⁺ = [P(L) − P(base_i)]⁺``
    early, an early over-pace ``δv_i = ΔP_i/(dP/dv|_L)`` [m/s]. The term fires only above a
    front-runner-count pace-WAR threshold (``front_war_gate``). Returns ``−g(n_front)·mean_i δv_i``
    — the signed gradient perturbation (validated by correlation to the de-confounded realised
    gradient residual, so its monotone direction, not its sign convention, is load-bearing).

    Parameters
    ----------
    base : per-horse strictly-prior sustainable opening cruise [m/s] (CP-level pace, leak-free).
    style : per-horse forwardness ∈ [0,1] (1 = front-runner that contests the lead, 0 = closer).
    res : resistance model setting the pace→power scale; defaults to the EOM reference (~500 kg,
          good turf). Physical, fixed — not fitted; the burn is a power ratio so it is robust to the
          exact mass / going.

    Limits (unit-tested): returns 0.0 when no front-runner is held above its base (``L`` ≤ every
    front base) and when the front-runner count is at/below ``threshold`` (no pace war) — the
    isolated-deployment null.
    """
    b = np.asarray(base, dtype=float)
    s = np.clip(np.asarray(style, dtype=float), 0.0, 1.0)
    if b.size == 0:
        return float("nan")
    rr = _REF_RES if res is None else res
    gate = front_war_gate(front_count(s, cutoff), threshold)
    if gate <= 0.0:
        return 0.0                                          # no pace war → no induced gradient term
    front = s >= cutoff
    if not front.any():
        return 0.0
    lead = float(b[front].max())                            # contested front pace = fastest front-runner
    p_lead = float(rr.resist_power(lead))
    dpdv = _dp_dv(lead, rr)
    if dpdv <= 0.0:
        return 0.0
    dv = np.zeros_like(b)
    for i in range(b.size):
        if not front[i] or lead <= b[i]:
            continue                                        # not a contesting front-runner / not held up
        delta_p = p_lead - float(rr.resist_power(b[i]))     # ΔP_i⁺ at the contested pace
        if delta_p > 0.0:
            dv[i] = delta_p / dpdv                          # early over-pace this front-runner spends
    return float(-gate * np.mean(dv))

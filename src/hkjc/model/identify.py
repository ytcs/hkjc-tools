"""DEDICATED-SURFACE IDENTIFICATION of the EOM physical constants (journal DRAFT 0048).

The EOM core (``dynamics.py``, journal 0047) infers the four per-horse physiological constants
(CP, W′_0, P_max, τ) by a SINGLE bounded-NLS fit of the whole section-velocity path per run. That
joint fit UNDER-identifies coupled constants: many (CP, W′_0, P_max) combinations reproduce the
aggregate path about equally well (CP trades against W′_0 trades against P_max along the path).

This module identifies EACH constant on the DEDICATED DATA SURFACE where it DOMINATES the
dynamics, with a dedicated metric, and CROSS-CHECKS the same constant measured multiple
independent ways. Cross-surface CONSISTENCY is the deep validation that the physics is real
(G from a falling apple == G from the Moon's orbit). A DISAGREEMENT is a falsification that
points at missing physics.

There is NO regression of outcome on features here, NO market odds, NO win, NO pool book. Every
estimator inverts the SAME EOM physics (``dynamics.py``, imported untouched) on the surface where
one constant is least aliased. Mass + going are KNOWN inputs (never fitted).

================================================================================================
THE SURFACES (one dominant surface + a cross-check per constant)
================================================================================================
CP  — the critical (sustainable) power. TWO independent measurements that MUST AGREE:
  (a) CRUISE-IN-ROUTES (:func:`cp_cruise_run`): in long races (≥1800 m) the settled mid-race
      sections are quasi-steady-state at sustainable power. We read the settled cruise velocity
      v_settled and invert the force-balance root (eq. 3) → CP_cruise. CP dominates here because
      a settled cruise IS the v_c equilibrium; W′_0 / P_max barely touch a flat mid-race arc.
  (b) DISTANCE–TIME / HILL (:func:`hill_fit`): across a horse's races at DIFFERENT distances,
      each race is a near-maximal time-trial obeying the 2-parameter critical-velocity law
          d = CV · t + D'              (Hill 1925; the linear distance–time model)
      slope CV [m/s] = the critical velocity (the velocity-space CP), intercept D' [m] = the
      anaerobic distance capacity (the velocity-space W′). CV is identified by the SLOPE across
      distances — orthogonal to the per-race cruise level — so it is an independent CP estimate.
  The HEADLINE consistency test compares v_c(CP_cruise) against CV (both are the critical
  velocity, in m/s) per horse and in aggregate (correlation, mean abs diff, Bland–Altman).

P_max — peak power. (a) OPENING TRANSIENT (:func:`pmax_opening_run`): the gate acceleration
  dv/dt over the first section is set by P_max/m (drag is low at low v, W′ is untouched), so the
  opening accel inverts to P_max. (b) SPRINT CEILING: the maximum achievable sustained velocity
  in short races bounds P_max. Cross-check the two.

W′_0 — anaerobic reserve. (a) HILL intercept D' (velocity-space, → Joules via the force at CV).
  (b) RESERVE-SPENT-ABOVE-CP (:func:`wprime_spent_run`): the observed ∫(P−CP)^+ dt in a hard
  race, where P(t) is back-solved from the observed velocity path via the EOM. Two estimates.

τ — recovery time constant. SIT-AND-KICK closers (:func:`tau_sit_kick`): horses that settled
  BELOW v_c mid-race then kicked replenished W′ during the off-pace window; the late-kick
  magnitude per unit off-pace time reads τ. #52 found τ in-race-negligible — this surface either
  PINS τ or declares it UNIDENTIFIABLE in-race (a valid finding).

C_dA — drag area. The high-speed sprint ceiling constrains C_dA given P_max; the lead-vs-draft
  energy delta isolates slipstream (needs lateral position → deferred to CV). We report the
  sprint-ceiling sensitivity only.

All estimators are pure functions of one run (or one horse's runs) — leak-free by construction
when pooled strictly-prior. The cross-surface consistency checks are descriptive (cross-sectional).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.model.dynamics import (
    ETA, Resistances, cp_for_cruise, cruise_velocity, make_resistances,
)

# ------------------------------------------------------------------ shared run container
@dataclass(frozen=True)
class Run:
    """One run's observed kinematics + KNOWN conditions (the substrate for every estimator)."""
    horse_id: str
    distance: float                 # m, total trip
    section_dist: np.ndarray        # m, per-section distances (sum == distance)
    section_time: np.ndarray        # s, per-section elapsed times
    velocity: np.ndarray            # m/s, per-section mean velocity (= dist/time)
    mass: float                     # kg, KNOWN total moving mass
    going: str                      # surface state (→ C_rr)

    @property
    def total_time(self) -> float:
        return float(np.sum(self.section_time))

    @property
    def res(self) -> Resistances:
        return make_resistances(self.mass, self.going)


# ================================================================== CP — SURFACE (a): CRUISE
def settled_cruise_velocity(run: Run) -> float | None:
    """The settled mid-race cruise velocity of a route run (the CP-dominant surface).

    In a route the middle sections (excluding the opening ramp section 1 and the closing
    kick/fade section K) are the quasi-steady cruise arc. The CP-limited sustainable cruise is
    the SLOWEST settled level (a horse cruising faster than v_c is spending reserve), so we take
    the MINIMUM of the interior section velocities as the cleanest steady-state read — robust to a
    fast early-cruise burst (the §III.4 pace hump that 0047 named). Returns None if too short.
    """
    v = run.velocity
    k = len(v)
    if k < 4:                          # need ≥2 interior sections (drop opening + closing)
        return None
    interior = v[1:-1]
    return float(np.min(interior))


def cp_cruise_run(run: Run, *, level: str = "min") -> float | None:
    """CP_cruise [W] from one route run: invert the force-balance root at the settled cruise.

    ``level`` selects the interior-section summary used as the steady cruise velocity:
    ``"min"`` (the sustainable floor, primary), ``"median"`` (sensitivity), ``"mean"``.
    CP = resist_power(v_settled) / η  — eq. 3 solved for CP. The dedicated metric: CP dominates a
    flat settled arc (W′_0, P_max barely move a level mid-race section), so this read is CP-pinned.
    """
    v = run.velocity
    k = len(v)
    if k < 4:
        return None
    interior = v[1:-1]
    if level == "median":
        v_settled = float(np.median(interior))
    elif level == "mean":
        v_settled = float(np.mean(interior))
    else:
        v_settled = float(np.min(interior))
    return cp_for_cruise(v_settled, run.res)


# ================================================================== CP — SURFACE (b): HILL
@dataclass(frozen=True)
class HillFit:
    """Per-horse distance–time (Hill) critical-velocity fit  d = CV·t + D'."""
    cv: float                       # m/s, critical velocity (slope) = the velocity-space CP
    d_prime: float                  # m, anaerobic distance capacity (intercept) = velocity-space W′
    n_dist: int                     # number of DISTINCT race distances used
    n_run: int                      # number of runs used
    r2: float                       # fit R² of d on t
    cv_se: float                    # standard error of the slope (CV) — identifiability width
    span_m: float                   # distance span (max−min) — the lever arm of the slope


def hill_fit(dists: np.ndarray, times: np.ndarray, *, min_dist: int = 3) -> HillFit | None:
    """Fit the 2-parameter critical-velocity law  d = CV·t + D'  by OLS across a horse's runs.

    Each run is a near-maximal time-trial (the horse ran the full trip ~all-out), so the
    distance–time points lie on a line whose SLOPE is the critical velocity CV and whose INTERCEPT
    is the anaerobic distance capacity D'. CV is the velocity-space critical power — identified by
    the across-distance slope, ORTHOGONAL to any single race's cruise level. Needs ≥``min_dist``
    DISTINCT distances for a non-degenerate slope (else the line is unidentified).

    Returns None if too few distinct distances.
    """
    d = np.asarray(dists, float)
    t = np.asarray(times, float)
    n_dist = int(np.unique(d).size)
    if n_dist < min_dist or t.size < min_dist:
        return None
    # OLS d = a·t + b ; CV = a, D' = b
    design = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(design, d, rcond=None)
    cv, d_prime = float(coef[0]), float(coef[1])
    pred = design @ coef
    ss_res = float(np.sum((d - pred) ** 2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # slope SE = sqrt( σ² / Σ(t−t̄)² ),  σ² = ss_res/(n−2)
    n = t.size
    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    sxx = float(np.sum((t - t.mean()) ** 2))
    cv_se = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else float("inf")
    span = float(d.max() - d.min())
    return HillFit(cv=cv, d_prime=d_prime, n_dist=n_dist, n_run=n, r2=r2, cv_se=cv_se, span_m=span)


def hill_wprime_joules(fit: HillFit, res: Resistances) -> float:
    """Convert the Hill distance-capacity D' [m] to an energy reserve W′ [J].

    D' is the extra distance covered on anaerobic reserve at the critical velocity. The reserve
    energy is that distance times the force that must be overcome there, divided by efficiency
    (the metabolic cost of the extra distance):  W′_J = D' · F_resist(CV) / η. This puts the
    velocity-space Hill intercept on the same Joule axis as the EOM W′_0 / the spent-reserve.
    """
    f = float(res.resist_force(fit.cv))
    return fit.d_prime * f / ETA


# ================================================================== P_max — OPENING TRANSIENT
def pmax_opening_run(run: Run, *, v0: float = 4.0) -> float | None:
    """P_max [W] from the gate acceleration over the opening section (the P_max-dominant surface).

    Over the first section the horse accelerates from the gate lunge v0 to the opening-section
    mean velocity under P = P_max. Drag is low at low v and W′ is essentially untouched, so the
    opening acceleration is governed by P_max/m vs the (small) resistances — P_max-pinned.

    We invert the work–energy form over the opening section of length L1 covered in time t1.
    Mean propulsive power over the section solves the kinetic-energy + dissipation budget:
        η·P_max·t1 = ½·m·(v_end² − v0²) + ∫ F_resist·v dt
    approximating the velocity by the section mean v̄ for the (small) dissipation integral and
    taking v_end ≈ 2·v̄ − v0 (the section-mean of a near-linear ramp). Returns P_max [W].
    """
    if len(run.velocity) < 1 or run.section_time[0] <= 0:
        return None
    t1 = float(run.section_time[0])
    l1 = float(run.section_dist[0])
    vbar = l1 / t1
    v_end = 2.0 * vbar - v0            # section mean of a ~linear ramp ⇒ end ≈ 2·mean − start
    res = run.res
    dke = 0.5 * run.mass * (v_end ** 2 - v0 ** 2)
    diss = float(res.resist_force(vbar)) * vbar * t1     # ∫F·v dt ≈ F(v̄)·v̄·t1
    work = dke + diss
    p_max = work / (ETA * t1)
    return float(p_max) if p_max > 0 else None


def sprint_ceiling_velocity(run: Run) -> float:
    """The peak sustained velocity of a run (the sprint-ceiling cross-check surface for P_max).

    The maximum section velocity is the fastest the horse actually held over a 200–400 m window —
    a lower bound on the velocity P_max can sustain. In short races this is the sprint ceiling.
    """
    return float(np.max(run.velocity))


def pmax_for_ceiling(v_peak: float, res: Resistances, *, frac: float = 1.0) -> float:
    """The P_max [W] needed to hold the observed peak velocity ``v_peak`` (force balance at v_peak).

    If P=P_max sustains v_peak (dv/dt≈0 at the ceiling) then P_max = resist_power(v_peak)/η — the
    same root inversion as CP but at the SPRINT speed. ``frac`` (<1) accounts for the peak being a
    window-mean below the instantaneous ceiling. This is the sprint-ceiling P_max estimate.
    """
    return cp_for_cruise(v_peak * frac, res)


# ================================================================== W′_0 — RESERVE SPENT ABOVE CP
def wprime_spent_run(run: Run, cp: float) -> float | None:
    """W′_0 [J] = the reserve actually SPENT above CP in a hard race (back-solved from the path).

    Given a CP (e.g. the horse's CP_cruise), the metabolic power demanded to produce the observed
    velocity path is back-solved per section from the EOM (eq. 2) treating dv/dt from the observed
    section-to-section velocity change:
        P_demand(s) = [ m·v·dv/dt + F_resist(v)·v ] / η
    and the reserve spent is the time-integral of the excess over CP:
        W′_spent = Σ_s (P_demand(s) − CP)^+ · t_s .
    In a maximal race the reserve is fully spent ⇒ W′_spent ≈ W′_0. Independent of the Hill D'.
    Returns None if the run is too short.
    """
    v = run.velocity
    k = len(v)
    if k < 2:
        return None
    res = run.res
    t = run.section_time
    # section-boundary velocities for dv/dt: use the section-mean velocities, dv across adjacent
    spent = 0.0
    for s in range(k):
        vs = float(v[s])
        # acceleration over the section: forward difference of section-mean velocities / mid-time
        if s < k - 1:
            dv = float(v[s + 1]) - vs
            dt = 0.5 * (float(t[s]) + float(t[s + 1]))
        else:
            dv = vs - float(v[s - 1])
            dt = 0.5 * (float(t[s]) + float(t[s - 1]))
        dvdt = dv / dt if dt > 0 else 0.0
        p_demand = (run.mass * vs * dvdt + float(res.resist_force(vs)) * vs) / ETA
        over = max(p_demand - cp, 0.0)
        spent += over * float(t[s])
    return float(spent)


# ================================================================== τ — SIT-AND-KICK CLOSERS
@dataclass(frozen=True)
class SitKick:
    """A sit-and-kick diagnostic for one run: off-pace settle then a late kick."""
    settle_deficit: float           # m/s, how far the settle ran BELOW v_c (mean over off-pace)
    off_pace_time: float            # s, time spent below v_c (recovery window)
    kick_excess: float              # m/s, peak late velocity above the settle (the kick)
    v_c: float                      # m/s, the run's cruise root (from CP_cruise)


def sit_kick_run(run: Run, cp: float) -> SitKick | None:
    """Identify a sit-and-kick: sections below v_c (recovery window) then a closing kick above.

    A horse that settles below its cruise root v_c is recovering W′ (eq. 4, under-CP branch). The
    magnitude of the subsequent kick (closing velocity above the settle) per unit off-pace time is
    the empirical reserve-replenishment rate, from which τ is read: a SHORT τ ⇒ a big kick after a
    short settle; a LONG τ ⇒ the off-pace settle replenishes little, so the kick is small.
    Returns None if the run did not settle below v_c (not a sit-and-kick).
    """
    v = run.velocity
    k = len(v)
    if k < 4:
        return None
    v_c = cruise_velocity(cp, run.res)
    interior = v[1:-1]
    below = interior < v_c
    if not below.any():
        return None
    # off-pace settle window (interior sections below v_c)
    settle_v = interior[below]
    settle_deficit = float(np.mean(v_c - settle_v))
    # off-pace time = section times of the below-v_c interior sections
    t_interior = run.section_time[1:-1]
    off_pace_time = float(np.sum(t_interior[below]))
    # the kick = closing-section velocity above the settle floor
    kick_excess = float(v[-1] - np.min(settle_v))
    return SitKick(settle_deficit=settle_deficit, off_pace_time=off_pace_time,
                   kick_excess=kick_excess, v_c=v_c)


def tau_from_sit_kick(sk: SitKick, w_prime0: float, res: Resistances) -> float | None:
    """Estimate τ [s] from one sit-and-kick: the reserve replenished during the off-pace window.

    During the off-pace window the under-CP branch of eq. 4 replenishes W′ at rate
        dW′/dt = (CP−P)^+ / τ · (1 − W′/W′_0) ≈ ΔP_under / τ   (near-empty reserve, 1−W′/W′_0≈1)
    so over the window of duration Δt the reserve gained is ≈ ΔP_under·Δt/τ. The kick then SPENDS
    that gained reserve: the extra energy of the kick above the settle ≈ the replenished reserve.
    Equating and solving for τ:
        τ ≈ ΔP_under · Δt / W′_gained
    where ΔP_under = CP − resist_power(v_settle)/η ≈ (the power deficit at the settle) and
    W′_gained ≈ the kick's extra mechanical energy = ½·m·(v_kick²−v_settle²)+dissipation, /η.
    A coarse single-run read (noisy); the DISTRIBUTION across closers is the τ signal. Returns
    None when the algebra is degenerate (no deficit / no kick).
    """
    v_settle = sk.v_c - sk.settle_deficit
    if sk.settle_deficit <= 0 or sk.kick_excess <= 0 or sk.off_pace_time <= 0:
        return None
    cp = cp_for_cruise(sk.v_c, res)
    p_settle = float(res.resist_power(v_settle)) / ETA
    dp_under = cp - p_settle                                   # W, the under-CP power deficit
    if dp_under <= 0:
        return None
    v_kick = v_settle + sk.kick_excess
    w_gained = (0.5 * res.m * (v_kick ** 2 - v_settle ** 2)) / ETA   # J, reserve the kick spent
    if w_gained <= 0:
        return None
    tau = dp_under * sk.off_pace_time / w_gained
    return float(tau) if tau > 0 else None


# ================================================================== consistency helpers
def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Bland–Altman agreement of two independent measurements of the same quantity.

    Returns the mean difference (bias), the SD of the difference, the 95% limits of agreement,
    the mean absolute difference, and Pearson correlation — the standard consistency battery.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return {"n": int(a.size), "bias": float("nan"), "sd_diff": float("nan"),
                "loa_lo": float("nan"), "loa_hi": float("nan"), "mad": float("nan"),
                "corr": float("nan")}
    diff = a - b
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
    return {"n": int(a.size), "bias": bias, "sd_diff": sd,
            "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
            "mad": float(np.mean(np.abs(diff))), "corr": corr}

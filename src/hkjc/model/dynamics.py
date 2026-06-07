"""FIRST-PRINCIPLES generative core — the longitudinal equation of motion of a galloping
horse, solved under the bang–CP–bang optimal power-deployment policy (journal DRAFT 0047).

This module REPLACES the fitted / heuristic ``energetics_dynamic.expected_profile`` (a linear-
then-geometric W′-spend ansatz with a fitted opening ratio ``a``) with the SOLUTION of a real
equation of motion. There is NO regression here: every quantity is a physical constant with
units and bounds, the velocity path is the integral of Newton's law, and the phase structure
(ramp / cruise / kick) is the Pontryagin optimal control of the energy budget — not fitted
phases. The only per-horse unknowns are the four PHYSIOLOGICAL constants (CP, W′_0, P_max, τ),
inferred by physically-constrained inversion in ``studies/pathfit_eom.py``.

================================================================================================
THE EQUATION OF MOTION  (Newton, power-balance form)
================================================================================================
A horse of mass ``m`` [kg] driven by metabolic power ``P(t)`` [W], at speed ``v`` [m/s]:

    m · v · dv/dt  =  η·P(t)  −  ½·ρ_air·C_dA·v³  −  C_rr·m·g·v  −  F_geom·v          (1)

i.e. (propulsive mechanical power)  −  (aerodynamic-drag dissipation)
                                    −  (rolling / ground-contact resistance power)
                                    −  (bend / gradient resistance power).
Dividing (1) by v gives the force-per-unit-mass form actually integrated:

    m · dv/dt  =  η·P(t)/v  −  ½·ρ_air·C_dA·v²  −  C_rr·m·g  −  F_geom .              (2)

This is exactly the THEORY §III.2 / Appendix-B plant ``m·dv/dt = P/v − F_resist`` with
``F_resist = C_rr·m·g (locomotion) + ½ρ·C_dA·v² (aero) + F_geom (bend)`` — revived from the
deprecated power-balance EOM, the same mechanics, now integrated analytically on the path.

The STEADY-STATE cruise velocity ``v_c`` is the ROOT (dv/dt = 0 at P = CP) of

    η·CP  =  ½·ρ_air·C_dA·v_c³  +  C_rr·m·g·v_c  +  F_geom·v_c .                       (3)

``v_c`` is therefore a DERIVED quantity (a function of the physical CP and the resistances),
NOT a fitted number. We invert (3) once at construction to express CP↔v_c interchangeably.

================================================================================================
CRITICAL-POWER ENERGETICS  (the energy law)
================================================================================================
``P(t) ≤ P_max``.  The finite anaerobic reserve ``W′`` [J] obeys the bi-exponential CP balance

    dW′/dt  =  −(P − CP)^+  +  (CP − P)^+ / τ · (1 − W′/W′_0) ,      W′(t) ≥ 0  (hard).  (4)

Above CP the reserve is SPENT at rate (P−CP); below CP it RECOVERS toward W′_0 with time
constant τ. ``CP`` is the critical (sustainable) power — the largest P with dW′/dt ≤ 0 forever.

================================================================================================
OPTIMAL POWER-DEPLOYMENT POLICY  (generates the phases from first principles)
================================================================================================
For "cover the fixed race distance in minimum time subject to W′ ≥ 0", the Pontryagin /
bang–singular–bang optimal control of this problem class is bang–CP–bang:

  Phase 1 RAMP  : P = P_max from the gun until v reaches the cruise root v_c. This is an ODE
                  TRANSIENT governed by P_max/m vs the drag terms — the physically-correct
                  replacement for the fitted opening ratio ``a``. Its shape (a saturating
                  rise) and duration fall out of integrating (2) with P = P_max.
  Phase 2 CRUISE: P = CP, v ≈ v_c (the singular arc — sustainable, W′ held).
  Phase 3 KICK  : a terminal P = P_max burn, started at the unique time t_burn such that W′
                  hits exactly 0 at the finish line (full extension). t_burn is solved from
                  the W′-budget: the reserve spent in the ramp plus the reserve spent in the
                  kick must equal W′_0.

No phase is fitted; each is the optimal arc, and the boundaries (v reaches v_c; W′ reaches 0
at the line) are physical conditions, not free parameters.

================================================================================================
PHYSICAL CONSTANTS — values, units, bounds, sources
================================================================================================
Resistance / efficiency constants (NOT fitted — equine-locomotion literature priors):

  g        = 9.81   m/s²    standard gravity.
  ρ_air    = 1.20   kg/m³   air density at ~25 °C / sea level (Hong Kong).
  C_dA     = 1.10   m²      drag area = C_d·A. Frontal area A≈0.85 m² for a ~500 kg
                            thoroughbred (chest ~0.55 m wide × ~1.55 m tall projected);
                            C_d≈1.3 (a horse+jockey is a bluff body). Scaled allometrically
                            with mass as A ∝ m^(2/3) about the 500 kg reference. Sources:
                            equine aerodynamics estimates (Spence et al.; Pugh 1971 for the
                            human-runner ½ρC_dA·v² form transferred with equine A).
  C_rr     = 0.040         locomotion / ground-contact "rolling" coefficient = the MECHANICAL
                            cost-of-transport coefficient: per-metre mechanical gait work ≈
                            C_rr·m·g·dx, so mechanical COT = C_rr·g ≈ 0.39 J/(kg·m). This is the
                            galloping mechanical COT (Minetti; Taylor cost-of-locomotion ~0.4–0.6
                            J/(kg·m) external+internal mechanical work). Turf good = baseline;
                            softer going raises it (going→surface map below). With η below it
                            gives CP/m ≈ 40–50 W/kg gross at v_c — the measured racehorse gallop
                            metabolic power (VO₂ ~140 ml O₂/kg/min ≈ 47 W/kg). This + aero are
                            the racing-speed resistances; ground dominates (~60%), aero ~40%.
  η        = 0.25          gross metabolic→mechanical efficiency of equine muscle
                            (muscle efficiency ~0.20–0.30; Minetti, Taylor cost-of-locomotion).
  F_geom   = C_bend·m·v_avg²/R_bend ground-loss on the turn (set 0 on the straight; see
                            ``surface_resistance``). Small vs locomotion+aero — included for
                            completeness, off by default until per-venue geometry is wired.

Per-horse PHYSIOLOGICAL constants (the latents we INFER, with physical bounds):

  CP  [W]    critical power. Bounds CP>0. For a ~500 kg horse cruising ~16.5 m/s the implied
             CP is O(10–15 kW) — inferred per horse (or expressed via v_c≈16–17 m/s).
  W′_0 [J]   anaerobic reserve. Bounds W′_0>0. Equine analogue of the human ~20 kJ scaled by
             mass → O(50–200 kJ).
  P_max [W]  peak power, P_max>CP. Sprint power ~1.5–3× CP.
  τ  [s]     reserve-recovery time constant, τ>0. O(100 s) (slow vs a ~70 s race ⇒ little
             in-race recovery — recovery matters mainly for the sub-CP settle).

All bounds are enforced in :func:`PhysioConstants.__post_init__` and the inversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

# ------------------------------------------------------------------ physical constants
G = 9.81                 # m/s², standard gravity
RHO_AIR = 1.20           # kg/m³, air density (HK climate)
C_DA_REF = 1.10          # m², drag area C_d·A at the reference mass
M_REF = 500.0            # kg, reference body mass for allometric scaling of A
C_RR_TURF = 0.040        # mechanical cost-of-transport coeff (locomotion/ground) on good turf
ETA = 0.25               # gross metabolic→mechanical efficiency
LB_TO_KG = 0.45359237    # declared weights are in pounds

# going → multiplicative surcharge on C_rr (softer ground = more ground resistance).
# Physical: yielding/soft turf deforms and absorbs energy, raising the per-metre ground cost.
# These are SURFACE PHYSICS multipliers (documented constants), not fitted gains.
_GOING_CRR = {
    "GOOD-FIRM": 0.95, "GOOD": 1.00, "GOOD TO FIRM": 0.97, "GOOD TO YIELDING": 1.06,
    "YIELDING": 1.10, "SOFT": 1.16, "SOFT+": 1.14, "HEAVY": 1.22,
    "WET FAST": 1.02, "WET SLOW": 1.10, "WET-AWT": 1.08, "SEALED": 1.05,
    "UNKNOWN": 1.00, "OTHER": 1.00,
}


def c_da_for_mass(m: float) -> float:
    """Drag area C_d·A scaled allometrically from the reference: frontal area A ∝ m^(2/3)."""
    return C_DA_REF * (m / M_REF) ** (2.0 / 3.0)


def surface_resistance(going: str | None) -> float:
    """Effective ground-resistance coefficient C_rr for the going (turf good = baseline)."""
    if going is None:
        return C_RR_TURF
    key = going.strip().upper()
    return C_RR_TURF * _GOING_CRR.get(key, 1.0)


def body_mass_kg(declared_horse_wt_lb: float, carried_wt_lb: float = 0.0) -> float:
    """Total moving mass = (declared body weight + carried weight) in kg. KNOWN, never fitted."""
    return (float(declared_horse_wt_lb) + float(carried_wt_lb)) * LB_TO_KG


# ----------------------------------------------------------------- resistance / force model
@dataclass(frozen=True)
class Resistances:
    """The non-propulsive resistances of the EOM (all physical constants, given the race)."""
    m: float                       # kg, total moving mass (KNOWN)
    c_da: float                    # m², drag area C_d·A (allometric in m)
    c_rr: float                    # ground-resistance coeff (going-dependent)
    f_geom: float = 0.0            # N, bend/gradient resistance force (0 on straight)
    rho_air: float = RHO_AIR
    g: float = G

    def resist_force(self, v: np.ndarray | float) -> np.ndarray | float:
        """Total resistive FORCE [N] at speed v: aero + ground + geometry."""
        return (0.5 * self.rho_air * self.c_da * np.asarray(v) ** 2
                + self.c_rr * self.m * self.g + self.f_geom)

    def resist_power(self, v: np.ndarray | float) -> np.ndarray | float:
        """Total resistive POWER [W] dissipated at speed v = F_resist · v."""
        return self.resist_force(v) * np.asarray(v)


def make_resistances(m: float, going: str | None = None, f_geom: float = 0.0) -> Resistances:
    """Build the resistance model for a run from its KNOWN mass + going (no fitting)."""
    return Resistances(m=m, c_da=c_da_for_mass(m), c_rr=surface_resistance(going),
                       f_geom=f_geom)


def cruise_velocity(cp: float, res: Resistances) -> float:
    """The steady-state cruise root v_c of  η·CP = ½ρ·C_dA·v³ + C_rr·m·g·v + F_geom·v   (eq. 3).

    A monotone-increasing cubic in v ⇒ a unique positive root, found by bracketed brentq.
    This is the PHYSICS definition of v_c — derived from CP and the resistances, never fitted.
    """
    target = ETA * cp

    def excess(v: float) -> float:
        return float(res.resist_power(v)) - target

    # the resistive power is 0 at v=0 and ↑∞; bracket the root generously (0 .. 40 m/s)
    if excess(40.0) < 0:                      # CP so large no bracket in range → clamp
        return 40.0
    return float(brentq(excess, 1e-6, 40.0, xtol=1e-6, rtol=1e-8))


def cp_for_cruise(v_c: float, res: Resistances) -> float:
    """Inverse of :func:`cruise_velocity` — the CP that yields cruise root v_c (eq. 3 solved for CP)."""
    return float(res.resist_power(v_c) / ETA)


# ---------------------------------------------------------------- the per-horse physiology
@dataclass(frozen=True)
class PhysioConstants:
    """The four per-horse PHYSIOLOGICAL constants — the only latents (with physical bounds)."""
    cp: float                      # W, critical power            (CP > 0)
    w_prime0: float                # J, anaerobic reserve         (W′_0 > 0)
    p_max: float                   # W, peak power                (P_max > CP)
    tau: float = 120.0             # s, reserve-recovery constant (τ > 0)

    def __post_init__(self) -> None:
        if not (self.cp > 0 and self.w_prime0 > 0 and self.p_max > self.cp and self.tau > 0):
            raise ValueError(
                f"PhysioConstants out of physical bounds: cp={self.cp}, w'={self.w_prime0}, "
                f"p_max={self.p_max} (must exceed cp), tau={self.tau}")


# ============================================================================ THE INTEGRATOR
@dataclass(frozen=True)
class EOMSolution:
    """The integrated velocity path and its derived section quantities."""
    t: np.ndarray                  # time grid [s]
    x: np.ndarray                  # cumulative distance [m]
    v: np.ndarray                  # velocity [m/s]
    w_prime: np.ndarray            # reserve [J]
    finish_time: float             # s
    v_c: float                     # cruise root [m/s]
    t_burn: float                  # s, kick start time (NaN if no kick fired)
    phase: np.ndarray = field(default_factory=lambda: np.array([]))


def _power_policy(v: float, w: float, v_c: float, phys: PhysioConstants,
                  x: float, dist: float, x_burn: float, ramp_done: bool) -> float:
    """The bang–CP–bang power at the current state, with the cruise arc LATCHED.

    RAMP  : the opening transient, P = P_max, until v first reaches the cruise root v_c. Once
            reached the ramp is DONE (latched) — we never re-enter it for a numerical dip,
            because the cruise (P = CP, v = v_c) is a STABLE equilibrium of the ODE (below v_c
            under CP, dv/dt > 0 pulls back up; above, < 0). This is the singular arc.
    CRUISE: P = CP, v held at v_c (the singular control), until the terminal kick.
    KICK  : past the kick-start distance x_burn, with reserve left, P = P_max.
    Tank empty (W′ ≤ 0): only the sustainable level P = CP is possible → v fades toward v_c.
    """
    if w <= 0.0:
        return phys.cp                     # tank empty: can only hold the sustainable level
    if x >= x_burn:
        return phys.p_max                  # terminal kick
    if not ramp_done:
        return phys.p_max                  # opening ramp transient (until v_c first reached)
    return phys.cp                         # cruise (singular arc, latched)


def _integrate(phys: PhysioConstants, res: Resistances, dist: float, v_c: float,
               x_burn: float, *, dt: float = 0.02, v0: float = 0.5) -> EOMSolution:
    """RK4-integrate the EOM (eq. 2) + the W′ balance (eq. 4) under a fixed kick-start distance.

    ``x_burn`` is the cumulative distance at which the terminal kick begins (the policy switch).
    Integrate to the finish line ``dist``; reserve is hard-clamped ≥ 0.
    """
    m = res.m
    n_max = int(dist / (v0 * dt) * 4) + 2000      # generous step budget
    ts = np.empty(n_max)
    xs = np.empty(n_max)
    vs = np.empty(n_max)
    ws = np.empty(n_max)
    phs = np.empty(n_max, dtype=np.int8)
    t, x, v, w = 0.0, 0.0, max(v0, 1e-3), phys.w_prime0
    ramp_done = False
    i = 0
    while x < dist and i < n_max:
        ts[i], xs[i], vs[i], ws[i] = t, x, v, w
        if not ramp_done and v >= v_c - 1e-3:
            ramp_done = True               # latch: the opening ramp has reached the cruise root
        p = _power_policy(v, w, v_c, phys, x, dist, x_burn, ramp_done)
        phs[i] = 2 if x >= x_burn else (0 if ramp_done else 1)

        def dvdt(vv: float, pp: float = p) -> float:
            return (ETA * pp / max(vv, 1e-3) - res.resist_force(vv)) / m

        def dwdt(vv: float, pp: float = p, ww: float = w) -> float:
            over = max(pp - phys.cp, 0.0)
            under = max(phys.cp - pp, 0.0)
            return -over + under / phys.tau * (1.0 - ww / phys.w_prime0)

        # RK4 on v (w uses its rate at the current power; power is piecewise-constant in the step)
        k1 = dvdt(v)
        k2 = dvdt(v + 0.5 * dt * k1)
        k3 = dvdt(v + 0.5 * dt * k2)
        k4 = dvdt(v + dt * k3)
        v_new = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        w_new = w + dt * dwdt(v)
        w_new = min(max(w_new, 0.0), phys.w_prime0)
        x_new = x + 0.5 * (v + v_new) * dt
        t, x, v, w = t + dt, x_new, max(v_new, 1e-3), w_new
        i += 1
    # record the terminal state
    if i < n_max:
        ts[i], xs[i], vs[i], ws[i] = t, x, v, w
        phs[i] = phs[i - 1] if i > 0 else 0
        i += 1
    sl = slice(0, i)
    return EOMSolution(t=ts[sl], x=xs[sl], v=vs[sl], w_prime=ws[sl], finish_time=t,
                       v_c=v_c, t_burn=float("nan"), phase=phs[sl])


def solve_eom(phys: PhysioConstants, res: Resistances, dist: float, *,
              dt: float = 0.02, v0: float = 0.5) -> EOMSolution:
    """Integrate the EOM under the bang–CP–bang policy with the kick TIMED so W′ → 0 at the line.

    Solves the kick-start distance ``x_burn`` by bisection on the physical condition "reserve is
    exactly exhausted at the finish": run with a candidate x_burn, read the terminal reserve;
    earlier burn (smaller x_burn) spends more reserve. The unique x_burn making W′(dist)=0 is the
    optimal terminal-kick start (the Pontryagin switch). If even a kick over the whole non-opening
    trip cannot empty the tank, the horse never fully extends → x_burn = dist (no kick, pure CP).
    """
    v_c = cruise_velocity(phys.cp, res)

    def terminal_w(x_burn: float) -> float:
        return float(_integrate(phys, res, dist, v_c, x_burn, dt=dt, v0=v0).w_prime[-1])

    # bracket: x_burn = dist → no kick (max terminal reserve); x_burn small → earliest kick.
    # earliest physical burn starts only after the ramp settles; use a small floor.
    x_lo = min(0.05 * dist, 50.0)         # earliest credible kick start
    x_hi = dist                           # no kick
    w_no_kick = terminal_w(x_hi)
    if w_no_kick <= 1e-6:
        # the cruise+ramp alone already empties the tank (rare; under-powered horse) → no kick
        sol = _integrate(phys, res, dist, v_c, x_hi, dt=dt, v0=v0)
        return EOMSolution(sol.t, sol.x, sol.v, sol.w_prime, sol.finish_time, v_c, float(dist),
                           sol.phase)
    w_full_kick = terminal_w(x_lo)
    if w_full_kick > 1e-6:
        # even the earliest kick cannot empty the tank → horse cannot fully extend; kick from x_lo
        sol = _integrate(phys, res, dist, v_c, x_lo, dt=dt, v0=v0)
        return EOMSolution(sol.t, sol.x, sol.v, sol.w_prime, sol.finish_time, v_c, float(x_lo),
                           sol.phase)
    # bisection: terminal_w is increasing in x_burn (later kick → more reserve left)
    a, b = x_lo, x_hi
    for _ in range(40):
        mid = 0.5 * (a + b)
        if terminal_w(mid) > 0.0:
            b = mid
        else:
            a = mid
        if b - a < 0.5:                   # 0.5 m resolution on the switch
            break
    x_burn = 0.5 * (a + b)
    sol = _integrate(phys, res, dist, v_c, x_burn, dt=dt, v0=v0)
    return EOMSolution(sol.t, sol.x, sol.v, sol.w_prime, sol.finish_time, v_c, float(x_burn),
                       sol.phase)


# --------------------------------------------------------- section-velocity projection
def section_velocities(sol: EOMSolution, section_dists: np.ndarray | list) -> np.ndarray:
    """Project the integrated path onto the K observed section boundaries.

    Each section's reported velocity is its distance / its elapsed time = the harmonic-mean
    speed over that distance window — exactly how the sectionals are measured. We read the
    crossing times of the cumulative section boundaries off the integrated x(t).
    """
    d = np.asarray(section_dists, dtype=float)
    bounds = np.concatenate([[0.0], np.cumsum(d)])
    # crossing time of each cumulative boundary via interpolation on x(t)
    cross = np.interp(bounds, sol.x, sol.t)
    seg_t = np.diff(cross)
    seg_t = np.where(seg_t <= 0, 1e-6, seg_t)
    return d / seg_t


def expected_profile_eom(section_dists, phys: PhysioConstants, res: Resistances, *,
                         dt: float = 0.02) -> np.ndarray:
    """GENERATIVE K-section velocity profile from the EOM solution (the drop-in for the heuristic).

    Integrates the bang–CP–bang EOM over the total section distance and projects onto the
    observed section boundaries. Pure physics: given (CP, W′_0, P_max, τ) and the resistances
    (from KNOWN mass + going), the profile is the ODE solution, not an ansatz.
    """
    d = np.asarray(section_dists, dtype=float)
    dist = float(d.sum())
    sol = solve_eom(phys, res, dist, dt=dt)
    return section_velocities(sol, d)


# ============================================================================ FAST CORE
# The reference integrator above is exact but pure-Python (~0.4 s/solve) — too slow to invert
# ~40 k runs. This numba-jitted core integrates the SAME EOM + W′ balance under the SAME
# bang–CP–bang policy, with the kick-timing bisection INSIDE the jit, and returns just what the
# inversion needs: the K section velocities + terminal reserve. Byte-for-byte the same physics,
# ~1000× faster. ``_eom_sections_fast`` is validated against the reference in the unit tests.
try:                                            # numba is optional (the reference path works w/o)
    from numba import njit                      # type: ignore

    @njit(cache=True, fastmath=True)
    def _integrate_fast(cp, w0, p_max, tau, v_c, eta, half_rho_cda, crr_mg, f_geom, m,
                        dist, x_burn, dt, v0):   # noqa: PLR0913
        """Return (terminal_W′, K-boundary crossing times) for a given kick-start distance."""
        x = 0.0
        v = v0 if v0 > 1e-3 else 1e-3
        w = w0
        t = 0.0
        ramp_done = False
        # we don't store the path; we only need boundary crossing times (filled by the caller via
        # a second pass). Here just return terminal w and finish time for the bisection.
        nmax = int(dist / (max(v0, 1.0) * dt) * 6.0) + 200000
        for _ in range(nmax):
            if x >= dist:
                break
            if (not ramp_done) and v >= v_c - 1e-3:
                ramp_done = True
            if w <= 0.0:
                p = cp
            elif x >= x_burn:
                p = p_max
            elif not ramp_done:
                p = p_max
            else:
                p = cp
            # RK4 on v
            vv = v
            k1 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + 0.5 * dt * k1
            k2 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + 0.5 * dt * k2
            k3 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + dt * k3
            k4 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            v_new = v + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            over = p - cp if p > cp else 0.0
            under = cp - p if cp > p else 0.0
            w_new = w + dt * (-over + under / tau * (1.0 - w / w0))
            if w_new < 0.0:
                w_new = 0.0
            elif w_new > w0:
                w_new = w0
            x = x + 0.5 * (v + v_new) * dt
            v = v_new if v_new > 1e-3 else 1e-3
            w = w_new
            t = t + dt
        return w, t

    @njit(cache=True, fastmath=True)
    def _path_fast(cp, w0, p_max, tau, v_c, eta, half_rho_cda, crr_mg, f_geom, m,
                   dist, x_burn, dt, v0, bounds, out_cross):   # noqa: PLR0913
        """Integrate and record the crossing time of each cumulative boundary in ``bounds``."""
        x = 0.0
        v = v0 if v0 > 1e-3 else 1e-3
        w = w0
        t = 0.0
        ramp_done = False
        nb = bounds.shape[0]
        bi = 0
        x_prev = 0.0
        t_prev = 0.0
        nmax = int(dist / (max(v0, 1.0) * dt) * 6.0) + 200000
        for _ in range(nmax):
            while bi < nb and x >= bounds[bi]:
                # linear-interp the crossing time between the last two samples
                if x > x_prev:
                    out_cross[bi] = t_prev + (t - t_prev) * (bounds[bi] - x_prev) / (x - x_prev)
                else:
                    out_cross[bi] = t
                bi += 1
            if x >= dist or bi >= nb:
                break
            if (not ramp_done) and v >= v_c - 1e-3:
                ramp_done = True
            if w <= 0.0:
                p = cp
            elif x >= x_burn:
                p = p_max
            elif not ramp_done:
                p = p_max
            else:
                p = cp
            vv = v
            k1 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + 0.5 * dt * k1
            k2 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + 0.5 * dt * k2
            k3 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            vv = v + dt * k3
            k4 = (eta * p / (vv if vv > 1e-3 else 1e-3)
                  - (half_rho_cda * vv * vv + crr_mg + f_geom)) / m
            v_new = v + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            over = p - cp if p > cp else 0.0
            under = cp - p if cp > p else 0.0
            w_new = w + dt * (-over + under / tau * (1.0 - w / w0))
            if w_new < 0.0:
                w_new = 0.0
            elif w_new > w0:
                w_new = w0
            x_prev, t_prev = x, t
            x = x + 0.5 * (v + v_new) * dt
            v = v_new if v_new > 1e-3 else 1e-3
            w = w_new
            t = t + dt
        # fill any unreached boundaries with the final time (degenerate)
        while bi < nb:
            out_cross[bi] = t
            bi += 1

    _HAS_NUMBA = True
except Exception:                               # pragma: no cover - numba always present here
    _HAS_NUMBA = False


def eom_section_velocities_fast(section_dists, cp: float, w_prime0: float, p_max: float,
                                tau: float, res: Resistances, *, dt: float = 0.05,
                                v0: float = 4.0) -> tuple[np.ndarray, float, float]:
    """FAST bang–CP–bang section velocities + (v_c, x_burn) — the inversion's forward model.

    Solves the cruise root, finds the kick-start distance making W′→0 at the line (bisection in
    the jitted core), and returns the K section velocities. Returns ``(v_sections, v_c, x_burn)``.
    Identical physics to :func:`expected_profile_eom`, ~1000× faster (numba).
    """
    d = np.asarray(section_dists, dtype=float)
    dist = float(d.sum())
    v_c = cruise_velocity(cp, res)
    half_rho_cda = 0.5 * res.rho_air * res.c_da
    crr_mg = res.c_rr * res.m * res.g
    args = (cp, w_prime0, p_max, tau, v_c, ETA, half_rho_cda, crr_mg, res.f_geom, res.m, dist)

    def term_w(xb):
        return _integrate_fast(*args, xb, dt, v0)[0]

    x_lo = min(0.05 * dist, 50.0)
    if term_w(dist) <= 1e-6:                    # ramp+cruise empties the tank → no kick
        x_burn = dist
    elif term_w(x_lo) > 1e-6:                   # earliest kick still cannot empty → kick from x_lo
        x_burn = x_lo
    else:
        a, b = x_lo, dist
        for _ in range(30):
            mid = 0.5 * (a + b)
            if term_w(mid) > 0.0:
                b = mid
            else:
                a = mid
            if b - a < 1.0:
                break
        x_burn = 0.5 * (a + b)
    bounds = np.concatenate([[0.0], np.cumsum(d)])
    cross = np.empty(bounds.shape[0])
    half_rho_cda = 0.5 * res.rho_air * res.c_da
    _path_fast(cp, w_prime0, p_max, tau, v_c, ETA, half_rho_cda, crr_mg, res.f_geom, res.m,
               dist, x_burn, dt, v0, bounds, cross)
    seg_t = np.diff(cross)
    seg_t = np.where(seg_t <= 1e-9, 1e-6, seg_t)
    return d / seg_t, v_c, x_burn

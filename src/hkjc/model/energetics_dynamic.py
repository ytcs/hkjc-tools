"""Hierarchical W′-balance dynamic-state recovery of (v_c, W′_0) from a velocity profile.

THE UPGRADE over :func:`hkjc.model.energetics.estimate_run_energetics` (the *scalar*
recovery: ``v_c`` = slower of the final two section velocities, ``W′ = Σ_k (v_k−v_c)^+·t_k``
— a depletion-only, distance-units anaerobic proxy with NO recovery term, NO joint fit of
the level). Here ``(v_c, W′_0)`` are fit JOINTLY to the whole K-section velocity profile by a
discrete **W′-balance** dynamic-state model (critical-power / Hill, THEORY §2.1/§2.2):

    state W′_k = reserve remaining at the START of section k
    W′_1     = W′_0
    DEPLETE_k = (v_k − v_c)^+ · weight(v_k) · t_k        # spent above the aerobic ceiling
    RECOVER_k = ρ · (W′_0 − W′_k) · t_k · 1[v_k < v_c]   # refilled below it (ρ FIXED at pop level)
    W′_{k+1} = clip( W′_k − DEPLETE_k + RECOVER_k ,  0 , W′_0 )

with weight(v) selecting the **reserve UNIT** (the user's energy question):
    dist   : weight = 1        — distance above CV (Σ(v−v_c)^+·t, the baseline-equivalent)
    energy : weight = v        — velocity-weighted work above CV (drag/efficiency aware)
    drag   : weight = v²       — drag-weighted (optional variant)

Identification (K is tiny, 3–6): a maximally-extended race empties the tank, so the
physically-correct (v_c, W′_0) is the pair for which the reserve path stays ≥0 throughout
AND is near-exhausted at the line. We fix W′_0 = the running PEAK net cumulative depletion
above v_c (so the reserve never goes negative — the binding ≥0 constraint), and choose v_c
by a 1-D search that makes the TERMINAL reserve ≈0 (full extension) while v_c tracks the
settled late-race level. At ρ=0, weight=1, this reduces to the scalar baseline
(W′_0 = Σ(v−v_c)^+·t) with v_c re-identified by the exhaustion criterion rather than fixed
to min-late — and the limit ``v_c = min(v[-2:])`` is reproducible for the sanity check.

Units: distances m, times s, velocities m/s. ``W′_0`` is rescaled to the dist-unit
population mean by the caller so the (1+W′/L) drift boost magnitude is matched across units
(we test the SHAPE/weighting, not a gain change) — exactly as the reserve-formalization probe.

Leak-free: this is a DETERMINISTIC per-run transform of that run's OWN public sectionals
(no other horse, no outcome). The leakage discipline is unchanged from the baseline: the
recovered per-run (v_c, W′_0) flow through the EXISTING train-only EB population pooling +
strictly-prior per-horse shrinkage (``calibration.fit_energetics_population`` /
``horse_residuals``), so a race's own (v_c, W′_0) never enters the fit that predicts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _weight(v: np.ndarray, unit: str) -> np.ndarray:
    if unit == "dist":
        return np.ones_like(v)
    if unit == "energy":
        return v
    if unit == "drag":
        return v * v
    raise ValueError(f"unknown reserve unit {unit!r}")


def _balance_path(v: np.ndarray, t: np.ndarray, v_c: float, w0: float,
                  unit: str, rho: float) -> np.ndarray:
    """Forward-simulate the reserve path W′_1..W′_{K+1} for given (v_c, W′_0, ρ).

    Returns the per-section START reserve plus the terminal reserve (length K+1).
    """
    exc = np.clip(v - v_c, 0.0, None)
    deplete = exc * _weight(v, unit) * t
    below = v < v_c
    w = np.empty(len(v) + 1)
    w[0] = w0
    for k in range(len(v)):
        recover = rho * (w0 - w[k]) * t[k] if (rho > 0.0 and below[k]) else 0.0
        w[k + 1] = min(max(w[k] - deplete[k] + recover, 0.0), w0)
    return w


def _w0_for_vc(v: np.ndarray, t: np.ndarray, v_c: float, unit: str, rho: float) -> float:
    """Minimal W′_0 that keeps the reserve ≥0 throughout = running peak net deficit above v_c.

    With ρ=0 this is exactly Σ_k (v_k−v_c)^+·weight·t_k (cumulative depletion is monotone, so
    the peak is the total). With ρ>0 the recovery credits back sub-v_c sections, so the peak
    cumulative *net* deficit (which W′_0 must cover) is ≤ the gross depletion.
    """
    exc = np.clip(v - v_c, 0.0, None)
    deplete = exc * _weight(v, unit) * t
    below = v < v_c
    # iterate to a fixed point because RECOVER scales with W′_0 (a few passes converge; ρ small)
    w0 = float(deplete.sum())
    if rho <= 0.0:
        return w0
    for _ in range(8):
        w = 0.0
        peak = 0.0
        for k in range(len(v)):
            recover = rho * (w0 - max(w, 0.0)) * t[k] if below[k] else 0.0
            w = w - deplete[k] + recover           # net cumulative reserve change (≤0 region)
            peak = min(peak, w)
        new = -peak
        if abs(new - w0) < 1e-9:
            break
        w0 = new
    return max(w0, 0.0)


@dataclass(frozen=True)
class DynamicRunEnergetics:
    v_c: float
    w_prime: float          # W′_0 in the chosen reserve unit (rescale to dist mean downstream)
    v_peak: float
    v_mean: float
    fade: float
    n_sections: int


def estimate_run_energetics_dynamic(section_times, section_dists, *, unit: str = "dist",
                                    rho: float = 0.0,
                                    vc_grid: int = 41) -> DynamicRunEnergetics:
    """Recover (v_c, W′_0) from one run's profile by the W′-balance dynamic-state fit.

    ``unit`` ∈ {dist, energy, drag} selects the reserve unit; ``rho`` ≥ 0 is the FIXED
    population recovery rate (0 = pure depletion). ``v_c`` is searched on a grid spanning the
    observed velocities; for each candidate W′_0 is the minimal reserve keeping the path ≥0,
    and v_c is chosen to make the terminal reserve ≈0 (full extension) — the closest fade-to
    level. Reduces to the scalar baseline at unit=dist, rho=0 (and v_c≈min-late).
    """
    t = np.asarray(section_times, dtype=float)
    d = np.asarray(section_dists, dtype=float)
    v = d / t
    k_sec = len(v)
    if k_sec < 2:
        vc = float(v[-1])
        return DynamicRunEnergetics(vc, 0.0, float(v.max()), float(d.sum() / t.sum()),
                                    float("nan"), k_sec)
    # candidate v_c grid: from a floor below the slowest to the run mean (v_c is a SUSTAINABLE
    # level — never above the average pace). Bracketed by observed velocities.
    lo = float(v.min()) * 0.85
    hi = float(np.average(v, weights=t))
    grid = np.linspace(lo, hi, vc_grid)
    best_vc, best_obj = float(v[-2:].min()), np.inf
    for vc_c in grid:
        w0 = _w0_for_vc(v, t, vc_c, unit, rho)
        if w0 <= 0.0:
            continue
        w = _balance_path(v, t, vc_c, w0, unit, rho)
        term = w[-1]               # terminal reserve; want ≈0 (maximally extended at the line)
        # objective: terminal reserve as a FRACTION of W′_0 (full extension) + a mild penalty
        # keeping v_c near the settled late level (the slower of the final two sections), which
        # anchors identification when the profile under-determines it (e.g. an easy winner).
        late = float(v[-2:].min())
        obj = (term / w0) ** 2 + 0.05 * ((vc_c - late) / max(late, 1.0)) ** 2
        if obj < best_obj:
            best_obj, best_vc = obj, vc_c
    vc = best_vc
    w0 = _w0_for_vc(v, t, vc, unit, rho)
    v_end = float(v[-1])
    return DynamicRunEnergetics(
        v_c=float(vc), w_prime=float(w0), v_peak=float(v.max()),
        v_mean=float(d.sum() / t.sum()),
        fade=float(v.max() / v_end - 1.0) if v_end > 0 else float("nan"),
        n_sections=k_sec,
    )


# ====================================================================================
# PROFILE-LIKELIHOOD EXTENSION (pilot 0040) — the 4th latent `a` (acceleration) and a
# GENERATIVE expected K-section velocity profile + log-velocity Gaussian likelihood.
#
# THE NEW LATENT. The W′-balance dynamics above generate the CRUISE+CLOSING velocity
# (steady-state aerobic ceiling `v_c` boosted by the spending of the reserve `W′_0`), but
# they have NO representation of the START TRANSIENT — the gate-to-settled ramp-up
# (THEORY §3e Phase 1: "from rest, acceleration-limited, off the gate; this phase is NOT
# the steady-state first-passage and the homogeneous law has no representation of it").
# `a` is the per-run **opening-split ramp factor**: the first section velocity relative to
# the horse's own cruise velocity,
#
#     a = v_1 / v_cruise          (v_cruise = mean of the settled non-opening sections)
#
# a<1 ⇒ a horse still accelerating through the first split (the typical, gate-limited
# case, since the opening section also carries the standing start); a≈1 ⇒ already at cruise.
# This is INVISIBLE to the win outcome at the population level (it does not move v_c, the
# settled level that sets the order statistic) but leaves a CLEAR fingerprint in the opening
# split of the profile — the pilot's test case for "the rich likelihood identifies what the
# win outcome cannot". `a` enters the generative profile ONLY in the opening section:
# expected v_1 = a · v_cruise, the rest of the expected profile is the W′-balance velocity.
# ====================================================================================


# Geometric W′-spend ratio: each non-opening section carries ~this fraction of the previous
# section's boost over v_c. q≈0.5 matches the observed multiplicative decay of the per-section
# excess (0.063→0.031→0.015 of v_c, ST 4-section; #49 path-fit foundation, journal 0044/0045).
_SPEND_RATIO = 0.5


@dataclass(frozen=True)
class ProfileRunEnergetics:
    """A run's recovered (v_c, W′_0, a) plus the cruise level used to define `a`."""
    v_c: float
    w_prime: float
    accel: float            # a = v_1 / v_cruise  (opening-split ramp; <1 = still accelerating)
    v_cruise: float         # settled non-opening mean velocity (the `a` denominator)
    v_peak: float
    v_mean: float
    fade: float
    n_sections: int


def _cruise_velocity(v: np.ndarray, t: np.ndarray) -> float:
    """Settled cruise velocity = time-weighted mean of the NON-opening sections.

    The opening section is the start transient (standing start + ramp); the cruise level is
    the steady-state pace the horse settles into, which `a` measures the opening against. For
    K≥3 use sections 2..K-1 (drop opening AND closing kick/fade so cruise is the mid level);
    for K=3 use the single middle section; for K=2 fall back to the final section.
    """
    if len(v) >= 4:
        sl = slice(1, len(v) - 1)
    elif len(v) == 3:
        sl = slice(1, 2)
    else:
        sl = slice(len(v) - 1, len(v))
    vv, tt = v[sl], t[sl]
    return float(np.average(vv, weights=tt)) if tt.sum() > 0 else float(v[-1])


def estimate_run_profile(section_times, section_dists, *, unit: str = "dist",
                         rho: float = 0.0, vc_grid: int = 41) -> ProfileRunEnergetics:
    """Recover (v_c, W′_0, a) from one run's profile — the W′-balance fit PLUS the opening ramp.

    (v_c, W′_0) come from :func:`estimate_run_energetics_dynamic` (unchanged); `a` is the
    per-run opening-split ramp `v_1 / v_cruise`. Deterministic per-run transform of the run's
    OWN public sectionals (no other horse, no outcome) — same leak status as the scalar/dynamic
    recoveries; the recovered `a` flows through the SAME train-only EB pooling downstream.
    """
    t = np.asarray(section_times, dtype=float)
    d = np.asarray(section_dists, dtype=float)
    v = d / t
    base = estimate_run_energetics_dynamic(section_times, section_dists,
                                           unit=unit, rho=rho, vc_grid=vc_grid)
    vcru = _cruise_velocity(v, t)
    a = float(v[0] / vcru) if vcru > 0 else float("nan")
    return ProfileRunEnergetics(
        v_c=base.v_c, w_prime=base.w_prime, accel=a, v_cruise=vcru,
        v_peak=base.v_peak, v_mean=base.v_mean, fade=base.fade, n_sections=base.n_sections,
    )


def expected_profile(section_dists, v_c: float, w_prime: float, accel: float, *,
                     unit: str = "dist", rho: float = 0.0) -> np.ndarray:
    """GENERATIVE expected K-section velocity profile from latents (v_c, W′_0, a).

    The W′-balance dynamics set the CRUISE+CLOSING shape: a horse runs ABOVE `v_c` while it has
    reserve, spending `W′_0` to hold a velocity above the aerobic ceiling through the cruise and
    into the close, fading toward `v_c` only at the very end as the tank empties. Two facts the
    path-fit foundation (#49, journal 0044/0045) pinned down and this form now honours:

      (1) **The boost MAGNITUDE.** ``W′_0 = Σ(v−v_c)^+·t`` is reserve-DISTANCE (m/s·s = m), the
          extra metres run above the ceiling. It is spent over the NON-OPENING trip (the opening
          is governed by `a`, the start transient — not the reserve), so the time-weighted mean
          velocity lift over the non-opening sections is ``v_c·(W′_0 / L_nonopening)``. The old
          form divided by the FULL trip distance and ramped the share linearly to ZERO at the
          line, which (a) under-scaled the budget and (b) pinned the closing to bare `v_c` —
          together a structural −0.42 m/s level bias on the WHOLE path (gen mean fell *below*
          v_c). Dividing by ``L_nonopening`` recovers the observed lift to ~1e-3 (0044/0045).

      (2) **The spend SHAPE.** The observed per-section excess over `v_c` decays geometrically
          (~halving each non-opening section: 0.063 → 0.031 → 0.015 of v_c, ST 4-section) — the
          reserve depletes multiplicatively, the close still sits ABOVE `v_c` (NOT pinned to it).
          A geometric share ``g_j = q^j`` (q≈0.5), normalised to time-weighted mean 1 over the
          non-opening sections so the realized lift equals the conserved budget, reproduces this.

    The OPENING section is overridden by the ramp: expected v_1 = a · (cruise level). This is the
    mean of the log-velocity Gaussian observation model (:func:`profile_loglik`).

    Returns an array of length K (expected velocity per section).
    """
    d = np.asarray(section_dists, dtype=float)
    k = len(d)
    if k == 0:
        return np.array([])
    w0 = max(w_prime, 0.0)
    if k == 1:
        # single section: the whole reserve is the lift over its own distance
        return np.array([v_c * (1.0 + (w0 / float(d[0]) if d[0] > 0 else 0.0))])
    # The reserve is spent over the NON-OPENING sections (1..k-1). Boost FRACTION = W′_0/L_nonop
    # (the time-weighted mean lift, conserving the reserve-distance budget). The per-section
    # share decays geometrically (q≈0.5: each section ~half the previous), staying > 0 at the
    # close — the horse fades TOWARD v_c, never pins to it.
    l_nonop = float(d[1:].sum())
    boost_frac = w0 / l_nonop if l_nonop > 0 else 0.0
    g = _SPEND_RATIO ** np.arange(k - 1, dtype=float)        # 1, q, q², … (decreasing, last>0)
    tw = d[1:] / v_c if v_c > 0 else d[1:]                   # ~time weights (v_c-uniform)
    gbar = float(np.average(g, weights=tw)) if tw.sum() > 0 else 1.0
    if gbar > 0:
        g = g / gbar                                         # time-weighted mean 1 over non-opening
    prof = np.empty(k)
    prof[1:] = v_c * (1.0 + boost_frac * g)
    # cruise level the opening ramp scales (mirror of `_cruise_velocity` on the EXPECTED profile)
    prof[0] = v_c                                            # placeholder so cruise calc is well-defined
    vt = np.ones_like(prof)
    vcru = _cruise_velocity(prof, vt)
    prof[0] = accel * vcru
    return prof


def profile_loglik(section_times, section_dists, v_c: float, w_prime: float, accel: float,
                   *, sigma: float = 0.06, unit: str = "dist", rho: float = 0.0) -> float:
    """log p(observed K-section profile | v_c, W′_0, a) under a log-velocity Gaussian noise model.

    The observation model is log-velocity Gaussian (≡ log-normal section TIMES, the production
    noise family — finishing time > 0): for each section, `log v_obs ~ N(log v_exp, σ²)`, σ the
    per-section log-velocity noise. Independent across sections. Returns the total log-likelihood.
    """
    t = np.asarray(section_times, dtype=float)
    d = np.asarray(section_dists, dtype=float)
    v = d / t
    exp = expected_profile(d, v_c, w_prime, accel, unit=unit, rho=rho)
    lv, le = np.log(np.clip(v, 1e-6, None)), np.log(np.clip(exp, 1e-6, None))
    r = lv - le
    return float(np.sum(-0.5 * (r / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)))

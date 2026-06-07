"""The consolidated fusion smoother (v2) — one per-horse RTS smoother on the shared clock.

This merges the three validated study layers into a single clean smoother:

- **base** (``studies/cv_fusion_4feed.py``): the per-horse Kalman forward filter + RTS backward
  smoother on the 5 Hz ``race_t`` grid, latent ``z = [s, ṡ, l, l̇]`` (arclen, long-vel, lateral,
  lat-vel); the :class:`Obs` per-horse observation; ``_kf_update``; ``_spearman``.
- **#97 phase-gated physics kernel** (``cv_fusion_physkernel.py``): the LONGITUDINAL process model
  is the EOM template velocity (gate-ramp -> cruise -> finish kick/fade), with per-horse levels
  ``(v_c, d_fin)`` fit from that horse's own observations and a phase-dependent OU pull (STRONG in
  the ~7.5 s gate-ramp window, WEAK in cruise/finish so the order-bearing velocity differences the
  RF rank carries are not smoothed away).
- **#96 constraints** (``cv_fusion_constrained.py``): per-node lateral truncated-Gaussian hard box
  ``0 <= y_lat <= W``, joint cross-horse PAVA isotonic ordering-within-box (gated by per-node
  order-reliability), and the geometry bend down-weight of the aerial lateral metric.

The LATERAL process model is bounds-only (bounded lateral accel) — NO physics SHAPE on lateral:
it is the measured CONTROL signal we are recovering, so a shape prior would be circular.

Bend classification is sourced from :class:`hkjc.reconstruction.geometry.RaceGeom` (which wraps
``core``). This module does NOT re-encode any track facts and does NOT attempt to fix ``core``'s
bend map (the 1650/1800 start-chute approximation is a separate deferred follow-up).

Pure numpy; ``scipy`` (the truncated-Gaussian CDF) is imported lazily inside the box step.
Faithful to the studies' math/thresholds — a consolidation, not a redesign.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RaceGeom

# ============================================================================ GRID / PROCESS NOISE
GRID_HZ = 5.0
DT = 1.0 / GRID_HZ

# --- LATERAL process model (bounds-only; the no-teleport prior). UNCHANGED across the studies. ---
SIGMA_A_LAT = 0.35     # m/s^2  bounded lateral accel (slow sideways drift)

# --- LONGITUDINAL physics kernel (#97; all from the EOM, NOT fitted gains) ---
TAU_RAMP = 2.5         # s  gate-ramp time constant: 1-exp(-t/2.5) hits ~0.95 v_c by ~7.5 s.
GATE_T0 = 0.0          # s  the gun (race_t origin == gate-jump on the shared clock).
FIN_FRAC = 0.12        # last 12% of the distance carries the W'-finish deviation (kick/fade).
RAMP_WINDOW_S = 3.0 * TAU_RAMP   # ~7.5 s gate-ramp phase (strong physics pull).
TAU_REVERT_RAMP = 1.0      # s  STRONG OU pull toward the EOM ramp shape in the gate window.
TAU_REVERT_CRUISE = 12.0   # s  WEAK pull in cruise/finish (light wiggle damping, keeps order).
SIGMA_A_RAMP = 0.15        # m/s^2  small residual accel in the ramp (physics drives it).
SIGMA_A_CRUISE = 0.55      # m/s^2  cruise residual accel (< white 0.8: damps squiggle, keeps order).
VC_LO, VC_HI = 13.0, 19.0       # m/s  plausible equine cruise band (bounds the level fit).
DFIN_LO, DFIN_HI = -0.18, 0.18  # bounded finish deviation (|kick/fade| <= ~18% of cruise).

# --- min lateral separation for the ordering projection (a horse body ~1.1 m). ---
HORSE_WIDTH_M = 1.10


# ============================================================================ OBSERVATION CONTRACT
@dataclass
class Obs:
    """One per-horse observation on the shared ``race_t`` axis.

    A feed contributes an arclen reading, a lateral (``y_lat``) reading, or both, at a given
    ``race_t``, each with its own noise (sd, metres). ``kind`` records what the obs constrains
    (``"arclen"`` / ``"y_lat"`` / ``"both"``), inferred from which value is present; ``feed``
    tags the source for diagnostics. Ported verbatim (field-for-field) from
    ``cv_fusion_4feed.Obs`` so existing build_obs adapters transfer unchanged.
    """

    runner: int
    race_t: float
    arclen: float | None
    ylat: float | None
    feed: str
    sd_arclen: float
    sd_ylat: float

    @property
    def kind(self) -> str:
        if self.arclen is not None and self.ylat is not None:
            return "both"
        return "arclen" if self.arclen is not None else "y_lat"


# ============================================================================ PHYSICS KERNEL (long.)
def _ramp(race_t: np.ndarray, vc: float) -> np.ndarray:
    """Gate-ramp velocity shape 0 -> vc, saturating (the EOM opening transient)."""
    tt = np.clip(np.asarray(race_t, float) - GATE_T0, 0.0, None)
    return vc * (1.0 - np.exp(-tt / TAU_RAMP))


def _kick_shape(arclen: np.ndarray, dist: float) -> np.ndarray:
    """Localised finish-deviation basis over the last ``FIN_FRAC`` of the race (half-cosine bump)."""
    a = np.asarray(arclen, float)
    x0 = (1.0 - FIN_FRAC) * dist
    u = np.clip((a - x0) / (FIN_FRAC * dist), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def template_velocity(race_t, arclen, dist: float, vc: float, d_fin: float) -> np.ndarray:
    """EOM-shaped template velocity ``v_EOM = v_c * ramp(t) * (1 + d_fin*kick(arclen))``."""
    return _ramp(race_t, vc) * (1.0 + d_fin * _kick_shape(arclen, dist))


def fit_levels(grid: np.ndarray, obs: list[Obs], dist: float) -> tuple[float, float, np.ndarray]:
    """Fit the EOM-shape LEVELS ``(v_c, d_fin)`` + a coarse arclen(t) anchor from THIS horse's feeds.

    No per-horse physiology, no other horse, no official outcome — only the feeds that drive the
    smoother. ``v_c`` = robust median raw velocity over the cruise band (clamped); ``d_fin`` =
    signed late-window deviation (clamped). Returns ``(v_c, d_fin, s_anchor(grid))``.
    """
    arc_obs = sorted((o.race_t, o.arclen) for o in obs if o.arclen is not None)
    if len(arc_obs) < 3:
        # not enough arclen support: flat default cruise; signal the caller to seed s_anchor.
        return 16.5, 0.0, np.full(len(grid), np.nan)
    ot = np.array([t for t, _ in arc_obs])
    oa = np.array([a for _, a in arc_obs])
    s_anchor = np.interp(grid, ot, oa, left=oa[0], right=oa[-1])
    s_anchor = np.maximum.accumulate(s_anchor)
    v_raw = np.gradient(s_anchor, DT)

    cruise_m = (grid > GATE_T0 + 3.0 * TAU_RAMP) & (s_anchor < (1.0 - FIN_FRAC) * dist)
    vc = float(np.median(v_raw[cruise_m])) if cruise_m.sum() >= 3 else float(np.median(v_raw))
    vc = float(np.clip(vc, VC_LO, VC_HI))

    fin_m = s_anchor >= (1.0 - FIN_FRAC) * dist
    if fin_m.sum() >= 2 and vc > 1e-3:
        d_fin = float(np.clip(float(np.median(v_raw[fin_m])) / vc - 1.0, DFIN_LO, DFIN_HI))
    else:
        d_fin = 0.0
    return vc, d_fin, s_anchor


# ============================================================================ KALMAN PRIMITIVES
def _kf_update(x: np.ndarray, p: np.ndarray, h: np.ndarray, z: float, r: float):
    """Scalar Kalman measurement update (port of ``cv_fusion_4feed._kf_update``)."""
    y = z - h @ x
    s = h @ p @ h + r
    k = (p @ h) / s
    x_new = x + k * y
    p_new = (np.eye(len(x)) - np.outer(k, h)) @ p
    return x_new, p_new


def _lat_block(dt: float, sa: float):
    """Lateral 1-D constant-velocity transition F (2x2) + white-accel process cov Q (2x2)."""
    f = np.array([[1.0, dt], [0.0, 1.0]])
    q = (sa ** 2) * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return f, q


# ============================================================================ PER-HORSE SMOOTHER
def smooth_horse(grid: np.ndarray, obs: list[Obs], dist: float,
                 *, race_t_to_arclen=None) -> dict[str, np.ndarray]:
    """RTS smoother for ONE horse on the shared ``race_t`` grid.

    Longitudinal state ``[s, ṡ]``: ``s`` integrates ``ṡ``; ``ṡ`` is pulled toward the data-fit EOM
    template velocity by a PHASE-DEPENDENT OU mean-reversion (the #97 physics kernel — strong in
    the gate-ramp window, weak after). Lateral state ``[l, l̇]``: the bounds-only white-accel block.
    Returns smoothed ``s, l`` + their sds + per-node coverage flags. The #96 constraints are applied
    afterwards by :func:`apply_constraints` (the field-joint step).
    """
    n = len(grid)
    vc, d_fin, s_anchor = fit_levels(grid, obs, dist)
    if not np.all(np.isfinite(s_anchor)):
        # fit_levels had too little arclen support: seed s_anchor from race_t -> arclen if available.
        s_anchor = (np.asarray(race_t_to_arclen(grid), float)
                    if race_t_to_arclen is not None
                    else np.full(n, np.median([o.arclen for o in obs if o.arclen is not None] or [0.0])))
    v_tmpl = template_velocity(grid, s_anchor, dist, vc, d_fin)

    # phase-dependent OU pull (strong gate-ramp prior; weak order-bearing cruise/finish prior).
    in_ramp = (grid - grid[0]) <= RAMP_WINDOW_S
    tau_rev = np.where(in_ramp, TAU_REVERT_RAMP, TAU_REVERT_CRUISE)
    sig_a = np.where(in_ramp, SIGMA_A_RAMP, SIGMA_A_CRUISE)
    beta = np.minimum(DT / tau_rev, 0.9)

    f_lat, q_lat = _lat_block(DT, SIGMA_A_LAT)

    # bin observations to nearest grid node.
    obs_by_node: dict[int, list[Obs]] = {}
    for o in obs:
        k = int(round((o.race_t - grid[0]) / DT))
        if 0 <= k < n:
            obs_by_node.setdefault(k, []).append(o)

    first = min(obs, key=lambda o: o.race_t)
    s0 = first.arclen if first.arclen is not None else float(s_anchor[0])
    l0 = first.ylat if first.ylat is not None else 0.5 * _width_guess(obs)
    v0 = float(v_tmpl[0])
    x = np.array([s0, v0, l0, 0.0])
    p = np.diag([50.0 ** 2, 3.0 ** 2, 12.0 ** 2, 2.0 ** 2])

    # time-varying long. transition + process cov + affine input (the OU template offset).
    f_all = np.zeros((n, 4, 4))
    q_all = np.zeros((n, 4, 4))
    u = np.zeros((n, 4))
    for k in range(n):
        b = float(beta[k])
        sa = float(sig_a[k])
        f_long = np.array([[1.0, DT], [0.0, 1.0 - b]])
        qpos = (sa ** 2) * (DT ** 3) / 3.0
        qcross = (sa ** 2) * (DT ** 2) / 2.0
        qvel = (sa ** 2) * DT
        f_all[k, :2, :2], f_all[k, 2:, 2:] = f_long, f_lat
        q_all[k, :2, :2] = np.array([[qpos, qcross], [qcross, qvel]])
        q_all[k, 2:, 2:] = q_lat
        u[k, 1] = b * float(v_tmpl[k])

    xp = np.zeros((n, 4))
    pp = np.zeros((n, 4, 4))
    xf = np.zeros((n, 4))
    pf = np.zeros((n, 4, 4))

    for k in range(n):
        if k == 0:
            x_pred, p_pred = x, p
        else:
            x_pred = f_all[k - 1] @ xf[k - 1] + u[k - 1]
            p_pred = f_all[k - 1] @ pf[k - 1] @ f_all[k - 1].T + q_all[k - 1]
        xp[k], pp[k] = x_pred, p_pred
        x_upd, p_upd = x_pred.copy(), p_pred.copy()
        for o in obs_by_node.get(k, []):
            if o.arclen is not None:
                h = np.array([1.0, 0.0, 0.0, 0.0])
                x_upd, p_upd = _kf_update(x_upd, p_upd, h, o.arclen, o.sd_arclen ** 2)
            if o.ylat is not None:
                h = np.array([0.0, 0.0, 1.0, 0.0])
                x_upd, p_upd = _kf_update(x_upd, p_upd, h, o.ylat, o.sd_ylat ** 2)
        xf[k], pf[k] = x_upd, p_upd

    # RTS backward pass (per-step F_k; the affine input cancels in the smoother gain).
    xs = xf.copy()
    ps = pf.copy()
    for k in range(n - 2, -1, -1):
        p_pred = pp[k + 1]
        fk = f_all[k]
        try:
            c = pf[k] @ fk.T @ np.linalg.inv(p_pred)
        except np.linalg.LinAlgError:
            c = pf[k] @ fk.T @ np.linalg.pinv(p_pred)
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - p_pred) @ c.T

    return {
        "s": xs[:, 0], "sd_s": np.sqrt(np.clip(ps[:, 0, 0], 0, None)),
        "v": xs[:, 1],
        "l": xs[:, 2], "sd_l": np.sqrt(np.clip(ps[:, 2, 2], 0, None)),
        "v_tmpl": v_tmpl,
        "node_has_obs": np.array([k in obs_by_node for k in range(n)]),
        "node_has_lat": np.array(
            [any(o.ylat is not None for o in obs_by_node.get(k, [])) for k in range(n)]),
        "_vc_fit": np.array([vc]), "_dfin_fit": np.array([d_fin]),
    }


def _width_guess(obs: list[Obs]) -> float:
    """Fallback track width (m) from the obs' lateral spread (only used to seed l0 when no obs)."""
    ys = [o.ylat for o in obs if o.ylat is not None]
    return (2.0 * float(np.max(ys))) if ys else 22.5


# ============================================================================ CONSTRAINTS (#96 lateral)
def _trunc_gauss_moments(mu: np.ndarray, sd: np.ndarray, lo: float, hi: float):
    """Moment-match a Gaussian(mu,sd) truncated to ``[lo,hi]`` -> (mean, sd) of the truncated law.

    Pulls the mean inside the box AND shrinks the sd by the removed tail mass (NOT a naive clip).
    """
    from math import sqrt

    from scipy.special import ndtr

    sd = np.maximum(sd, 1e-6)
    a = (lo - mu) / sd
    b = (hi - mu) / sd

    def _phi(x):
        return np.exp(-0.5 * x * x) / sqrt(2.0 * np.pi)

    za, zb = ndtr(a), ndtr(b)
    z = np.maximum(zb - za, 1e-9)
    pa, pb = _phi(a), _phi(b)
    mean = mu + (pa - pb) / z * sd
    term = 1.0 + (a * pa - b * pb) / z - ((pa - pb) / z) ** 2
    var = sd * sd * np.clip(term, 1e-6, None)
    return np.clip(mean, lo, hi), np.sqrt(var)


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators (weighted) isotonic regression onto a non-decreasing sequence."""
    val = y.astype(float).copy()
    wt = w.astype(float).copy()
    vals: list[float] = []
    wts: list[float] = []
    cnts: list[int] = []
    for i in range(len(val)):
        v, ww, c = val[i], wt[i], 1
        while vals and vals[-1] > v:
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww = pw + ww
            c = pc + c
        vals.append(v)
        wts.append(ww)
        cnts.append(c)
    out = np.empty(len(val))
    j = 0
    for v, c in zip(vals, cnts, strict=True):
        out[j:j + c] = v
        j += c
    return out


def _project_ordered_box(mu: np.ndarray, prec: np.ndarray, order: np.ndarray,
                         lo: float, hi: float, s_min: float):
    """Precision-weighted projection of ``mu`` (ascending ``order``) onto the feasible set
    ``{ lo <= x_(1), x_(k+1)-x_(k) >= s_min, x_(N) <= hi }`` (a weighted isotonic QP via PAVA).

    Returns the projected positions in ORIGINAL index space + the per-point feasible half-width
    (used to shrink the posterior sd). Port of ``cv_fusion_constrained._project_ordered_box``.
    """
    n = len(mu)
    if n == 0:
        return mu.copy(), np.zeros(0)
    m = mu[order]
    w = np.maximum(prec[order], 1e-9)
    shift = np.arange(n) * s_min
    y_t = m - shift
    ylo = lo
    yhi = hi - (n - 1) * s_min
    if yhi < ylo:  # field cannot fit at s_min — relax to an even spread.
        s_eff = (hi - lo) / max(n - 1, 1)
        x_even = lo + np.arange(n) * s_eff
        out = np.empty(n)
        out[order] = x_even
        return out, np.full(n, 0.5 * s_eff)
    yhat = _pava(y_t, w)
    yhat = np.clip(yhat, ylo, yhi)
    yhat = np.maximum.accumulate(yhat)
    yhat = np.clip(yhat, None, yhi)
    x_sorted = yhat + shift
    out = np.empty(n)
    out[order] = x_sorted
    hw = np.empty(n)
    for k in range(n):
        left = x_sorted[k] - (x_sorted[k - 1] + s_min) if k > 0 else x_sorted[k] - lo
        right = (x_sorted[k + 1] - s_min) - x_sorted[k] if k < n - 1 else hi - x_sorted[k]
        hw[order[k]] = 0.5 * max(min(left, right), 0.0)
    return out, hw


def apply_constraints(smoothed: dict, grid: np.ndarray, runners: list[int], geom: RaceGeom,
                      *, do_box=True, do_order=True) -> dict:
    """Apply the #96 lateral constraints per grid node. Longitudinal channel is NEVER touched.

    (1) HARD BOX ``0 <= y_lat <= W`` via per-node truncated-Gaussian moment-match (caps the
        unanchored bend posteriors AND shrinks their sds), with ``W = geom.width``.
    (2) JOINT cross-horse ORDERING-WITHIN-BOX: a precision-weighted PAVA projection onto the
        ordered, min-separated feasible set, with the constrained sd GATED by the per-node order
        reliability (the fraction of co-present horses with a real lateral observation this frame).
    Returns a deep-copied smoothed dict with ``l`` / ``sd_l`` updated.
    """
    n = len(grid)
    width = geom.width
    s_min = HORSE_WIDTH_M
    out = {rn: {k: v.copy() for k, v in smoothed[rn].items()} for rn in runners}

    # the box is a physical fact at EVERY in-window node, not only where an obs lands this frame.
    cover: dict[int, tuple[int, int]] = {}
    for rn in runners:
        idx = np.where(out[rn]["node_has_obs"])[0]
        cover[rn] = (int(idx.min()), int(idx.max())) if len(idx) else (n, -1)

    for k in range(n):
        present = [rn for rn in runners if cover[rn][0] <= k <= cover[rn][1]]
        if not present:
            continue
        mu = np.array([out[rn]["l"][k] for rn in present])
        sd = np.array([out[rn]["sd_l"][k] for rn in present])

        if do_box:
            mu_b, sd_b = _trunc_gauss_moments(mu, sd, 0.0, width)
        else:
            mu_b, sd_b = mu.copy(), sd.copy()

        if do_order and len(present) >= 2:
            prec = 1.0 / np.maximum(sd_b, 1e-3) ** 2
            order = np.argsort(mu_b, kind="stable")
            mu_o, hw = _project_ordered_box(mu_b, prec, order, 0.0, width, s_min)
            has_lat_now = np.array([smoothed[rn]["node_has_lat"][k] for rn in present], float)
            rel = float(has_lat_now.mean())  # 0..1 order-reliability at this node
            sd_tight = np.maximum(hw, 0.5 * s_min)
            sd_o = rel * np.minimum(sd_b, sd_tight) + (1.0 - rel) * sd_b
            sd_o = np.where(has_lat_now > 0, np.minimum(sd_o, sd_b), sd_b)
            mu_b, sd_b = mu_o, sd_o

        for i, rn in enumerate(present):
            out[rn]["l"][k] = mu_b[i]
            out[rn]["sd_l"][k] = sd_b[i]
    return out


# ============================================================================ FIELD ORCHESTRATION
def smooth_field(obs_by_runner: dict[int, list[Obs]], grid: np.ndarray, geom: RaceGeom,
                 *, do_box=True, do_order=True, race_t_to_arclen=None) -> dict[int, dict]:
    """Smooth every runner then apply the joint lateral constraints — the public fusion entry point.

    ``obs_by_runner`` maps runner number -> its :class:`Obs` list (already on the shared ``race_t``
    axis). ``geom`` supplies the box width and the bend down-weight. Returns
    ``{runner -> dict(race_t, arclen, y_lat, v, sd_arclen, sd_ylat, node_has_obs, node_has_lat, ...)}``.

    NOTE on the bend down-weight: ``geom.bend_weight(arclen)`` (full on straights, 0 on bends) is
    applied as an INPUT re-tag of the aerial/drone lateral-metric noise BEFORE smoothing — see
    :func:`bend_downweight_obs`, which the orchestrator calls when building the obs. ``smooth_field``
    consumes whatever sds it is given; here it only runs the smoother + the per-node constraints.
    """
    runners = sorted(obs_by_runner)
    dist = float(geom.distance)
    raw = {
        rn: smooth_horse(grid, obs_by_runner[rn], dist, race_t_to_arclen=race_t_to_arclen)
        for rn in runners
    }
    sm = apply_constraints(raw, grid, runners, geom, do_box=do_box, do_order=do_order)
    out: dict[int, dict] = {}
    for rn in runners:
        out[rn] = {
            "race_t": grid,
            "arclen": sm[rn]["s"],
            "y_lat": sm[rn]["l"],
            "v": sm[rn]["v"],
            "sd_arclen": sm[rn]["sd_s"],
            "sd_ylat": sm[rn]["sd_l"],
            "node_has_obs": sm[rn]["node_has_obs"],
            "node_has_lat": sm[rn]["node_has_lat"],
            "vc_fit": float(sm[rn]["_vc_fit"][0]),
            "dfin_fit": float(sm[rn]["_dfin_fit"][0]),
        }
    return out


def bend_downweight_obs(obs: list[Obs], geom: RaceGeom, lateral_feeds=("aerial", "drone"),
                        *, bend_inflation: float = 2.0) -> None:
    """Inflate the lateral-metric sd of bend-degraded aerial/drone obs IN PLACE (the #96 geometry map).

    On a bend the render foreshortens and the aerial lateral METRIC degrades to noise (CV_RECON §7),
    so the obs' ``sd_ylat`` is inflated where ``geom.bend_weight(arclen) == 0`` (a turn). Patrol (the
    metric anchor), RF and the draw anchor are untouched. Faithful to the constrained study's
    geometry re-tag (there it interpolated between the straight and an inflated run-in sd).
    """
    for o in obs:
        if o.ylat is None or o.arclen is None or o.feed not in lateral_feeds:
            continue
        bf = 1.0 - float(geom.bend_weight(o.arclen))  # 1 on a bend, 0 on a straight
        o.sd_ylat = o.sd_ylat * (1.0 + (bend_inflation - 1.0) * bf)


# ============================================================================ VALIDATION HELPER
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (port of ``cv_fusion_4feed._spearman``; for validation only)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")

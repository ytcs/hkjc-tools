"""TASK #97 — PHYSICS-shaped (EOM) LONGITUDINAL motion kernel for the 4-feed fusion smoother.

BASE = studies/cv_fusion_constrained.py (#96: the hard lateral box + joint ordering-within-box +
track-geometry down-weight + draw->y_lat(0) anchor). This module REPLACES ONLY the LONGITUDINAL
process model of the per-horse RTS smoother — the generic WHITE-ACCELERATION Gaussian motion prior
(latent vel random-walks freely, sigma_a ~0.8 m/s^2) — with a PHYSICS-shaped kernel that encodes the
SHAPE of the project's own equation of motion (src/hkjc/model/dynamics.py): a gate-RAMP from ~0 to a
cruise plateau v_c, a CRUISE phase, and a single bounded FINISH deviation (the W'-kick / fade). The
LATERAL channel is untouched (#96 box + ordering + bounded lateral accel) — lateral is the measured
CONTROL signal we are recovering; imposing a physics shape on it would be circular.

WHY (the artifact). The recon top-plot (gap-behind-leader vs race_t) shows (a) a wild 0-20s swing and
(b) an overall squiggle. Diagnosis (agreed with the user): the white-accel kernel lets the latent
LONGITUDINAL velocity random-walk freely and chase the discrete rank-based RF observations, whereas
real longitudinal velocity is SMOOTH and LOW-DIMENSIONAL — it follows the EOM (gate-ramp -> cruise ->
finish kick/fade). Encoding that SHAPE as the process model kills the wiggles and pins the gate
transient, WITHOUT fabricating per-horse physiology.

THE KERNEL (longitudinal velocity SHAPE prior; level fit from DATA — NO per-horse-theta circularity):
  The EOM velocity profile (verified by solving dynamics.solve_eom for v_c~17 m/s, routes/sprints):
    v_EOM(t) = v_c * ramp(t) * (1 + d_fin * kick(t))
    ramp(t)  = 1 - exp(-(t - t0)^+ / TAU_RAMP)     # saturating gate-ramp 0 -> v_c (~0.95 v_c by ~6s)
    kick(t)  = a localised deviation over the last FIN_FRAC of the race (the W'-kick or fade),
               a single signed degree of freedom d_fin (positive = kick, negative = fade).
  This is a LOW-DIMENSIONAL smooth basis whose SHAPE matches the EOM, NOT a free random walk and NOT a
  frozen forward-sim of estimated physiology. The LEVELS (v_c, d_fin, and the absolute arclen) are fit
  from EACH horse's OWN feed observations in a pre-pass; the feeds still drive the fit, the physics
  only regularises the velocity SHAPE.

  Mechanism in the smoother: the longitudinal velocity sub-state is given a strong MEAN-REVERSION
  toward the data-fit template velocity v_EOM(t) (an Ornstein-Uhlenbeck pull, time constant
  TAU_REVERT), replacing the free white-accel random walk. Position s still integrates velocity and is
  still corrected by every feed (aerial odometer / RF rank+margins / drone / patrol); the difference is
  that velocity can no longer wander to chase a noisy rank obs — it is pulled back to the smooth EOM
  shape, and only a small bounded residual accel (SIGMA_A_RESID) lets the data move the shape's level.

PAIRED WHITE vs PHYSICS: the SAME obs set + same lateral constraints; only the longitudinal process
model differs (white-accel == cv_fusion_4feed.smooth_horse; physics == this module). Validation
(official = validation-only; draw = allowed initial condition): the no-regression battery (winner 3/3,
per-pole order rho, finish-margin MAE, lateral sign 3/3) MUST hold; plus the smoothness deltas
(longitudinal-velocity total-variation + jerk RMS + gap-curve roughness, white vs physics).

DISCIPLINE: study-only; src/hkjc untouched; banked + #96 _constrained parquets NOT mutated (writes
*_physkernel-suffixed); official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_fusion_physkernel.py
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, "studies")

import cv_fusion_4feed as F1  # noqa: E402,N812
import cv_fusion_constrained as C96  # noqa: E402,N812

# ============================================================================ PHYSICS-KERNEL CONSTANTS
# All from the EOM (src/hkjc/model/dynamics.py), NOT fitted gains. Verified by solving solve_eom for a
# representative v_c~17 m/s horse: the ramp reaches 0.95 v_c by ~6s / ~70m; cruise is a flat plateau;
# the terminal kick/fade lives in the last ~5-10% of the distance.
TAU_RAMP = 2.5      # s  gate-ramp time constant: 1-exp(-t/2.5) hits 0.95 at t~7.5s (matches EOM ~6-7s).
GATE_T0 = 0.0       # s  the gun (race_t origin is the gate-jump on the shared clock).
FIN_FRAC = 0.12     # last 12% of the distance carries the W'-finish deviation (kick or fade window).
# PHASE-DEPENDENT pull strength (the #97 structural lesson — see the journal). The EOM-shape velocity
# prior is applied STRONGLY only in the GATE-RAMP window, where every horse is accelerating from the
# gate and there is NO overtaking / running-position signal yet (the first official call is at
# 200-450m). In CRUISE + FINISH the pull is WEAK, because there the per-horse VELOCITY DIFFERENCES
# *are* the order signal the RF rank feed carries (a horse moving through the field is exactly a
# per-horse velocity excess) — a stiff shared-cruise template would smooth that real signal away
# (it erases winner #3's 7->1 climb). So: pin the gate transient with physics, leave the order-bearing
# mid-race velocity to the data.
RAMP_WINDOW_S = 3.0 * TAU_RAMP   # s  the gate-ramp phase (strong pull) ~ 7.5s (~0.95 v_c reached).
TAU_REVERT_RAMP = 1.0   # s  STRONG OU pull toward the EOM ramp shape during the gate window.
TAU_REVERT_CRUISE = 12.0  # s  WEAK pull in cruise/finish — regularises wiggle lightly, keeps order.
SIGMA_A_RAMP = 0.15     # m/s^2  small residual accel in the ramp (physics drives it).
SIGMA_A_CRUISE = 0.55   # m/s^2  cruise residual accel: < white 0.8 (damps the squiggle) but loose
#                         enough that the data's order-bearing velocity differences survive.

# fit guards (level fit from data; physical bounds, NOT per-horse physiology).
VC_LO, VC_HI = 13.0, 19.0      # m/s  plausible equine cruise band (v_c~17 +/- ; bounds the level fit).
DFIN_LO, DFIN_HI = -0.18, 0.18  # bounded finish deviation (|kick/fade| <= ~18% of cruise — physical).


def _ramp(race_t: np.ndarray, vc: float) -> np.ndarray:
    """Gate-ramp velocity shape: 0 -> vc, saturating (the EOM opening transient)."""
    tt = np.clip(np.asarray(race_t, float) - GATE_T0, 0.0, None)
    return vc * (1.0 - np.exp(-tt / TAU_RAMP))


def _kick_shape(arclen: np.ndarray, dist: float) -> np.ndarray:
    """Localised finish-deviation basis over the last FIN_FRAC of the race, peaking at the line.

    A smooth half-cosine bump rising from 0 at (1-FIN_FRAC)*dist to 1 at the line — the single d.o.f.
    that lets the template kick UP (W'-burn) or fade DOWN near the finish. Shape only; the signed
    magnitude d_fin is fit from data.
    """
    a = np.asarray(arclen, float)
    x0 = (1.0 - FIN_FRAC) * dist
    u = np.clip((a - x0) / (FIN_FRAC * dist), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))   # 0 at x0, 1 at the line, smooth


def template_velocity(race_t: np.ndarray, arclen: np.ndarray, dist: float,
                      vc: float, d_fin: float) -> np.ndarray:
    """The EOM-shaped template velocity v_EOM(t) = v_c*ramp(t)*(1 + d_fin*kick(arclen)).

    Levels (vc, d_fin) come from the per-horse data pre-fit; the shape is the EOM's.
    """
    return _ramp(race_t, vc) * (1.0 + d_fin * _kick_shape(arclen, dist))


# ============================================================================ PER-HORSE LEVEL PRE-FIT
def fit_levels(grid: np.ndarray, obs: list, dist: float) -> tuple[float, float, np.ndarray]:
    """Fit the EOM-shape LEVELS (v_c, d_fin) + a per-horse coarse arclen(t) anchor from THIS horse's
    own arclen observations. NO per-horse physiology, NO other horse, NO official outcome — only the
    feeds that already drive the smoother.

    Method (robust + cheap): take this horse's arclen observations (aerial odometer / RF margins+rank /
    drone / patrol), bin to a coarse arclen(t) curve, differentiate to a raw velocity, then:
      * v_c   = robust median of the raw velocity over the CRUISE band (after the ramp settles,
                before the finish window) — the data's own cruise level, clamped to [VC_LO,VC_HI].
      * d_fin = (median late-window velocity / v_c - 1), clamped to [DFIN_LO,DFIN_HI] — the data's own
                finish deviation, the single signed kick/fade d.o.f.
    Returns (v_c, d_fin, s_anchor(grid)) — s_anchor a coarse monotone arclen guess used only to seed s.
    """
    arc_obs = [(o.race_t, o.arclen) for o in obs if o.arclen is not None]
    if len(arc_obs) < 3:
        return 16.5, 0.0, F1.race_t_to_arclen(grid)
    arc_obs.sort()
    ot = np.array([t for t, _ in arc_obs])
    oa = np.array([a for _, a in arc_obs])
    # coarse monotone arclen(t): bin to the grid, cumulative-max to kill backward rank-noise jitter.
    s_anchor = np.interp(grid, ot, oa, left=oa[0], right=oa[-1])
    s_anchor = np.maximum.accumulate(s_anchor)
    # raw velocity from the coarse curve.
    v_raw = np.gradient(s_anchor, F1.DT)

    # cruise band: after the ramp (race_t > ~3*TAU_RAMP) and before the finish window (arclen<(1-FIN)).
    cruise_m = (grid > GATE_T0 + 3.0 * TAU_RAMP) & (s_anchor < (1.0 - FIN_FRAC) * dist)
    vc = float(np.median(v_raw[cruise_m])) if cruise_m.sum() >= 3 else float(np.median(v_raw))
    vc = float(np.clip(vc, VC_LO, VC_HI))

    # finish deviation: late-window velocity vs cruise.
    fin_m = s_anchor >= (1.0 - FIN_FRAC) * dist
    if fin_m.sum() >= 2 and vc > 1e-3:
        v_fin = float(np.median(v_raw[fin_m]))
        d_fin = float(np.clip(v_fin / vc - 1.0, DFIN_LO, DFIN_HI))
    else:
        d_fin = 0.0
    return vc, d_fin, s_anchor


# ============================================================================ PHYSICS-KERNEL SMOOTHER
def _lat_block(dt: float, sa: float):
    """Lateral 1-D constant-velocity transition + white-accel cov (UNCHANGED from #96/F1)."""
    f = np.array([[1.0, dt], [0.0, 1.0]])
    q = (sa ** 2) * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return f, q


def smooth_horse_physkernel(grid: np.ndarray, obs: list, dist: float) -> dict:
    """RTS smoother with the PHYSICS-shaped LONGITUDINAL kernel + the #96 lateral block.

    Longitudinal state [s, s_dot]: s integrates s_dot (constant-velocity transition); s_dot is pulled
    toward the data-fit EOM template velocity v_EOM(t) by an OU mean-reversion (the physics SHAPE
    prior), with only a SMALL residual process noise (SIGMA_A_RESID) so the data can move the level but
    velocity cannot random-walk freely. This makes the long. process AFFINE (a control/offset term),
    handled in a Kalman filter+RTS smoother with a constant input u_k. Lateral [l, l_dot] is the
    unchanged white-accel block (then the #96 constraints are applied by the caller).
    """
    n = len(grid)
    vc, d_fin, s_anchor = fit_levels(grid, obs, dist)
    v_tmpl = template_velocity(grid, s_anchor, dist, vc, d_fin)   # EOM template velocity on the grid

    # --- longitudinal AFFINE transition with a PHASE-DEPENDENT OU pull toward v_tmpl ---
    # continuous: ds=v dt ; dv = -(v - v_tmpl)/TAU_REVERT(t) dt + sigma(t) dW.
    # discretise per step k:  s_{k+1}=s_k+v_k*dt ; v_{k+1}=(1-beta_k)*v_k + beta_k*v_tmpl[k] + noise.
    # beta_k / sigma_k switch between the STRONG gate-ramp prior and the WEAK cruise/finish prior so the
    # physics pins the opening transient but leaves the order-bearing mid-race velocity to the data.
    in_ramp = (grid - grid[0]) <= RAMP_WINDOW_S
    tau_rev = np.where(in_ramp, TAU_REVERT_RAMP, TAU_REVERT_CRUISE)
    sig_a = np.where(in_ramp, SIGMA_A_RAMP, SIGMA_A_CRUISE)
    beta = np.minimum(F1.DT / tau_rev, 0.9)   # per-node pull weight

    fT, qT = _lat_block(F1.DT, F1.SIGMA_A_LAT)

    # bin observations to nearest grid node.
    obs_by_node: dict[int, list] = {}
    for o in obs:
        k = int(round((o.race_t - grid[0]) / F1.DT))
        if 0 <= k < n:
            obs_by_node.setdefault(k, []).append(o)

    first = min(obs, key=lambda o: o.race_t)
    s0 = first.arclen if first.arclen is not None else float(F1.race_t_to_arclen(first.race_t))
    l0 = first.ylat if first.ylat is not None else F1.TRACK_WIDTH / 2.0
    # seed velocity from the template at the start (ramp value), not a flat 16.5 — the gate transient.
    v0 = float(v_tmpl[0])
    x = np.array([s0, v0, l0, 0.0])
    p = np.diag([50.0 ** 2, 3.0 ** 2, 12.0 ** 2, 2.0 ** 2])

    # per-step transition F_k, process cov Q_k, and affine input u_k (all time-varying on the long block)
    f_all = np.zeros((n, 4, 4))
    q_all = np.zeros((n, 4, 4))
    u = np.zeros((n, 4))
    for k in range(n):
        b = float(beta[k])
        sa = float(sig_a[k])
        fL = np.array([[1.0, F1.DT], [0.0, 1.0 - b]])
        qpos = (sa ** 2) * (F1.DT ** 3) / 3.0
        qcross = (sa ** 2) * (F1.DT ** 2) / 2.0
        qvel = (sa ** 2) * F1.DT
        f_all[k, :2, :2], f_all[k, 2:, 2:] = fL, fT
        q_all[k, :2, :2] = np.array([[qpos, qcross], [qcross, qvel]])
        q_all[k, 2:, 2:] = qT
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
                x_upd, p_upd = F1._kf_update(x_upd, p_upd, h, o.arclen, o.sd_arclen ** 2)
            if o.ylat is not None:
                h = np.array([0.0, 0.0, 1.0, 0.0])
                x_upd, p_upd = F1._kf_update(x_upd, p_upd, h, o.ylat, o.sd_ylat ** 2)
        xf[k], pf[k] = x_upd, p_upd

    # RTS backward pass (per-step F_k; the affine input cancels in the smoother gain — F1 form).
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
        # scalars kept OUT of the array dict (apply_constraints .copy()s every value): stash on attrs.
        "_vc_fit": np.array([vc]), "_dfin_fit": np.array([d_fin]),
    }


# ============================================================================ SMOOTHNESS METRICS
def _vel_smoothness(s: np.ndarray, mask: slice) -> tuple[float, float, float]:
    """Return (cruise_vel_TV, jerk_RMS, post-ramp 0-20s swing) for the longitudinal channel.

    The TV / swing are measured AFTER the gate-ramp window (RAMP_WINDOW_S) so the legitimate gate
    transient (velocity 0->v_c, a real ~17 m/s rise) is NOT counted as roughness — what we want to
    measure is the post-ramp CRUISE squiggle (the artifact), not the physical ramp.
    """
    v = np.gradient(s[mask], F1.DT)
    if len(v) < 8:
        return 0.0, 0.0, 0.0
    # drop the gate-ramp nodes (first RAMP_WINDOW_S seconds == RAMP_WINDOW_S/DT nodes).
    nramp = int(RAMP_WINDOW_S / F1.DT)
    vc = v[nramp:] if len(v) > nramp + 4 else v
    jerk = np.gradient(np.gradient(vc, F1.DT), F1.DT)
    tv = float(np.sum(np.abs(np.diff(vc))))
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    # post-ramp 0-20s swing (the next 20s after the ramp settles).
    early = vc[:min(100, len(vc))]
    swing = float(early.max() - early.min()) if len(early) else 0.0
    return tv, jerk_rms, swing


def gap_roughness(smoothed: dict, grid: np.ndarray, runners: list) -> float:
    """Mean over horses of the total-variation of the gap-behind-leader curve (the top-plot squiggle).

    Lower == smoother. The leader arclen at each node is the field max; each horse's gap = s - lead.
    """
    lead = np.full(len(grid), -np.inf)
    for rn in runners:
        has = smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        lead[m] = np.maximum(lead[m], smoothed[rn]["s"][m])
    tvs = []
    for rn in runners:
        s, has = smoothed[rn]["s"], smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if len(idx) < 5:
            continue
        m = slice(idx.min(), idx.max() + 1)
        gap = s[m] - lead[m]
        # roughness = TV of the SECOND difference (curvature) — penalises wiggle, not the trend.
        tvs.append(float(np.sum(np.abs(np.diff(gap, 2)))))
    return float(np.mean(tvs)) if tvs else float("nan")


# ============================================================================ RACE DRIVER (paired)
def run_race(name: str) -> dict:
    """Build the SAME #96 obs set, run WHITE (F1.smooth_horse) and PHYSICS smoothers, apply the #96
    lateral constraints to both, validate, and return everything for the figure."""
    cfg, draw_anchor, off, rf_loader = C96.setup_race(name)
    dist = float(cfg.dist)
    bands = C96.off_bands(cfg)

    all_obs, _aer_df, _frt, _farc = C96.build_obs(
        cfg, draw_anchor, use_geometry=True, rf_loader=rf_loader)
    runners = sorted({o.runner for o in all_obs})
    t_lo = min(o.race_t for o in all_obs)
    t_hi = max(o.race_t for o in all_obs)
    grid = np.arange(t_lo, t_hi + F1.DT, F1.DT)

    # WHITE kernel (the #96 base: F1 smoother + #96 constraints).
    sm_white_raw = {rn: F1.smooth_horse(grid, [o for o in all_obs if o.runner == rn])
                    for rn in runners}
    sm_white = C96.apply_constraints(sm_white_raw, grid, runners, all_obs, cfg,
                                     do_box=True, do_order=True)

    # PHYSICS kernel (this module's long. smoother + the SAME #96 lateral constraints).
    sm_phys_raw = {rn: smooth_horse_physkernel(grid, [o for o in all_obs if o.runner == rn], dist)
                   for rn in runners}
    sm_phys = C96.apply_constraints(sm_phys_raw, grid, runners, all_obs, cfg,
                                    do_box=True, do_order=True)

    ev_white = C96.evaluate(sm_white, grid, runners, off, cfg, bands)
    ev_phys = C96.evaluate(sm_phys, grid, runners, off, cfg, bands)

    # smoothness deltas (longitudinal velocity TV + jerk RMS + 0-20s swing + gap-curve roughness).
    def field_smooth(sm):
        tvs, jerks, swings = [], [], []
        for rn in runners:
            has = sm[rn]["node_has_obs"]
            idx = np.where(has)[0]
            if len(idx) < 5:
                continue
            m = slice(idx.min(), idx.max() + 1)
            tv, jr, sw = _vel_smoothness(sm[rn]["s"], m)
            tvs.append(tv)
            jerks.append(jr)
            swings.append(sw)
        return (float(np.mean(tvs)), float(np.mean(jerks)), float(np.mean(swings)),
                gap_roughness(sm, grid, runners))

    sm_white_metrics = field_smooth(sm_white)
    sm_phys_metrics = field_smooth(sm_phys)

    # write the *_physkernel trajectory parquet (PHYSICS).
    rows = []
    for rn in runners:
        idx = np.where(sm_phys[rn]["node_has_obs"])[0]
        if not len(idx):
            continue
        for i in range(idx.min(), idx.max() + 1):
            rows.append(dict(runner=rn, race_t=float(grid[i]), arclen=float(sm_phys[rn]["s"][i]),
                             y_lat_m=float(sm_phys[rn]["l"][i]),
                             sd_arclen=float(sm_phys[rn]["sd_s"][i]),
                             sd_ylat=float(sm_phys[rn]["sd_l"][i])))
    out_traj = f"data_cache/cv_fusion_trajectories_{name}_physkernel.parquet"
    pl.DataFrame(rows).write_parquet(out_traj)

    return dict(name=name, cfg=cfg, grid=grid, runners=runners, off_tab=off, bands=bands,
                sm_white=sm_white, sm_phys=sm_phys, sm_phys_raw=sm_phys_raw,
                ev_white=ev_white, ev_phys=ev_phys,
                white_metrics=sm_white_metrics, phys_metrics=sm_phys_metrics,
                out_traj=out_traj)


# ============================================================================ REPORT
def finish_margin_mae(sm: dict, grid: np.ndarray, runners: list, off: dict) -> float:
    """Fused finish-margin MAE vs official LBW (lengths) — the S3 surface no-regression check."""
    lbw = off.get("lbw", {})
    fin_order = off["finish_order"]
    fin_arc = {rn: C96._arclen_at_t(sm, rn, grid, F1.WINNER_FINISH_RACE_T) for rn in runners}
    if fin_order[0] not in fin_arc:
        return float("nan")
    win_arc = fin_arc[fin_order[0]]
    common = [h for h in runners if h in lbw]
    if not common:
        return float("nan")
    ae = [abs((win_arc - fin_arc[h]) / F1.BODY_LEN_M - lbw[h]) for h in common]
    return float(np.mean(ae))


def report(res: dict) -> None:
    n = res["name"]
    w, p = res["ev_white"], res["ev_phys"]
    wm, pm = res["white_metrics"], res["phys_metrics"]
    print("\n" + "=" * 88)
    print(f"{n}  (W={res['cfg'].width}m, {len(res['runners'])} runners)  — WHITE kernel vs PHYSICS kernel")
    print("=" * 88)
    print("  [SMOOTHNESS — the #97 target] "
          "                       WHITE      PHYSICS    delta")
    labels = ["cruise vel total-variation(m/s)", "cruise jerk RMS (m/s^3)         ",
              "post-ramp 20s swing pk-pk (m/s) ", "gap-curve roughness (TV d2, m)  "]
    for lab, a, b in zip(labels, wm, pm, strict=True):
        d = b - a
        arrow = "smoother" if d < 0 else "rougher "
        print(f"    {lab} {a:9.2f}  {b:9.2f}  {d:+8.2f} {arrow}")
    print("\n  [NO-REGRESSION battery — must NOT degrade]")
    print(f"    winner recovery : WHITE {'YES' if w['winner_ok'] else 'NO'}   "
          f"PHYSICS {'YES' if p['winner_ok'] else 'NO'}")
    print(f"    finish-order rho: WHITE {w['rho_fin']:+.2f}   PHYSICS {p['rho_fin']:+.2f}")
    pr_w = " ".join(f"{r:+.2f}" for r in w["pole_rhos"])
    pr_p = " ".join(f"{r:+.2f}" for r in p["pole_rhos"])
    print(f"    per-pole order rho WHITE  : {pr_w}")
    print(f"    per-pole order rho PHYSICS: {pr_p}")
    print(f"    rail-vs-wide GAP: WHITE {w['gap']:+.2f}m ({w['correct']}/{w['ncov']} class)   "
          f"PHYSICS {p['gap']:+.2f}m ({p['correct']}/{p['ncov']} class)")
    mae_w = finish_margin_mae(res["sm_white"], res["grid"], res["runners"], res["off_tab"])
    mae_p = finish_margin_mae(res["sm_phys"], res["grid"], res["runners"], res["off_tab"])
    print(f"    finish-margin MAE (len): WHITE {mae_w:.2f}   PHYSICS {mae_p:.2f}")
    # per-horse fitted levels (the data-driven shape levels).
    fits = [(rn, float(res["sm_phys_raw"][rn]["_vc_fit"][0]),
             float(res["sm_phys_raw"][rn]["_dfin_fit"][0]))
            for rn in res["runners"]]
    print("    fitted EOM levels (v_c m/s, d_fin): "
          + ", ".join(f"#{rn}:{vc:.1f}/{df:+.2f}" for rn, vc, df in fits[:6]) + " ...")


def make_figure(results: list[dict]) -> str:
    """Before/after: gap-behind-leader vs race_t, WHITE (top) vs PHYSICS (bottom), all 3 races."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for col, res in enumerate(results):
        n = res["name"]
        grid, runners = res["grid"], res["runners"]
        cmap = plt.colormaps["tab20"]
        winner = res["off_tab"]["finish_order"][0]
        for row, (sm, lbl) in enumerate([(res["sm_white"], "WHITE kernel (white-accel, #96 base)"),
                                          (res["sm_phys"], "PHYSICS kernel (EOM-shaped longitudinal)")]):
            ax = axes[row, col]
            lead = np.full(len(grid), -np.inf)
            for rn in runners:
                has = sm[rn]["node_has_obs"]
                idx = np.where(has)[0]
                if len(idx) == 0:
                    continue
                m = slice(idx.min(), idx.max() + 1)
                lead[m] = np.maximum(lead[m], sm[rn]["s"][m])
            for i, rn in enumerate(runners):
                s, has = sm[rn]["s"], sm[rn]["node_has_obs"]
                idx = np.where(has)[0]
                if len(idx) == 0:
                    continue
                m = slice(idx.min(), idx.max() + 1)
                gap = s[m] - lead[m]
                lw = 2.6 if rn == winner else 1.0
                ax.plot(grid[m], gap, color=cmap(i % 20), lw=lw,
                        label=f"#{rn}" + (" WIN" if rn == winner else ""))
            ax.axhline(0, color="0.3", lw=0.7)
            ax.set_xlabel("race_t (s)")
            ax.set_ylabel("gap behind leader (m)  [0=leader]")
            ax.set_ylim(-32, 4)
            wm = res["white_metrics"] if row == 0 else res["phys_metrics"]
            ax.set_title(f"{n} — {lbl}\nvel-TV {wm[0]:.1f} m/s | gap-rough {wm[3]:.1f}")
            if row == 0 and col == 0:
                ax.legend(ncol=4, fontsize=6, loc="lower left")
    fig.suptitle("TASK #97 — physics (EOM-shaped) longitudinal kernel vs white-accel kernel: "
                 "gap-behind-leader vs race_t.  Top=WHITE, Bottom=PHYSICS.  R01 | R03 | R08",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "data_cache/figs/cv_fusion_physkernel_qc.png"
    fig.savefig(out, dpi=110)
    print(f"\n[fig] {out}")
    return out


def main() -> list[dict]:
    print("TASK #97 — PHYSICS (EOM-shaped) LONGITUDINAL kernel for the 4-feed fusion smoother")
    print(f"KERNEL: ramp tau {TAU_RAMP}s; phase-dependent OU pull -> RAMP window {RAMP_WINDOW_S:.1f}s "
          f"(strong: tau_rev {TAU_REVERT_RAMP}s, sigma_a {SIGMA_A_RAMP}) then CRUISE/FINISH "
          f"(weak: tau_rev {TAU_REVERT_CRUISE}s, sigma_a {SIGMA_A_CRUISE} vs white {F1.SIGMA_A_LONG}). "
          f"Levels (v_c in [{VC_LO},{VC_HI}], d_fin in [{DFIN_LO},{DFIN_HI}]) fit per-horse from feeds. "
          f"Lateral = #96 (box+order+geom).")
    results = []
    for name in ("R01", "R03", "R08"):
        results.append(run_race(name))
        report(results[-1])
    fig = make_figure(results)
    print("\nartifacts:")
    for r in results:
        print(f"  {r['name']}: {r['out_traj']}")
    print(f"  figure: {fig}")
    return results


if __name__ == "__main__":
    main()

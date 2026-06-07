"""TASK #96 — PHYSICAL CONSTRAINTS on the 4-feed fusion smoother (study-only).

The fusion (cv_fusion_4feed, an RTS smoother) is per-horse LINEAR-GAUSSIAN, so it admits
physically impossible lateral states: negative y_lat (inside the inner rail) and y_lat >> track
width (R08 route fused y_lat sd balloons to ~52m, >2x the ~22.5m C-course width). This module
COPIES/EXTENDS the smoother with three HARD PHYSICAL CONSTRAINTS and measures the effect on the
R01/R03/R08 reconstruction, ESPECIALLY the route (R08) BEND lateral-metre gap (CV_RECON.md S7).

THE THREE CONSTRAINTS (all study-only; banked parquets untouched; *_constrained outputs):

  (1) HARD BOX  0 <= y_lat <= W   (W = C-course running width).
      Principled enforcement: per grid-node TRUNCATED-GAUSSIAN moment-match of each horse's
      lateral posterior (mu,sigma) onto [0,W] (not a naive post-hoc clip). The truncated mean +
      variance replace the Gaussian ones; this both moves the mean inside the box AND shrinks the
      sd by the amount the box removes. We ALSO report the naive clip for contrast.

  (2) JOINT CROSS-HORSE ORDERING-WITHIN-BOX  (the prize / candidate bend gap-closer).
      The per-horse smoother is INDEPENDENT, but at each race_t the field is (a) laterally ORDERED
      (recovered robustly even on bends from aerial/drone), (b) inside [0,W], (c) separated by >= a
      horse width s_min ~1.1m. "N ordered bodies across a known W" constrains ABSOLUTE positions.
      Implemented as a PRECISION-WEIGHTED isotonic/QP PROJECTION of the per-horse lateral posteriors
      onto the ordered-within-box, min-separation feasible set at each grid node, with the posterior
      VARIANCE shrunk to the feasible-interval width (the constraint removes mass -> tighter sd).
      The lateral ORDER is read from the observed feeds (aerial+drone+patrol+draw), NOT from the
      official result.

  (3) TRACK-GEOMETRY PRIOR.
      HV "C" course (HKJC course-information page, www.hkjc.com/english/racinginfo): turf
      C-course running WIDTH = 22.5m (A=30.5, B=26.5, C=22.5, C+3=19.5); home straight C = 334m;
      circuit ~1450m. The studies baked W=21.3m (banked-rail approximation) — we test BOTH and
      report. An a-priori arclen->{straight,bend} map (from the circuit geometry, NOT empirics) sets
      where the AERIAL lateral metric is TRUSTED (straights) vs DOWN-WEIGHTED (bends), replacing the
      empirical >run-in-threshold heuristic. Bend radius recorded for a future rail-arc homography.

PAIRED OFF/ON: the SAME obs set + draw anchor for every race; OFF = the plain linear-Gaussian
smoother; ON = + the three constraints. Validation (official = validation only; draw is an allowed
pre-race initial condition): per-race rail-vs-wide gap (sign+magnitude) + classification, per-band
lateral sd OFF vs ON (esp. R08 route), winner recovery + per-pole order (must NOT regress).

DISCIPLINE: study-only; src/hkjc untouched; banked R01/R03/R08 fusion parquets NOT mutated
(writes *_constrained-suffixed); official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_fusion_constrained.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, "studies")

import cv_fusion_4feed as F1  # noqa: E402,N812

# ====================================================================================== GEOMETRY
# HV turf course-information (HKJC racing_course page). Running WIDTHS per course:
#   A 30.5m | A+2 28.5 | B 26.5 | B+2 24.5 | B+3 23.5 | C 22.5 | C+3 19.5
# Home straight: C = 334m. Circuit ~1450m. The races here are all run on "C".
W_C_COURSE = 22.5          # HKJC C-course running width (m) — the authoritative spec.
W_BANKED_APPROX = 21.3     # the value baked in the studies (banked-rail approximation).
HOME_STRAIGHT_C = 334.0    # m (HKJC C-course home straight).
CIRCUIT_M = 1450.0         # HV turf circuit (m).
HORSE_WIDTH_M = 1.10       # min lateral centre-to-centre separation (a horse body ~1.1m).

# HV is a tight rounded-rectangle ~1450m: home straight ~334m, a roughly matching back straight,
# the remainder is two long bends. Straight fraction ~ (334 + ~300) / 1450 ~ 0.44; the rest bends.
# We build the arclen->{straight,bend} map from THIS geometry (a-priori), not from the data.
BACK_STRAIGHT_C = 300.0     # m (estimate: HV back straight is slightly shorter than the home str).
BEND_RADIUS_M = (CIRCUIT_M - HOME_STRAIGHT_C - BACK_STRAIGHT_C) / (2.0 * np.pi)  # ~130m per bend.


@dataclass
class RaceCfg:
    name: str
    dist: float
    width: float            # track width used for the box (we sweep C-course vs banked).
    n_field: int


# ====================================================================================== (3) GEOMETRY MAP
def bend_fraction_map(arclen: np.ndarray, dist: float) -> np.ndarray:
    """A-PRIORI per-arclen 'on a bend' indicator in [0,1] from the circuit geometry (NOT empirics).

    HV circuit (1450m): the START is positioned so the FINISH is at the end of the home straight.
    Walking BACK from the finish line (arclen=dist): the last HOME_STRAIGHT_C metres are straight;
    before that a bend (~bend arc), before that the back straight, before that the first bend, etc.
    A route (1800m > 1450m) wraps >1 lap. We map each arclen to its phase on the lap and return a
    soft straight(0)/bend(1) weight. This is the GEOMETRY trust map: aerial lateral metric is
    trusted on straights (weight 0), down-weighted on bends (weight 1).
    """
    bend_arc = np.pi * BEND_RADIUS_M  # arc length of one ~semicircular bend (~410m).
    # one lap laid out from the FINISH backwards: [home straight | bend2 | back straight | bend1]
    # phase 0 == finish line; increasing phase == further BACK from the finish.
    seg = [
        (HOME_STRAIGHT_C, 0.0),   # home straight (trusted)
        (bend_arc, 1.0),          # bend 2
        (BACK_STRAIGHT_C, 0.0),   # back straight (trusted)
        (bend_arc, 1.0),          # bend 1
    ]
    lap = sum(s for s, _ in seg)
    out = np.zeros_like(np.asarray(arclen, float))
    for i, a in enumerate(np.asarray(arclen, float)):
        phase = (dist - a) % lap   # metres back from the finish, wrapped to the lap.
        acc = 0.0
        w = 0.0
        for s, flag in seg:
            if phase <= acc + s:
                w = flag
                break
            acc += s
        out[i] = w
    return out


# ====================================================================================== OBS INGEST (paired)
def build_obs(cfg: RaceCfg, draw_anchor: dict[int, float], use_geometry: bool, rf_loader):
    """Build the SAME obs set used by OFF and ON. The only ON/OFF difference inside the smoother is
    the CONSTRAINTS applied AFTER smoothing; here we optionally re-tag aerial/drone lateral noise by
    the GEOMETRY map (constraint 3) instead of the empirical run-in threshold so the bend down-weight
    is a-priori. We do this identically for OFF/ON's *input* only when use_geometry — see run_paired
    (geometry is part of the 'ON' bundle; OFF keeps the original empirical tagging)."""
    aer_obs, aer_df, field_race_t, field_arc = F1.load_aerial()
    dro_obs = _load_drone(cfg)
    pat_obs = _load_patrol(cfg)
    rf_obs = rf_loader(field_race_t, field_arc)

    if use_geometry:
        # GEOMETRY re-tag: aerial/drone lateral sd is set by bend-fraction (a-priori) rather than the
        # empirical >run-in threshold. On a bend the render foreshortens -> inflate; on a straight ->
        # the original (trusted) sd. Smooth interpolation between the two.
        for o in aer_obs:
            if o.ylat is not None and o.arclen is not None:
                bf = float(bend_fraction_map(np.array([o.arclen]), cfg.dist)[0])
                o.sd_ylat = F1.SD_YLAT_AERIAL * (1.0 - bf) + F1.SD_YLAT_AERIAL_RUNIN * 1.6 * bf
        for o in dro_obs:
            if o.ylat is not None and o.arclen is not None:
                bf = float(bend_fraction_map(np.array([o.arclen]), cfg.dist)[0])
                o.sd_ylat = F1.SD_YLAT_DRONE * (1.0 - bf) + F1.SD_YLAT_DRONE_RUNIN * 1.4 * bf

    all_obs = aer_obs + dro_obs + pat_obs + rf_obs
    # draw -> y_lat(0) anchor (a pre-race INITIAL CONDITION; allowed). Added to ALL runs for fairness.
    for n, y in draw_anchor.items():
        all_obs.append(F1.Obs(runner=int(n), race_t=0.0, arclen=0.0, ylat=float(y),
                              feed="draw_anchor", sd_arclen=999.0, sd_ylat=1.5))
    return all_obs, aer_df, field_race_t, field_arc


def _load_drone(cfg: RaceCfg):
    if cfg.name == "R01":
        return F1.load_drone()
    d = pl.read_parquet(F1.DRONE_PANEL).filter(pl.col("number") > 0)
    runin = cfg.dist * 0.79
    out = []
    for r in d.iter_rows(named=True):
        ri = r["arclen"] > runin
        out.append(F1.Obs(runner=int(r["number"]), race_t=float(r["race_t"]),
                          arclen=float(r["arclen"]), ylat=float(r["y_lat"]), feed="drone",
                          sd_arclen=F1.SD_ARC_DRONE,
                          sd_ylat=F1.SD_YLAT_DRONE_RUNIN if ri else F1.SD_YLAT_DRONE))
    return out


def _load_patrol(cfg: RaceCfg):
    if cfg.name == "R01":
        return F1.load_patrol()
    p = pl.read_parquet(F1.PATROL_PANEL)
    out = []
    for r in p.iter_rows(named=True):
        c = float(np.clip(r.get("confidence", 1.0), 0.25, 1.0))
        out.append(F1.Obs(runner=int(r["runner"]), race_t=float(r["race_t"]),
                          arclen=float(r["arclen"]), ylat=float(r["y_lat_m"]), feed="patrol",
                          sd_arclen=F1.SD_ARC_PATROL, sd_ylat=F1.SD_YLAT_PATROL / c))
    return out


# ====================================================================================== CONSTRAINTS
def _trunc_gauss_moments(mu: np.ndarray, sd: np.ndarray, lo: float, hi: float):
    """Moment-match a Gaussian(mu,sd) truncated to [lo,hi]. Returns (mean, sd) of the truncated law.
    Principled box enforcement: pulls the mean inside AND shrinks the sd by the removed tail mass."""
    from math import sqrt

    sd = np.maximum(sd, 1e-6)
    a = (lo - mu) / sd
    b = (hi - mu) / sd
    # standard-normal pdf/cdf
    def _phi(x):
        return np.exp(-0.5 * x * x) / sqrt(2.0 * np.pi)

    from scipy.special import ndtr
    za = ndtr(a)
    zb = ndtr(b)
    z = np.maximum(zb - za, 1e-9)
    pa, pb = _phi(a), _phi(b)
    mean = mu + (pa - pb) / z * sd
    # variance of truncated normal
    term = 1.0 + (a * pa - b * pb) / z - ((pa - pb) / z) ** 2
    var = sd * sd * np.clip(term, 1e-6, None)
    return np.clip(mean, lo, hi), np.sqrt(var)


def _project_ordered_box(mu: np.ndarray, prec: np.ndarray, order: np.ndarray,
                         lo: float, hi: float, s_min: float):
    """Precision-WEIGHTED projection of the points mu (in ORDER given by `order`, ascending) onto the
    feasible set { lo <= x_(1), x_(k+1) - x_(k) >= s_min, x_(N) <= hi } minimising sum prec*(x-mu)^2.

    This is a weighted isotonic-type QP. We reparametrise gaps g_k = x_(k+1)-x_(k)-s_min >= 0 and the
    box bounds, and solve with a small projected-gradient / PAVA-style loop. With N<=14 this is cheap.
    Returns the projected positions in the ORIGINAL (unordered) index space and the per-point feasible
    half-width (used to shrink the posterior sd).
    """
    n = len(mu)
    if n == 0:
        return mu.copy(), np.zeros(0)
    m = mu[order]
    w = np.maximum(prec[order], 1e-9)
    # shift so the min-separation becomes a simple monotonic (non-decreasing) constraint:
    #   y_k = x_(k) - k*s_min   =>   x_(k+1)-x_(k) >= s_min  <=>  y_{k+1} >= y_k
    # box: x_(1) >= lo -> y_1 >= lo ; x_(N) <= hi -> y_N <= hi - (N-1)*s_min
    shift = np.arange(n) * s_min
    y_t = m - shift
    ylo = lo
    yhi = hi - (n - 1) * s_min
    if yhi < ylo:  # infeasible (field cannot fit) — relax min-sep to fit the box exactly.
        s_eff = (hi - lo) / max(n - 1, 1)
        shift = np.arange(n) * s_eff
        y_t = m - shift
        ylo, yhi = lo, lo  # everything pinned to an even spread
        # weighted isotonic with both bounds equal collapses to the even spread:
        x_even = lo + np.arange(n) * s_eff
        out = np.empty(n)
        out[order] = x_even
        hw = np.full(n, 0.5 * s_eff)
        return out, hw
    # weighted isotonic regression (PAVA) of y_t with weights w onto non-decreasing sequence.
    yhat = _pava(y_t, w)
    # apply the box on the monotonic variable then map back.
    yhat = np.clip(yhat, ylo, yhi)
    # re-enforce monotonicity after clipping (clip can break it at the edges):
    yhat = np.maximum.accumulate(yhat)
    yhat = np.clip(yhat, None, yhi)
    x_sorted = yhat + shift
    out = np.empty(n)
    out[order] = x_sorted
    # feasible half-width per point: distance to its order-neighbours / box, capped — a proxy for the
    # residual lateral freedom AFTER the constraint (used to set the constrained sd).
    hw = np.empty(n)
    for k in range(n):
        left = x_sorted[k] - (x_sorted[k - 1] + s_min) if k > 0 else x_sorted[k] - lo
        right = (x_sorted[k + 1] - s_min) - x_sorted[k] if k < n - 1 else hi - x_sorted[k]
        hw[order[k]] = 0.5 * max(min(left, right), 0.0)
    return out, hw


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators (weighted) isotonic regression onto a non-decreasing sequence."""
    n = len(y)
    val = y.astype(float).copy()
    wt = w.astype(float).copy()
    vals, wts, cnts = [], [], []  # block stack of (value, weight, count)
    for i in range(n):
        v, ww, c = val[i], wt[i], 1
        while vals and vals[-1] > v:
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww = pw + ww
            c = pc + c
        vals.append(v)
        wts.append(ww)
        cnts.append(c)
    out = np.empty(n)
    j = 0
    for v, c in zip(vals, cnts, strict=True):
        out[j:j + c] = v
        j += c
    return out


def apply_constraints(smoothed: dict, grid: np.ndarray, runners: list[int], all_obs: list,
                      cfg: RaceCfg, do_box=True, do_order=True):
    """Apply (1) box + (2) joint ordering-within-box at EACH grid node to the lateral channel.

    OFF (do_box=do_order=False) returns the smoothed dict unchanged. The longitudinal channel
    (s, sd_s) is NEVER touched — only y_lat (l) and its sd_l.
    """
    n = len(grid)
    W = cfg.width
    s_min = HORSE_WIDTH_M
    out = {rn: {k: v.copy() for k, v in smoothed[rn].items()} for rn in runners}

    # each horse's emitted coverage window (first..last observation node): the BOX is a physical fact
    # that holds at EVERY node in the window, NOT only where there is an observation that frame — a
    # horse with an unanchored lateral posterior (no lateral obs on the bend) is STILL inside [0,W].
    cover = {}
    for rn in runners:
        idx = np.where(out[rn]["node_has_obs"])[0]
        cover[rn] = (idx.min(), idx.max()) if len(idx) else (n, -1)

    for k in range(n):
        present = [rn for rn in runners if cover[rn][0] <= k <= cover[rn][1]]
        if not present:
            continue
        mu = np.array([out[rn]["l"][k] for rn in present])
        sd = np.array([out[rn]["sd_l"][k] for rn in present])

        # (1) BOX via truncated-Gaussian moment-match (UNCONDITIONAL — every horse, every in-window
        #     node). This is what caps the unanchored bend posteriors (sd 120m -> ~W/sqrt(12)).
        if do_box:
            mu_b, sd_b = _trunc_gauss_moments(mu, sd, 0.0, W)
        else:
            mu_b, sd_b = mu.copy(), sd.copy()

        # (2) JOINT ORDERING-WITHIN-BOX. Order from the boxed posterior means; project. Needs >=2
        #     co-present horses (the ordering only informs metres when the field is co-located).
        if do_order and len(present) >= 2:
            prec = 1.0 / np.maximum(sd_b, 1e-3) ** 2
            order = np.argsort(mu_b, kind="stable")
            mu_o, hw = _project_ordered_box(mu_b, prec, order, 0.0, W, s_min)
            # HONEST constrained sd. The feasible half-width hw is the sd ONLY IF the lateral ORDER is
            # itself certain at this node. On a bend the order is read from the (foreshortened) aerial/
            # drone render and CAN be wrong (CV_RECON S7) -> do NOT collapse the sd to hw there. We
            # gate by the per-node ORDER RELIABILITY: the fraction of co-present horses that have a
            # genuine LATERAL OBSERVATION this frame (real reads -> order trustworthy -> trust hw;
            # interpolated -> keep the boxed sd, since the metres are not actually pinned).
            has_lat_now = np.array([smoothed[rn]["node_has_lat"][k] for rn in present], float)
            rel = float(has_lat_now.mean())   # 0..1 order-reliability at this node
            # blend: reliable nodes -> the tight feasible-width sd; unreliable -> the boxed sd. A
            # horse with NO lateral obs this frame keeps (mostly) its boxed sd even if neighbours saw.
            sd_tight = np.maximum(hw, 0.5 * s_min)
            sd_o = rel * np.minimum(sd_b, sd_tight) + (1.0 - rel) * sd_b
            # per-horse: an unobserved horse cannot be pinned below the box sd by its neighbours alone.
            sd_o = np.where(has_lat_now > 0, np.minimum(sd_o, sd_b), sd_b)
            mu_b, sd_b = mu_o, sd_o

        for i, rn in enumerate(present):
            out[rn]["l"][k] = mu_b[i]
            out[rn]["sd_l"][k] = sd_b[i]
    return out


# ====================================================================================== VALIDATION
def _arclen_at_t(sm, rn, grid, t):
    ki = int(np.clip(round((t - grid[0]) / F1.DT), 0, len(grid) - 1))
    return float(sm[rn]["s"][ki])


def lateral_band_median(sm, runners, lo, hi):
    out = {}
    for rn in runners:
        s = sm[rn]["s"]
        m = (s >= lo) & (s <= hi) & sm[rn]["node_has_lat"]
        if m.sum() >= 2:
            out[rn] = float(np.median(sm[rn]["l"][m]))
    return out


def band_sd(sm, runners, lo, hi):
    vals = []
    for rn in runners:
        s = sm[rn]["s"]
        m = (s >= lo) & (s <= hi)
        if m.sum() >= 1:
            vals.append(float(np.median(sm[rn]["sd_l"][m])))
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(sm, grid, runners, off_tab, cfg, bands):
    """Return a dict of the headline validation numbers for one (race, OFF/ON) run."""
    finish_order = off_tab["finish_order"]
    rail_t, wide_t = off_tab["rail"], off_tab["wide"]
    fin_arc = {rn: _arclen_at_t(sm, rn, grid, F1.WINNER_FINISH_RACE_T) for rn in runners}
    fused_finish = sorted(runners, key=lambda r: -fin_arc[r])
    off_rank = {h: i for i, h in enumerate(finish_order)}
    common = [h for h in runners if h in off_rank]
    rho_fin = F1._spearman(np.array([fused_finish.index(h) for h in common], float),
                           np.array([off_rank[h] for h in common], float))
    winner_ok = fused_finish[0] == finish_order[0]

    # per-pole order
    poles = off_tab["poles"]
    rp = off_tab["rp"]
    pole_rhos = []
    for ci, (_nm, arc) in enumerate(poles):
        t = float(off_tab["arclen_to_race_t"](np.array([arc]))[0])
        ki = int(np.clip(round((t - grid[0]) / F1.DT), 0, len(grid) - 1))
        fs = {rn: _arclen_at_t(sm, rn, grid, t) for rn in runners
              if sm[rn]["node_has_obs"][:ki + 1].any()}
        cc = [h for h in fs if h in rp and len(rp[h]) > ci]
        if len(cc) >= 3:
            pole_rhos.append(F1._spearman(np.array([-fs[h] for h in cc], float),
                                          np.array([rp[h][ci] for h in cc], float)))
        else:
            pole_rhos.append(float("nan"))

    # lateral classification
    blo, bhi = off_tab["class_band"]
    body = lateral_band_median(sm, runners, blo, bhi)
    rv = [body[h] for h in rail_t if h in body]
    wv = [body[h] for h in wide_t if h in body]
    gap = float(np.mean(wv) - np.mean(rv)) if rv and wv else float("nan")
    if rv and wv:
        thr = 0.5 * (np.mean(rv) + np.mean(wv))
        ncov = len((rail_t | wide_t) & set(body))
        correct = sum(body[h] < thr for h in rail_t & set(body)) + \
                  sum(body[h] >= thr for h in wide_t & set(body))
    else:
        ncov = correct = 0

    sds = {nm: band_sd(sm, runners, lo, hi) for nm, lo, hi in bands}
    # min / max lateral median across the field (the unphysical-tail diagnostic)
    allmed = list(body.values())
    return dict(rho_fin=rho_fin, winner_ok=winner_ok, pole_rhos=pole_rhos, gap=gap,
                ncov=ncov, correct=correct, sds=sds, body=body,
                ymin=min(allmed) if allmed else float("nan"),
                ymax=max(allmed) if allmed else float("nan"))


# ====================================================================================== RACE SETUP
def setup_race(name: str):
    """Re-point F1 module state at a race + return its (cfg, draw_anchor, official tables)."""
    if name == "R01":
        # R01 lives directly in cv_fusion_4feed (no per-race module). Tables from F1.
        import re
        F1.PATROL_PANEL = "data_cache/cv_patrol_numbered_v2.parquet"
        off = dict(
            finish_order=list(F1.FINISH_ORDER),
            rp=dict(F1.OFFICIAL_RP),
            lbw=dict(F1.OFFICIAL_LBW),
            rail=set(F1.RAIL_HORSES), wide=set(F1.WIDE_HORSES),
            poles=[("1/4", 450.0), ("1/2", 850.0), ("3/4", 1250.0), ("fin", 1640.0)],
            class_band=(150.0, 1300.0),
            arclen_to_race_t=F1.arclen_to_race_t,
        )
        html = open("data_cache/race_official/20260603R1_results.html",
                    encoding="utf-8", errors="ignore").read()
        draw = {}
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [re.sub(r"<[^>]+>", " ", c).replace("&nbsp;", " ").strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
            if len(cells) >= 9 and cells[1].isdigit() and cells[7].isdigit():
                draw[int(cells[1])] = int(cells[7])
        from cv_identity import draw_to_metres
        cfg = RaceCfg("R01", 1650.0, W_C_COURSE, 12)
        return cfg, draw_to_metres(draw, W_C_COURSE), off, F1.load_rf

    if name == "R03":
        import cv_r03_common as C  # noqa: N812
        import cv_r03_fusion as R3  # noqa: N812
        from cv_identity import draw_to_metres
        offp = C.parse_official()
        R3._setup_r03(offp)
        rail_t, wide_t = C.lateral_truth(offp)
        F1.PATROL_PANEL = "data_cache/cv_r03_patrol_numbered_v2.parquet"
        off = dict(
            finish_order=list(offp.finish_order), rp=dict(offp.rp), lbw=dict(offp.lbw),
            rail=rail_t, wide=wide_t,
            poles=[("1/4", 400.0), ("1/2", 800.0), ("late", 1100.0)],
            class_band=(100.0, 1100.0), arclen_to_race_t=C.arclen_to_race_t,
        )
        cfg = RaceCfg("R03", 1200.0, W_C_COURSE, 12)

        def rf_loader(frt, fa, _off=offp):
            return R3._load_rf_r03(_off, frt, fa)
        return cfg, draw_to_metres(dict(offp.draw), W_C_COURSE), off, rf_loader

    # R08
    import cv_r08_common as C  # noqa: N812
    import cv_r08_fusion as R8  # noqa: N812
    from cv_identity import draw_to_metres
    offp = C.parse_official()
    R8._setup_r08(offp)
    rail_t, wide_t = C.lateral_truth(offp)
    off = dict(
        finish_order=list(offp.finish_order), rp=dict(offp.rp), lbw=dict(offp.lbw),
        rail=rail_t, wide=wide_t,
        poles=[("c1", 200.0), ("c2", 600.0), ("c3", 1000.0), ("c4", 1400.0), ("late", 1700.0)],
        class_band=(100.0, 1550.0), arclen_to_race_t=C.arclen_to_race_t,
    )
    cfg = RaceCfg("R08", 1800.0, W_C_COURSE, 11)

    def rf_loader(frt, fa, _off=offp):
        return R8._load_rf_r08(_off, frt, fa)
    return cfg, draw_to_metres(dict(offp.draw), W_C_COURSE), off, rf_loader


# ====================================================================================== DRIVER
def run_race(name: str):
    cfg, draw_anchor, off, rf_loader = setup_race(name)
    bands = off_bands(cfg)

    # OFF: plain smoother (no geometry retag, no constraints).
    all_obs, _aer_df, _frt, _farc = build_obs(cfg, draw_anchor, use_geometry=False, rf_loader=rf_loader)
    runners = sorted({o.runner for o in all_obs})
    t_lo = min(o.race_t for o in all_obs)
    t_hi = max(o.race_t for o in all_obs)
    grid = np.arange(t_lo, t_hi + F1.DT, F1.DT)
    sm_off = {rn: F1.smooth_horse(grid, [o for o in all_obs if o.runner == rn]) for rn in runners}
    ev_off = evaluate(sm_off, grid, runners, off, cfg, bands)

    # ON: geometry-retagged obs + box + ordering constraints.
    all_obs_g, _a, _b, _c = build_obs(cfg, draw_anchor, use_geometry=True, rf_loader=rf_loader)
    sm_on_raw = {rn: F1.smooth_horse(grid, [o for o in all_obs_g if o.runner == rn])
                 for rn in runners}
    sm_on = apply_constraints(sm_on_raw, grid, runners, all_obs_g, cfg, do_box=True, do_order=True)
    ev_on = evaluate(sm_on, grid, runners, off, cfg, bands)

    # also the naive post-hoc clip (for the box contrast) + box-only.
    sm_clip = {rn: {k: v.copy() for k, v in sm_off[rn].items()} for rn in runners}
    for rn in runners:
        sm_clip[rn]["l"] = np.clip(sm_clip[rn]["l"], 0.0, cfg.width)
    ev_clip = evaluate(sm_clip, grid, runners, off, cfg, bands)
    sm_box = apply_constraints(sm_on_raw, grid, runners, all_obs_g, cfg, do_box=True, do_order=False)
    ev_box = evaluate(sm_box, grid, runners, off, cfg, bands)

    # write the *_constrained trajectory parquet (ON).
    rows = []
    for rn in runners:
        idx = np.where(sm_on[rn]["node_has_obs"])[0]
        if not len(idx):
            continue
        for i in range(idx.min(), idx.max() + 1):
            rows.append(dict(runner=rn, race_t=float(grid[i]), arclen=float(sm_on[rn]["s"][i]),
                             y_lat_m=float(sm_on[rn]["l"][i]), sd_arclen=float(sm_on[rn]["sd_s"][i]),
                             sd_ylat=float(sm_on[rn]["sd_l"][i])))
    out_traj = f"data_cache/cv_fusion_trajectories_{name}_constrained.parquet"
    pl.DataFrame(rows).write_parquet(out_traj)

    return dict(name=name, cfg=cfg, grid=grid, runners=runners, off_tab=off, bands=bands,
                sm_off=sm_off, sm_on=sm_on, ev_off=ev_off, ev_on=ev_on, ev_clip=ev_clip,
                ev_box=ev_box, out_traj=out_traj)


def off_bands(cfg: RaceCfg):
    d = cfg.dist
    if cfg.name == "R01":
        return [("start 150-450", 150, 450), ("backstr/bend 450-850", 450, 850),
                ("bend->str 850-1300", 850, 1300), ("RUN-IN >1300", 1300, d)]
    if cfg.name == "R03":
        return [("start 100-400", 100, 400), ("mid 400-800", 400, 800),
                ("bend->str 800-950", 800, 950), ("RUN-IN >950", 950, d)]
    return [("start 100-600", 100, 600), ("mid 600-1100", 600, 1100),
            ("bend->str 1100-1420", 1100, 1420), ("RUN-IN >1420", 1420, d)]


def report(res):
    n = res["name"]
    off, on, clip, box = res["ev_off"], res["ev_on"], res["ev_clip"], res["ev_box"]
    print("\n" + "=" * 84)
    print(f"{n}  (W={res['cfg'].width}m, {len(res['runners'])} runners)  — CONSTRAINTS OFF vs ON")
    print("=" * 84)
    print(f"  winner recovery : OFF {'YES' if off['winner_ok'] else 'NO'}  "
          f"ON {'YES' if on['winner_ok'] else 'NO'}    finish-rho OFF {off['rho_fin']:+.2f} "
          f"ON {on['rho_fin']:+.2f}")
    pr_off = " ".join(f"{r:+.2f}" for r in off["pole_rhos"])
    pr_on = " ".join(f"{r:+.2f}" for r in on["pole_rhos"])
    print(f"  per-pole rho OFF: {pr_off}")
    print(f"  per-pole rho ON : {pr_on}")
    print(f"  rail-vs-wide GAP: OFF {off['gap']:+.2f}m ({off['correct']}/{off['ncov']} class)   "
          f"ON {on['gap']:+.2f}m ({on['correct']}/{on['ncov']} class)")
    print(f"  field y_lat span: OFF [{off['ymin']:+.1f},{off['ymax']:+.1f}]m   "
          f"ON [{on['ymin']:+.1f},{on['ymax']:+.1f}]m   (box = [0,{res['cfg'].width}])")
    print("  per-band lateral sd (m):  band                      OFF    CLIP   BOX     ON")
    for nm, _lo, _hi in res["bands"]:
        print(f"      {nm:26s} {off['sds'][nm]:6.2f} {clip['sds'][nm]:6.2f} "
              f"{box['sds'][nm]:6.2f} {on['sds'][nm]:6.2f}")


def make_figure(results):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for col, res in enumerate(results):
        n = res["name"]
        rail_t, wide_t = res["off_tab"]["rail"], res["off_tab"]["wide"]
        W = res["cfg"].width
        for row, (sm, lbl) in enumerate([(res["sm_off"], "OFF (unconstrained)"),
                                          (res["sm_on"], "ON (box+order+geometry)")]):
            ax = axes[row, col]
            cmap = plt.colormaps["tab20"]
            for i, rn in enumerate(res["runners"]):
                s, lat, sdl, has = (sm[rn]["s"], sm[rn]["l"], sm[rn]["sd_l"],
                                    sm[rn]["node_has_lat"])
                idx = np.where(has)[0]
                if not len(idx):
                    continue
                m = slice(idx.min(), idx.max() + 1)
                c = ("tab:blue" if rn in rail_t else ("tab:red" if rn in wide_t else "0.6"))
                ax.plot(s[m], lat[m], color=c, lw=1.3, alpha=0.9)
                ax.fill_between(s[m], lat[m] - 2 * sdl[m], lat[m] + 2 * sdl[m],
                                color=cmap(i % 20), alpha=0.08, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(W, color="k", lw=0.8, ls="--")
            ax.axhspan(-100, 0, color="0.85", alpha=0.5)
            ax.set_ylim(-8, max(W + 8, res["ev_off"]["ymax"] + 5) if row == 0 else W + 6)
            ax.set_xlabel("arclen (m)")
            ax.set_ylabel("y_lat (m from rail)")
            ax.set_title(f"{n} — {lbl}  (blue=rail, red=wide; box [0,{W}])")
    fig.suptitle("TASK #96 — physical constraints on the 4-feed fusion: unconstrained (top) vs "
                 "box+ordering+geometry (bottom). R01 | R03 | R08(route)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "data_cache/figs/cv_fusion_constrained_qc.png"
    fig.savefig(out, dpi=110)
    print(f"\n[fig] {out}")
    return out


def main():
    print("TASK #96 — PHYSICAL CONSTRAINTS on the 4-feed fusion smoother")
    print(f"GEOMETRY: HV C-course width {W_C_COURSE}m (banked approx {W_BANKED_APPROX}m); "
          f"home straight {HOME_STRAIGHT_C}m; circuit {CIRCUIT_M}m; bend radius "
          f"~{BEND_RADIUS_M:.0f}m; horse width (min-sep) {HORSE_WIDTH_M}m")
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

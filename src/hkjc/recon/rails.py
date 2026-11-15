"""Rails, track frame and distance metric on the static world map: racing rail -> (s, lateral) coordinates.

Rails are extracted ONCE from the mosaic (horses averaged out), as 1-D ridge peaks on NORMAL
PROFILES taken at stations along the smoothed camera path, linked across stations by dynamic
programming. The racing (movable) rail is the line the field runs just OUTSIDE of: the innermost
horses bound it from outside, so the rail = the strongest thin-bright ridge a short way inside them.
The INNER side is the right of travel (HK racing is clockwise; see ``inner_side``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def resample_path(C: np.ndarray, step: float, smooth: float = 60.0):
    """Smooth an ordered 2-D path and resample at uniform arclength ``step``.
    Returns P (N,2), T (N,2) unit tangents, s (N,) arclength."""
    from scipy.ndimage import gaussian_filter1d

    d = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    C, d = C[keep], d[keep]
    s = np.arange(0, d[-1], step)
    P = np.c_[np.interp(s, d, C[:, 0]), np.interp(s, d, C[:, 1])]
    sig = smooth / step
    P = np.c_[
        gaussian_filter1d(P[:, 0], sig, mode="nearest"),
        gaussian_filter1d(P[:, 1], sig, mode="nearest"),
    ]
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return P, T, s


def inner_sign(T: np.ndarray) -> int:
    """+1 if the track turns to the RIGHT of travel on average (clockwise in image coords), else -1.
    The right normal of tangent (tx,ty) in y-down image coords is (-ty, tx)."""
    cross = T[:-1, 0] * T[1:, 1] - T[:-1, 1] * T[1:, 0]  # >0 = turning right (y down)
    return 1 if cross.sum() > 0 else -1


def inner_side(T: np.ndarray, handedness: str | None = None) -> int:
    """The racing rail's side of travel (+1 right / clockwise, -1 left / anti-clockwise).
    1. The camera path TURNS decisively (|summed turning| >= a quarter turn): its measured sign --
       physical, so clockwise and anti-clockwise races are both handled without a table. MEASURED (26
       turf races): ovals |sum| 2.7-8.8 rad.
    2. Otherwise (a STRAIGHT: the three Sha Tin 1000 m races |sum| < 0.2) the sign is noise and comes
       from the race's ``handedness`` ("left" -> -1, else +1: on the HKJC straight stall 1 is on the
       right of travel).
    REJECTED: the bare turning sign on straights -- ST 0419 R3 and ST 0509 R1 flipped between two
    runs of the same video (pose differing only by the avatar mask), ST8 was -1 in both. With -1 the
    "rail" was the other band edge -- on ST 0509 R1 it traced the mosaic's warped coverage boundary
    and dipped 15-25 m before the line, sliding the whole field ~3.5 m laterally, so association
    scrambled (#4 ended on a back-marker, 3.4 m short) -- and the draw prior put stall 1 on the wrong
    side (draw-vs-lateral rho +0.73 with +1, -0.21 with -1; on the +1 roots ST 05-09 R1 +0.5..+0.9
    on 12/12 runs, ST8 +0.2..+0.7 on 11/12, ST3 ~0 where identity is weakest). 6-seed ensembles on
    the 3 straights x 2 pose variants: all gates 285 -> 286 (ST 0509 R1 45 -> 48, rho 1.00 on every
    seed; ST3 rho 0.91 -> 0.98; ST8 rho 0.92-0.95 -> 0.98-1.00 and behind-winner 0.055 -> 0.025 s,
    but its close #2/#12 finish flipped on 1 of 6 seeds). Turf racing is clockwise (working
    assumption): on every oval the measured sign is +1 anyway; it is kept as a physical check rather
    than a hard-coded constant. HKJC_INNER=auto: the bare turning sign everywhere (diagnostic)."""
    cross = T[:-1, 0] * T[1:, 1] - T[:-1, 1] * T[1:, 0]
    turn = float(np.sum(np.arcsin(np.clip(cross, -1.0, 1.0))))
    if os.environ.get("HKJC_INNER", "") == "auto" or abs(turn) >= np.pi / 2:
        return inner_sign(T)
    return -1 if (handedness or "").lower().startswith(("left", "anti")) else 1


def right_normal(T):
    return np.c_[-T[:, 1], T[:, 0]]


def sample_profiles(img, cnt, to_px, P, Nin, offs):
    """Bilinear-sample map luminance/saturation + coverage along INWARD normals.
    img: BGR mosaic, cnt: coverage, to_px: world->mosaic px. offs: offsets (world px), +ve = inward.
    Returns L, S, V (N, len(offs))."""
    import cv2

    X = P[:, None, :] + offs[None, :, None] * Nin[:, None, :]
    uv = to_px(X.reshape(-1, 2)).astype(np.float32)
    mx, my = uv[:, 0].reshape(len(P), -1), uv[:, 1].reshape(len(P), -1)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = cv2.remap(hls[..., 1], mx, my, cv2.INTER_LINEAR).astype(np.float32)
    S = cv2.remap(hls[..., 2], mx, my, cv2.INTER_LINEAR).astype(np.float32)
    V = cv2.remap((cnt > 0.5).astype(np.float32), mx, my, cv2.INTER_NEAREST)
    return L, S, V


def ridge_response(L, S, V, half_w: int = 2, ctx: int = 6, along: int = 15):
    """Thin bright low-saturation line response along each profile (world px sampling)."""
    from scipy.ndimage import uniform_filter1d

    core = uniform_filter1d(L, 2 * half_w + 1, axis=1)
    sm = uniform_filter1d(L, ctx, axis=1)
    # a thin line is darker on BOTH sides: compare with the BRIGHTER side (the mean of the two sides
    # passes bright->dark step edges, e.g. the turf/apron boundary that captured the R02A rail fit)
    side = np.maximum(np.roll(sm, half_w + ctx, axis=1), np.roll(sm, -(half_w + ctx), axis=1))
    from scipy.ndimage import minimum_filter1d

    Vw = minimum_filter1d(V, 2 * (half_w + 2 * ctx) + 1, axis=1)  # whole window must be on-map
    r = np.clip((core - side) * (S < 140) * Vw, 0, None)
    if along > 1:
        # a rail PERSISTS along s; painted markers / glyphs crossing the profile are thin along s
        from scipy.ndimage import median_filter

        r = median_filter(r, size=(along, 1), mode="nearest")
    return r


def dp_track(R, offs, prior_hi, *, jump: float, lam: float = 0.4, min_r: float = 4.0):
    """Viterbi over stations: choose one offset per station maximising ridge response with a
    smoothness penalty on |delta offset| (world px). ``prior_hi[n]`` = the innermost horse offset —
    the rail must lie INSIDE it (offset > prior_hi + margin is required by caller via R masking).
    Returns offset per station (NaN where no support)."""
    N, M = R.shape
    # HKJC_RAIL_SCORE=binary (opt-in, NOT adopted): every credible ridge scores +1, so the Viterbi
    # rewards CONTINUITY, not brightness. MEASURED: the raw score abandoned the thin movable rail for
    # the brighter fixed rail / apron edge ~6.6 m inside, bridged by a straight diagonal across open
    # turf (ST 05-31 and ST7 chute junctions, HV 04-22 R1 bend exit), giving field-wide lateral
    # "slides" of 4-6 m/s; on the ST 05-31 bend the raw rail followed the FIXED rail for ~4700
    # stations while the field ran outside the movable one (pixels, 2x). Binary follows the movable
    # rail and fixed ST 04-26 R1's marker grid (FINISH was on the 50 m marker). 26 races x 6 seeds
    # (rule declared before reading): hand-truth majority 57 -> 60 (null spread 3), per-seed mean
    # 50.7 -> 59.3 (ST 05-24 19/19 on every seed); but non-lateral gates 198.0 -> 193.8 per seed:
    # ST 04-26 R1 (0.05 s finish now #9 on 5/6 seeds, seeds 3-5 collapse), HV 05-27 R9 finish-n
    # (honest: the misplaced 10th horse is dropped), ST8 #2/#12 flips; HV 04-15 R1 lost its marker
    # grid (two marginal peaks); lateral 10/1 -> 10/3 (ST course-B races now measurable: 6.6 m vs
    # table 4.5 on both, correct rail in pixels).
    if os.environ.get("HKJC_RAIL_SCORE", "raw") == "binary":
        score = np.where(R >= min_r, 1.0, -1.0)
    else:
        score = np.where(R >= min_r, R, -1.0)
    acc = score[0].copy()
    back = np.zeros((N, M), int)
    oi = np.arange(M)
    step = offs[1] - offs[0]
    wj = int(jump / step)
    for n in range(1, N):
        best = np.full(M, -1e9)
        arg = np.zeros(M, int)
        for dj in range(-wj, wj + 1):
            src = np.clip(oi - dj, 0, M - 1)
            cand = acc[src] - lam * abs(dj * step)
            better = cand > best
            best[better] = cand[better]
            arg[better] = src[better]
        acc = best + score[n]
        back[n] = arg
    path = np.zeros(N, int)
    path[-1] = int(np.argmax(acc))
    for n in range(N - 1, 0, -1):
        path[n - 1] = back[n, path[n]]
    out = offs[path].astype(float)
    sup = R[np.arange(N), path] >= min_r
    out[~sup] = np.nan
    return out


@dataclass
class RailFrame:
    """Racing rail as a dense world-px polyline with arclength; projects points to (s, lateral).
    lateral > 0 = OUTSIDE the rail (toward the outer rail), in world px; s in world px along the rail
    in the direction of travel."""

    R: np.ndarray  # (N,2) rail points, direction of travel
    s: np.ndarray  # (N,) cumulative arclength (world px)
    Nout: np.ndarray  # (N,2) outward unit normals

    @classmethod
    def from_points(cls, R, inner, win=121):
        """Rail position AND normals from a local cubic fit over ``win`` stations (~12 m): real rails
        are smooth over tens of metres; per-station ridge snaps wobble (R02A ~0.1 m rms at ~9 m
        wavelength) and differentiating that wobble made the normals noisy (amplified at 5-10 m
        offsets -> lateral/d noise for wide runners). REJECTED: Gaussian sigma=3 stations."""
        from scipy.signal import savgol_filter

        if len(R) > win + 2:
            T = savgol_filter(R, win, 3, deriv=1, axis=0, mode="interp")
            R = savgol_filter(R, win, 3, axis=0, mode="interp")
        else:
            T = np.gradient(R, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]
        return cls(R, s, -inner * right_normal(T))

    def project(self, X, s_hint=None, window: float = 1500.0):
        """X (M,2) world px -> (s, lat) world px, via nearest rail vertex + tangent refinement.
        ``s_hint`` (M,) restricts the search to rail vertices with |s - hint| < window — REQUIRED
        when the trip revisits ground (multi-pass): the nearest vertex in space may be another lap."""
        from scipy.spatial import cKDTree

        if s_hint is None:
            _, j = cKDTree(self.R).query(X)
        else:
            j = np.empty(len(X), int)
            # bucket by hint so each bucket queries a local tree
            order = np.argsort(s_hint)
            for a in range(0, len(order), 2000):
                ids = order[a : a + 2000]
                lo, hi = s_hint[ids].min() - window, s_hint[ids].max() + window
                v = np.where((self.s >= lo) & (self.s <= hi))[0]
                tree = cKDTree(self.R[v])
                dd, jj = tree.query(X[ids], k=min(8, len(v)))
                jj = np.atleast_2d(jj)
                cand = v[jj]
                ok = np.abs(self.s[cand] - s_hint[ids][:, None]) < window
                dd = np.where(ok, np.atleast_2d(dd), np.inf)
                j[ids] = cand[np.arange(len(ids)), np.argmin(dd, axis=1)]
        T = np.c_[self.Nout[:, 1], -self.Nout[:, 0]]  # any unit tangent (sign irrelevant below)
        T = T * np.sign((np.gradient(self.R, axis=0) * T).sum(1))[:, None]
        d = X - self.R[j]
        along = (d * T[j]).sum(1)
        lat = (d * self.Nout[j]).sum(1)
        return self.s[j] + along, lat


def marker_profile(img, cnt, to_px, rail: "RailFrame", s_lo, s_hi, lat=(30.0, 450.0)):
    """Per-s (1 world px) response of thin bright lines PERPENDICULAR to the rail in an outward band
    — the render's painted distance markers (FINISH/50/100/150/200/300, drawn across the track)."""
    import cv2
    from scipy.ndimage import uniform_filter1d

    ss = np.arange(max(s_lo, 0), min(s_hi, rail.s[-1]), 1.0)
    j = np.searchsorted(rail.s, ss).clip(0, len(rail.s) - 1)
    T = np.gradient(rail.R, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    base = rail.R[j] + (ss - rail.s[j])[:, None] * T[j]
    ls = np.arange(lat[0], lat[1], 2.0)
    X = base[:, None, :] + ls[None, :, None] * rail.Nout[j][:, None, :]
    uv = to_px(X.reshape(-1, 2)).astype(np.float32)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = cv2.remap(
        hls[..., 1], uv[:, 0].reshape(len(ss), -1), uv[:, 1].reshape(len(ss), -1), cv2.INTER_LINEAR
    ).astype(np.float32)
    V = cv2.remap(
        (cnt > 0.5).astype(np.float32),
        uv[:, 0].reshape(len(ss), -1),
        uv[:, 1].reshape(len(ss), -1),
        cv2.INTER_NEAREST,
    )
    # line contrast along s per lat row, then the fraction of rows where a line is present
    c = L - uniform_filter1d(L, 15, axis=0)
    resp = ((c > 18) & (V > 0)).sum(1) / np.maximum(V.sum(1), 1)
    return ss, resp


def fit_marker_grid(ss, resp, *, spacing_m=50.0, ppm_range=(14.0, 24.0), kmax=8, tol=40.0):
    """Fit s_k = sF + direction*k*spacing*ppm to the marker peaks. Returns (ppm, sF, matched k list,
    peak positions) or None. Direction: markers count UP away from the finish, against travel."""
    from scipy.signal import find_peaks

    pk, pr = find_peaks(resp, height=0.10, distance=int(20 * ppm_range[0]), prominence=0.08)
    # tolerance is generous (map scale drift ~1-5% between markers); LS refine below
    if len(pk) < 3:
        return None
    pos = ss[pk]
    best = None
    for ppm in np.arange(ppm_range[0], ppm_range[1], 0.01):
        step = spacing_m * ppm
        for sF in pos:  # FINISH = the downstream-most matched peak
            k = (sF - pos) / step
            kr = np.round(k)
            ok = (np.abs(k - kr) * step < tol) & (kr >= 0) & (kr <= kmax)
            sc = ok.sum() - 0.01 * (np.abs(k - kr)[ok] * step).sum()
            if best is None or sc > best[0]:
                best = (sc, ppm, sF, kr[ok].astype(int), pos[ok])
    if best[0] < 3.5:
        return None
    _, ppm, sF, ks, ps = best
    # least-squares refine ppm, sF on matched peaks
    A = np.c_[np.ones(len(ks)), -ks * spacing_m]
    (sF, ppm), *_ = np.linalg.lstsq(A, ps, rcond=None)
    return float(ppm), float(sF), ks, ps


def tick_events(fu, s_lead, odo):
    """Odometer TICK events (V+10 -> V within 3 frames): (s_lead at the tick, V)."""
    o = clean_odometer(odo)
    ev_s, ev_v = [], []
    last = None
    for k in range(len(o)):
        if np.isnan(o[k]):
            continue
        if last is not None and o[last] - o[k] == 10 and k - last <= 3:
            ev_s.append(0.5 * (s_lead[last] + s_lead[k]))
            ev_v.append(o[k])
        last = k
    return np.array(ev_s), np.array(ev_v)


def fit_marker_grid_seeded(ss, resp, ev_s, ev_v, *, spacing_m=50.0, kmax=8, rel=0.08, win_m=60.0):
    """Marker grid SEEDED by the odometer ticks (owner's prior: markers are exactly 50 m apart):
    scale within +-rel of the ticks' zone scale (tick px/m 3-4 % above paint px/m: the HUD counts
    FEWER metres than the paint), FINISH within +-win_m of the tick-predicted finish; peaks at low
    thresholds, scored by response.
    The unseeded fit locked onto a coverage-edge artefact as FINISH on HV 0429 and read the true
    markers as map-scale drift. -> (ppm, sF, ks, ps) or None."""
    from scipy.signal import find_peaks

    zone = (ev_v > 0) & (ev_v < 400)
    if zone.sum() < 5:
        return None
    a, b = np.polyfit(ev_v[zone], ev_s[zone], 1)  # s = b + a*V  (a < 0: s falls as V grows)
    ppm_t, sF_t = -a, b
    pk, _ = find_peaks(resp, height=0.04, distance=int(15 * ppm_t), prominence=0.03)
    if len(pk) < 3:
        return None
    pos, w = ss[pk], resp[pk]
    best = None
    for ppm in np.linspace(ppm_t * (1 - rel), ppm_t * (1 + rel), 81):
        step = spacing_m * ppm
        tol = 0.35 * spacing_m * ppm * 0.1  # 1.75 m
        for sF in np.arange(sF_t - win_m * ppm, sF_t + win_m * ppm, 0.25 * ppm):
            k = (sF - pos) / step
            kr = np.round(k)
            ok = (np.abs(k - kr) * step < tol) & (kr >= 0) & (kr <= kmax)
            if ok.sum() < 3 or len(np.unique(kr[ok])) < ok.sum():
                continue
            # prior: paint scale ~ tick-zone scale / 1.035 (MEASURED HUD/paint ratio), sd 3 %
            sc = w[ok].sum() + 0.2 * ok.sum() - 0.5 * ((ppm * 1.035 / ppm_t - 1) / 0.03) ** 2
            if best is None or sc > best[0]:
                best = (sc, kr[ok].astype(int), pos[ok])
    if best is None:
        return None
    _, ks, ps = best
    if len(ks) < 4 or 0 not in ks:  # accept only grids that include the FINISH line + >= 3 more
        return None
    A = np.c_[np.ones(len(ks)), -ks * spacing_m]
    (sF, ppm), *_ = np.linalg.lstsq(A, ps, rcond=None)
    return float(ppm), float(sF), ks, ps


TIME_RES_M = 10.0  # m: tick residual (vs its median) that flags a time-local pose slip
TIME_MIN_TICKS = 3  # consecutive flagged ticks


def time_local_shift(tf, d_lead, odo, *, res_m=None, min_ticks=None):
    """Per-frame ALONG-TRACK shift (m, subtract from d) where the calibrated leader departs from the
    odometer ticks for a few seconds: a per-FRAME pose slip that no d(s) mapping can absorb.

    ``tf``: frame times (s), ``d_lead``: calibrated leader distance-to-go per frame, ``odo``: HUD
    odometer per frame. At each tick event (V+10 -> V) the residual r = d_lead - V (minus its median,
    i.e. the convention/lag offset); runs of >= ``min_ticks`` consecutive ticks with |r| > ``res_m``
    (+- one tick) get shift(t) = r interpolated in time, tapered to 0 over one tick at both ends.
    MEASURED (pixels + ticks): on two of the three 1000 m straights the render camera swings 180 deg at
    t ~ 28-32 s and the pose solve slips per frame: ST 0419 R3 r = -29 m at 30.2 s -> +26 m at 33.4 s
    (after the d(s) refit), ST 0509 R1 -14 -> +16; ST 0607 R8 one tick at -10. Labelled detections fall to
    0-6 % in that window. Every other race: no run of 3 ticks > 10 m. -> (shift per frame, n_windows)"""
    res_m = TIME_RES_M if res_m is None else res_m
    min_ticks = TIME_MIN_TICKS if min_ticks is None else min_ticks
    o = clean_odometer(odo)
    et, er = [], []
    last = None
    for k in range(len(o)):
        if np.isnan(o[k]):
            continue
        if last is not None and o[last] - o[k] == 10 and k - last <= 3:
            et.append(0.5 * (tf[last] + tf[k]))
            er.append(0.5 * (d_lead[last] + d_lead[k]) - o[k])
        last = k
    shift = np.zeros(len(tf))
    if len(er) < 2 * min_ticks:
        return shift, 0
    et, er = np.array(et), np.array(er)
    er = er - np.median(er)
    bad = np.abs(er) > res_m
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    n = 0
    for a0, a1 in zip(edges[::2], edges[1::2]):
        if a1 - a0 < min_ticks:
            continue
        i0, i1 = max(0, a0 - 1), min(len(et) - 1, a1)  # one tick either side: the taper
        kt = np.r_[et[i0], et[a0:a1], et[i1]]
        kr = np.r_[0.0, er[a0:a1], 0.0]
        m = (tf >= kt[0]) & (tf <= kt[-1])
        shift[m] = np.interp(tf[m], kt, kr)
        n += 1
    return shift, n


def clean_odometer(odo, win=15, tol=15.0):
    """Drop OCR misreads: readings > ``tol`` m from a running median of the interpolated sequence."""
    from scipy.ndimage import median_filter

    o = np.asarray(odo, float).copy()
    ok = ~np.isnan(o)
    if ok.sum() < 5:
        return o
    med = median_filter(np.interp(np.arange(len(o)), np.where(ok)[0], o[ok]), win, mode="nearest")
    o[np.abs(o - med) > tol] = np.nan
    return o


LOCAL_RES_M = 10.0  # m: smoothed tick residual that triggers a local dense-knot refit
LOCAL_KNOT_M = 25.0  # m: knot spacing of that local refit
LOCAL_MIN_TICKS = 4  # consecutive flagged ticks (~40 m) needed


def calibrate_distance_ticks(
    fu,
    s_lead,
    odo,
    ppm0,
    sF,
    *,
    c=0.0,
    knot_m=150.0,
    huber=3.0,
    smooth=2.0,
    pin_finish=None,
    have_markers=True,
):
    """Distance-to-go d(s) from odometer TICK events ALONE (in-video; no painted markers needed).

    The display is a round-up readout: it ticks V+10 -> V as the leader REACHES V, so at a tick the
    leader's true distance-to-go = V + c. The convention c is a property of the render software,
    measured against the painted markers on two races (R08 -0.55 m, R03 +0.42 m) -> c = 0.
    Model: a robust global line through the ticks (scale + finish position, replacing the marker
    grid, which mis-assigned markers on some races: R02A/HVB) + a smooth cubic-spline correction B(s)
    for slow map-scale drift (2nd-difference penalty: linear trends stay free).
    ``ppm0``/``sF`` (marker estimates) are only an initial guess — unless ``pin_finish`` (True/False;
    None = auto): the painted FINISH line is then the exact d=0 anchor. Auto pins only when the
    marker grid is CONSISTENT with the ticks (marker-zone offset |c_m| < 3 m and ppm within 3%) — a
    cross-feed check; a mis-assigned marker grid fails it and falls back to the ticks. Returns
    (d_of_s, c, ppm_local, resid_at_ticks, (ppm_odo, sF_odo))."""
    from scipy.interpolate import BSpline
    from scipy.optimize import least_squares

    o = clean_odometer(odo)
    ev_s, ev_v = [], []
    last = None
    for k in range(len(o)):
        if np.isnan(o[k]):
            continue
        if last is not None and o[last] - o[k] == 10 and k - last <= 3:
            ev_s.append(0.5 * (s_lead[last] + s_lead[k]))
            ev_v.append(o[k])
        last = k
    ev_s, ev_v = np.array(ev_s), np.array(ev_v)
    zone = (ev_v > 0) & (ev_v < 200)
    c_m = float(np.median((sF - ev_s[zone]) / ppm0 - ev_v[zone])) if zone.sum() >= 5 else np.inf
    # The tick convention c is MEASURED per race on its own painted markers (in-video: HUD vs paint):
    # Sha Tin c ~ -3.2 m (ST6 -3.2, ST7 -3.3), HV ~ 0 (R08 -1.4, R03 +0.4). The fixed c = 0 produced
    # a constant +0.3 s at every boundary at ST (misread as a "render lag"). A marker grid shifted by
    # n markers shows c_m ~ -50 n + c (ST8 -52.9, ST5 -153.6): resolve n, correct the finish.
    n_shift = int(np.round(c_m / 50.0)) if np.isfinite(c_m) else 0
    c_res = c_m - 50.0 * n_shift
    grid_ok = have_markers and np.isfinite(c_m) and abs(c_res) < 6.0
    ppm_zone = -1.0 / np.polyfit(ev_s[zone], ev_v[zone], 1)[0] if zone.sum() >= 5 else np.nan
    if pin_finish is None:
        pin_finish = bool(grid_ok) and abs(ppm_zone / ppm0 - 1) < 0.08
    if pin_finish:
        # the measured convention is used ONLY with a fully consistent, pinned grid (AWT: the grid
        # was unreliable and applying its offset unpinned made ST5/ST6-0401 worse)
        sF = sF - n_shift * 50.0 * ppm0  # markers count UP against travel: s decreases with k
        c = c_res if grid_ok else c
    target = ev_v + c
    # robust global line d = (sF' - s) / ppm'
    lin = least_squares(
        lambda p: (p[0] - ev_s) / p[1] - target, [sF, ppm0], loss="soft_l1", f_scale=huber
    )
    sF1, ppm1 = float(lin.x[0]), float(lin.x[1])
    # scale check IN the marker zone (map scale drifts along the race, so the global tick ppm may
    # legitimately differ from the local marker ppm: R02A 20.26 global vs 21.72 local, c_m +0.3 m)
    # MEASURED: over the final 200 m the tick px/m is 2.7-4.3 % above the painted-marker px/m, i.e.
    # the HUD counts FEWER metres than the markers (R08 4.3, R03 3.4, R02A 2.7 %; same span — marker ppm 0-200 m = full-grid ppm, so a
    # span/map-zoom mismatch is RULED OUT), while the global tick scale and the zone offset agree
    # (< 1.5 m). A property of the render's HUD. Mis-assigned grids are off by tens of metres or
    # > 20 % in scale, so the offset is the discriminating test and the scale tolerance is 8 %.
    if pin_finish:
        sF1 = sF
    lo, hi = min(ev_s.min(), sF1) - 50, max(ev_s.max(), sF1) + 50
    nk = max(2, int(np.ceil((hi - lo) / (knot_m * ppm1))))
    k = 3
    t = np.r_[[lo] * k, np.linspace(lo, hi, nk + 1), [hi] * k]
    nb = len(t) - k - 1

    def res(coef):
        Bv = BSpline(t, coef, k, extrapolate=True)
        pin = 50.0 * Bv(np.array([sF1])) if pin_finish else np.zeros(0)
        return np.r_[(sF1 - ev_s) / ppm1 + Bv(ev_s) - target, pin, smooth * np.diff(coef, 2)]

    r = least_squares(res, np.zeros(nb), loss="huber", f_scale=huber)
    # the BASE fit's derivative sets the local px/m for LATERAL offsets: the local refit below is an
    # along-track slip (the leader stalls then jumps); letting its derivative rescale lateral put
    # ST3's p99 lateral at 30 m (course width 30.5; base fit 19.4 m)
    dB = BSpline(t, r.x, k, extrapolate=True).derivative()
    # LOCAL map distortion: where the smoothed tick residual stays > LOCAL_RES_M, refit with dense
    # knots (LOCAL_KNOT_M) around it. MEASURED (26 turf races, in-sample): both 1000 m straights
    # (ST 0419 R3, 0509 R1) show a 14-28 m distortion at ~410 m to go: on all three straights the
    # render camera SWINGS 180 deg between t ~ 28 and 32 s, and on these two the pose solve takes a
    # ~2.5 % scale step during the swing (0.963 -> 0.938, 1.003 -> 0.978; ST 0607 R8: 0.993 ->
    # 0.992, no distortion). The leader "stalls" 40 m in 4 s, then jumps; the 150 m spline (huber:
    # outliers) ignored it -> leader 1.66 s late at 400 m. Every other race: smoothed residual
    # <= 6.9 m, so the refit never triggers there.
    from scipy.ndimage import median_filter

    order = np.argsort(ev_s)
    rt = res(r.x)[: len(ev_s)][order]
    bad = np.abs(median_filter(rt, 5, mode="nearest")) > LOCAL_RES_M
    # contiguous runs of >= LOCAL_MIN_TICKS flagged ticks only (an isolated edge tick -- the race
    # start, where the pose is weak -- sent dense knots into the sparse start: 170 m residuals),
    # dense knots over each run +- 50 m, kept inside the tick-covered range
    es = ev_s[order]
    extra = []
    import os

    if os.environ.get("HKJC_LOCAL_REFIT", "1") == "0":  # diagnostic
        bad[:] = False
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    for a0, a1 in zip(edges[::2], edges[1::2]):
        if a1 - a0 >= LOCAL_MIN_TICKS:
            x0 = max(es[a0] - 50 * ppm1, es[0] + 25 * ppm1)
            x1 = min(es[a1 - 1] + 50 * ppm1, es[-1] - 25 * ppm1)
            extra += list(np.arange(x0, x1, LOCAL_KNOT_M * ppm1))
    if extra:
        extra = np.array(extra)
        inner_k = np.unique(np.r_[np.linspace(lo, hi, nk + 1), extra[(extra > lo) & (extra < hi)]])
        t = np.r_[[lo] * k, inner_k, [hi] * k]
        nb = len(t) - k - 1
        r = least_squares(res, np.zeros(nb), loss="huber", f_scale=huber)
    Bsp = BSpline(t, r.x, k, extrapolate=True)

    def d_of_s(x):
        x = np.asarray(x, float)
        return (sF1 - x) / ppm1 + Bsp(x)

    def ppm_local(x):
        x = np.clip(np.asarray(x, float), lo, hi)
        return 1.0 / np.clip(1.0 / ppm1 - dB(x), 1e-4, None)

    return d_of_s, c, ppm_local, res(r.x)[: len(ev_s)], (ppm1, sF1, bool(pin_finish), c_m, n_shift)


# ----------------------------------------------------------------------------- racing rail
def inner_field(Ps, Nin, X, win=25, q=90):
    """Per-station inward offset of the INNERMOST horses (robust percentile over a station window)."""
    from scipy.spatial import cKDTree

    j = cKDTree(Ps).query(X)[1]
    ho = ((X - Ps[j]) * Nin[j]).sum(1)
    order = np.argsort(j)
    js, hs = j[order], ho[order]
    bnd = np.searchsorted(js, np.arange(len(Ps) + 1))
    hin = np.full(len(Ps), np.nan)
    for n in range(len(Ps)):
        a, b = bnd[max(0, n - win)], bnd[min(len(Ps), n + win + 1)]
        if b - a >= 8:
            hin[n] = np.percentile(hs[a:b], q)
    good = ~np.isnan(hin)
    return np.interp(np.arange(len(Ps)), np.where(good)[0], hin[good])


def smooth_rail(W, ok, *, wavelength=200.0, huber=10.0, iters=6):
    """HKJC_RAIL_SMOOTH=whittaker (opt-in): the rail is a SMOOTH CURVE. Robust (Huber, IRLS)
    support-weighted Whittaker smoother in world px with a THIRD-difference penalty (straights and
    constant-radius arcs cost nothing; cutoff ~``wavelength`` stations = 20 m at 2 px/station).
    Unsupported stations get no data weight, so gaps are bridged by the smooth curve instead of
    linear segments. MEASURED (9 problem races, 4 m-chord kinks > 8 deg): raw 16/6/6/18/13/145/14/64/66
    -> 3/5/2/2/0/7/0/0/4, p95 distance to supported ridge points <= 8 px (0.4 m); ST8's rail heading
    over the final 150 m swung +-12 deg -> +-5 deg. Off by default -> returns W unchanged."""
    if os.environ.get("HKJC_RAIL_SMOOTH", "off") != "whittaker" or len(W) < 10 or ok.sum() < 10:
        return W
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = len(W)
    D = sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(n - 3, n))
    P = (wavelength / (2 * np.pi)) ** 6 * (D.T @ D)
    w0 = ok.astype(float) + 1e-4
    w, Z = w0.copy(), W.copy()
    for _ in range(iters):
        A = (sparse.diags(w) + P).tocsc()
        Z = np.c_[spsolve(A, w * W[:, 0]), spsolve(A, w * W[:, 1])]
        r = np.hypot(*(W - Z).T)
        w = w0 * np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
    # unsupported END runs carry no data: continue straight along the last supported tangent (ST3
    # binary: the rail beyond the finish curled back on itself, a 138 deg kink, at support 0.12)
    idx = np.flatnonzero(ok)
    lo, hi = int(idx[0]), int(idx[-1])
    k = min(20, (hi - lo) // 2)
    if k >= 2:
        step = np.hypot(*np.diff(Z[lo : hi + 1], axis=0).T).mean()
        t0 = Z[lo] - Z[lo + k]
        t0 /= np.linalg.norm(t0) + 1e-12
        t1 = Z[hi] - Z[hi - k]
        t1 /= np.linalg.norm(t1) + 1e-12
        Z[:lo] = Z[lo] + np.outer(np.arange(lo, 0, -1), t0) * step
        Z[hi + 1 :] = Z[hi] + np.outer(np.arange(1, len(Z) - hi), t1) * step
    return Z


def fill_world(W, ok):
    """Bridge unsupported rail stations in WORLD coordinates (linear in station index between the
    supported neighbours). Interpolating the OFFSET instead puts a hump in the rail wherever the
    station scaffold itself jogs (R08 home straight at 200 m: a ~1.5-2 m dip into the track that
    bent the normals, so (d, lat) jumped and two adjacent horses' tracklets swapped). REJECTED:
    offset-space interpolation."""
    if ok.all() or ok.sum() < 2:
        return W
    n_ = np.arange(len(W))
    lo, hi = n_[ok][0], n_[ok][-1]
    gap = ~ok & (n_ > lo) & (n_ < hi)  # INTERIOR gaps only: np.interp clamps the array ends to a
    out = (
        W.copy()
    )  # constant point -> zero tangents/NaN normals at the finish (HV 0429 markers lost)
    out[gap, 0] = np.interp(n_[gap], n_[ok], W[ok, 0])
    out[gap, 1] = np.interp(n_[gap], n_[ok], W[ok, 1])
    return out


def despike_world(W, ok, *, win=241, thr=10.0, iters=4):
    """Robust smoothness check in WORLD px: rails are smooth, so supported points more than ``thr``
    px (~0.55 m) off a local QUADRATIC trend (Savitzky-Golay, ``win`` stations x 2 px ~ 26 m) are
    false ridges (R08 home straight: a faint line ~1.7 m inside the track captured the Viterbi
    path for ~40 stations). They are demoted to unsupported and bridged with ``fill_world``.
    REJECTED: a Gaussian trend (mode='nearest' flattens curvature at the array ends -> demoted the
    good rail at the finish, shifted the marker grid by one marker). -> (W, ok)."""
    from scipy.signal import savgol_filter

    ok = ok.copy()
    if len(W) < win + 2:
        return fill_world(W, ok), ok
    for _ in range(iters):
        F = fill_world(W, ok)
        Tr = savgol_filter(F, win, 2, axis=0, mode="interp")
        T = savgol_filter(F, win, 2, deriv=1, axis=0, mode="interp")
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        dev = np.abs(((W - Tr) * right_normal(T)).sum(1))
        bad = ok & (dev > thr)
        if not bad.any():
            break
        ok &= ~bad
    return fill_world(W, ok), ok


def refine_rail(img, cnt, to_px, Rail, inner, ok, half=25.0):
    """Snap the rail to the sub-pixel ridge peak along its own normals; keep coarse where unsupported."""
    from scipy.ndimage import gaussian_filter1d

    T = np.gradient(Rail, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nn = inner * right_normal(T)
    offs = np.arange(-half, half + 0.01, 0.5)
    L, S, V = sample_profiles(img, cnt, to_px, Rail, Nn, offs)
    R = ridge_response(L, S, V, half_w=3, ctx=10)
    k = np.argmax(R, axis=1)
    pk = R[np.arange(len(R)), k]
    sup = (pk >= 4.0) & (k > 0) & (k < len(offs) - 1)
    km = np.clip(k, 1, len(offs) - 2)
    a, b, c = (R[np.arange(len(R)), km + d] for d in (-1, 0, 1))
    den = a - 2 * b + c
    frac = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / den, 0.0).clip(-1, 1)
    dlt = offs[km] + frac * 0.5
    n_ = np.arange(len(Rail))
    if sup.sum() < 10:
        return Rail, ok
    dlt = gaussian_filter1d(np.interp(n_, n_[sup], dlt[sup]), 4)
    keep = ok & sup
    return fill_world(Rail + dlt[:, None] * Nn, keep), keep


def extract_racing_rail(img, cnt, to_px, G, shape, X, handedness=None):
    """Racing rail on the map, in 3 passes:
      A. stations along the (extended) camera-centre path; innermost-horse offsets per station;
      B. re-station along the field's INNER EDGE (parallel to the racing rail by construction);
         thin-bright-ridge profiles -2..+14 m inward; Viterbi DP (stiff: rails are smooth);
      C. sub-px refinement along the rail's own normals.
    Also returns each camera frame's station index (time-ordered -> disambiguates revisited ground).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter
    from .pose import centres

    C = centres(G, shape)

    def ext(a, b, L=700.0):
        t = (a - b) / np.linalg.norm(a - b)
        return a + np.outer(np.arange(20, L + 1, 20), t)

    Ce = np.vstack([ext(C[0], C[min(30, len(C) - 1)])[::-1], C, ext(C[-1], C[max(-31, -len(C))])])
    Ps, T, _s = resample_path(Ce, step=2.0, smooth=80.0)
    inner = inner_side(T, handedness)
    Nin = inner * right_normal(T)
    hin = inner_field(Ps, Nin, X)
    Q = (
        Ps
        + gaussian_filter1d(median_filter(hin, 51, mode="nearest"), 40, mode="nearest")[:, None]
        * Nin
    )
    Ps, T, _s = resample_path(Q, step=2.0, smooth=40.0)
    Nin = inner * right_normal(T)
    hin = inner_field(Ps, Nin, X)
    offs = np.arange(-40, 261, 1.0)
    L, S, V = sample_profiles(img, cnt, to_px, Ps, Nin, offs)
    R = ridge_response(L, S, V)
    for n in range(len(Ps)):
        R[n, (offs < hin[n] - 30) | (offs > hin[n] + 200)] = 0
    rail_off = dp_track(R, offs, hin, jump=2.0, lam=2.0, min_r=4.0)
    ok = ~np.isnan(rail_off)
    n_ = np.arange(len(Ps))
    ro = gaussian_filter1d(np.interp(n_, n_[ok], rail_off[ok]), 5)
    Rail0, ok = despike_world(Ps + ro[:, None] * Nin, ok)
    Rail, ok = refine_rail(img, cnt, to_px, Rail0, inner, ok)
    Rail, ok = despike_world(Rail, ok)
    Rail = smooth_rail(Rail, ok)
    st = np.zeros(len(C), int)
    last = 0
    for k, c in enumerate(C):
        lo, hi = max(0, last - 50), min(len(Ps), last + 800)
        st[k] = lo + int(np.argmin(np.hypot(*(Ps[lo:hi] - c).T)))
        last = st[k]
    return dict(rail=Rail, support=ok, inner=inner, stations=Ps, Nin=Nin, st_of_frame=st)


def movable_rail_check(img, cnt, to_px, rail, inner, X, expected_m, *, ppm_lo=15.0, ppm_hi=24.0):
    """Is the extracted rail the FIXED inner rail instead of the movable racing rail? Course-table
    prior (owner: HKJC geometry is known): the movable rail lies ``expected_m`` (A width - course
    width; B ~3 m) OUTWARD of the fixed rail. The strongest ridge can be the fixed rail (HV 0527, B:
    every lateral inflated ~3.5 m). Test: a supported ridge exists at that offset toward the field on
    most stations AND the innermost horses still lie outside it. -> (shift_px or 0.0, support).
    REJECTED: an exponential nearest-to-field preference in the DP (pulled the rail into the field
    on the home straight)."""
    from scipy.spatial import cKDTree

    if expected_m < 1.5 or len(X) < 50:
        return 0.0, 0.0
    T = np.gradient(rail, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    Nout = -inner * right_normal(T)
    # innermost-horse lateral per station (px, outward positive)
    j = cKDTree(rail).query(X)[1]
    lat = ((X - rail[j]) * Nout[j]).sum(1)
    per = {}
    for jj, ll in zip(j, lat):
        per.setdefault(jj // 50, []).append(ll)
    inner_h = np.array([np.percentile(v, 5) for v in per.values() if len(v) >= 10])
    if len(inner_h) < 10:
        return 0.0, 0.0
    h_in = float(np.median(inner_h))
    lo, hi = expected_m * ppm_lo * 0.8, expected_m * ppm_hi * 1.2
    if h_in < lo + 15:  # horses too close: no room for a movable rail outside this one
        return 0.0, 0.0
    offs = np.arange(lo, min(hi, h_in - 10), 1.0)
    if len(offs) < 5:
        return 0.0, 0.0
    L, S, V = sample_profiles(img, cnt, to_px, rail[::5], -Nout[::5], -offs)
    Rr = ridge_response(L, S, V)
    k = np.argmax(Rr, axis=1)
    sup = Rr[np.arange(len(Rr)), k] >= 4.0
    if sup.mean() < 0.5:
        return 0.0, float(sup.mean())
    return float(np.median(offs[k[sup]])), float(sup.mean())


def fixed_rail_gap(img, cnt, to_px, rail, inner, ppm, expected_m, *, band_m=2.5, min_support=0.15):
    """Lateral-scale check: measure the gap from the racing rail to the FIXED inner rail where the
    map shows it (search a +-band around the course-table gap, inward). Returns (median gap m,
    support fraction) or (nan, 0) when not measurable (e.g. A course: the rails coincide)."""
    if expected_m < 3.0:
        return float("nan"), 0.0
    T = np.gradient(rail, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Nin = inner * right_normal(T)
    # ``ridge_response`` compares each offset with its sides via np.roll, which WRAPS at the window
    # ends. Sampling only the +-band made offsets near the band edge compare against the far end
    # and fire. MEASURED (ST 06-07 R6): on pod run Y's map the artefact at 10.8-11.5 m won 3755
    # stations against 232 at the real rail (9.73 m on a wide window), giving 11.27 m; V's map had
    # the same artefact at 786 stations. So the profile is sampled on a PADDED window and the
    # argmax is taken inside the band only.
    pad = 32
    offs_all = np.arange((expected_m - band_m) * ppm - pad, (expected_m + band_m) * ppm + pad, 1.0)
    L, S, V = sample_profiles(img, cnt, to_px, rail, Nin, offs_all)
    R = ridge_response(L, S, V)[:, pad:-pad]
    offs = offs_all[pad:-pad]
    k = np.argmax(R, axis=1)
    pk = R[np.arange(len(R)), k]
    sup = pk >= 6.0
    if sup.mean() < min_support:
        return float("nan"), float(sup.mean())
    return float(np.median(offs[k[sup]]) / ppm), float(sup.mean())

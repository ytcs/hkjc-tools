"""Aerial ANONYMOUS per-horse PATHS on the rail backbone (task #104).

Uses the PROMOTED library primitives (hkjc.vision.feeds.aerial_detect.detect_edgefill +
geometry_cv.fit_rail_curve/project_to_polyline) to turn per-frame blobs into persistent ANONYMOUS
tracklets, with the rail doing two jobs (IDENTITY is intentionally NOT assigned here — it is better
recovered from the drone feed; these tracks stay anonymous):

  (1) RAIL-BOUND DETECTION — project every blob to the rail; the field all sits on ONE side within
      ~one track-width. Blobs on the infield side, or far beyond the field's own lateral spread, are
      off-track background (hedge/markers/text) and are REJECTED. Lane width is data-driven
      (p98 of the field's own |perp|), never hand-tuned.
  (2) LONGITUDINAL ORDER + LATERAL TRIP — (s, perp) = arc-length along / perpendicular offset from
      the rail give the real front-to-back order (the shoulder-to-shoulder fix) and lateral trip.

Tracklets: greedy nearest-neighbour association in panel pixels across stride-4 frames (avatars move
smoothly), gated by a motion radius. Each tracklet is one anonymous runner; s/perp are only globally
consistent once the rail is in a consistent frame (the geometric-prediction rail) — within a stable
rail segment the per-horse band is already clean.

Diagnostics (data_cache/diag/): a few overlays (rail + s-rank order) and an anonymous tracklet
perp-vs-frame plot, plus printed in-lane + tracklet stats. VISUALLY SCRUTINISE.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_aerial_track_id.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

# RACE env selects the validation race; all are HV "C" (same rail geometry) -> tests field-size +
# distance generalisation. (video, dist m, field size, course)
RACES = {
    "R08": ("aerial_20260603_08", 1800.0, 11, "C"),
    "R03": ("aerial_20260603_03", 1200.0, 12, "C"),
    "R06": ("aerial_20260603_06", 1200.0, 13, "C"),
    "R01": ("aerial_20260603_01", 1650.0, 12, "C"),
    "R02A": ("aerial_20260520_02", 1500.0, 12, "A"),   # A-config: single-rail-dominant (outer ~28% only)
}
RACE = os.environ.get("RACE", "R08")
_vid, DIST, FIELD, COURSE = RACES[RACE]
AERIAL_VIDEO = f"data_cache/video/{_vid}.mp4"
OBS = f"data_cache/cv_joint_aerial_{RACE}_obs.npz"
PASS1_CACHE = f"data_cache/cv_trackid_pass1_{RACE}.npz"   # per-frame detect+rail cache (skip 12-min decode)
DIAG = "data_cache/diag"
SFX = "" if RACE == "R08" else f"_{RACE}"     # race suffix on diag outputs (keep R08 canonical names)
HALF = 960
STRIDE = 2               # the render visibly updates only ~every 2nd frame at 25 fps -> stride-2 is
                         # lossless and halves decode/processing. Gates are in VIDEO-frame units
                         # (frame index i), so they stay valid under any stride.
GATE_PX = 60.0           # max avatar displacement between stride-4 frames (panel px) for association
LANE_K = 1.35            # lane half-width = LANE_K * p98(field |perp|): data-driven, not hand-tuned
SAVE_FRAMES = [None, 1080, 1760, 2440]   # None -> seed frame; rest = mid-race ID overlays


def _ry_side(perps):
    """Running side sign = sign of the field's median perp (the bulk sits on one side of the rail)."""
    m = float(np.median(perps))
    return 1.0 if m >= 0 else -1.0


def _medfilt(a, w=5):
    """NaN-aware rolling-median filter — de-spikes; NaN (rejected/invalid) points stay gaps."""
    a = np.asarray(a, float)
    if len(a) < 3:
        return a
    h = w // 2
    out = np.empty(len(a))
    for i in range(len(a)):
        win = a[max(0, i - h):i + h + 1]
        win = win[np.isfinite(win)]
        out[i] = float(np.median(win)) if len(win) else np.nan
    return out


def _robust_zoom_track(sep_raw, fw=90, mad_k=2.5, return_trust=False):
    """Robust near-constant-zoom inner->outer separation track.

    The virtual camera holds a near-CONSTANT zoom, so the true 22.5 m rail separation is a smooth,
    slowly-varying px curve. A wrong outer (the rail's shadow / a parallel marker -> too NEAR; a far
    background line -> too FAR) injects a separation off that curve for WHOLE stretches, which a local
    rolling median cannot reject (it chases the regional bias). So gate GLOBALLY first against the
    robust zoom level (kills both-direction gross outliers even where most outers are wrong), then a
    gentle local median tracks the slow drift among survivors, and interp the rejected gaps.

    With ``return_trust`` also returns a per-frame bool: True where the raw outer SURVIVED the gate
    (the scale is anchored on a believed-good 22.5 m ruler), False where it was interpolated.
    """
    n = len(sep_raw)
    idx = np.arange(n)
    ok = np.isfinite(sep_raw)
    if ok.sum() < 5:
        g = float(np.nanmedian(sep_raw)) if ok.any() else 200.0
        out = np.full(n, g)
        return (out, np.zeros(n, bool)) if return_trust else out
    f = idx[ok].astype(float)
    v = np.asarray(sep_raw, float)[ok]
    m0 = np.median(v)
    mad0 = np.median(np.abs(v - m0)) + 1e-6
    glob = np.abs(v - m0) <= mad_k * 1.4826 * mad0
    keep = glob.copy()
    smooth = np.full(len(v), m0)
    for _ in range(2):
        for j in range(len(v)):
            w = keep & (np.abs(f - f[j]) <= fw)
            smooth[j] = np.median(v[w]) if w.sum() >= 5 else m0
        resid = v - smooth
        mad = np.median(np.abs(resid[keep])) + 1e-6
        keep = glob & (np.abs(resid) <= mad_k * 1.4826 * mad)
        if keep.sum() < 5:
            break
    out = np.interp(idx, f[keep], smooth[keep])
    if not return_trust:
        return out
    trust = np.zeros(n, bool)
    trust[idx[ok][keep]] = True
    return out, trust


def _nanmedfilt(a, w):
    """NaN-aware rolling-median over an array (de-spike; gaps stay gaps)."""
    a = np.asarray(a, float)
    h = w // 2
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        win = a[max(0, i - h):i + h + 1]
        win = win[np.isfinite(win)]
        if len(win):
            out[i] = np.median(win)
    return out


def _consensus(M, min_n):
    """Per-frame median over tracks, only where >= ``min_n`` tracks contribute; else NaN -> interp.

    The robustness core: a per-frame nuisance (rail baseline / longitudinal origin / scale) is taken
    from the FIELD CONSENSUS (median over all co-visible horses), never one fragile reference, so a
    single mis-detected/-associated track cannot move it.
    """
    cnt = np.isfinite(M).sum(0)
    med = np.full(M.shape[1], np.nan)
    good = cnt >= min_n
    if good.any():
        med[good] = np.nanmedian(M[:, good], 0)
        med = np.interp(np.arange(M.shape[1]), np.where(good)[0], med[good])
    return med


def field_consensus_recon(S, P, odo, scale0, trust, rail_w, sep_global, ylat_max=25.0,
                          min_cons=4, scale_tol=0.06, n_iter=5):
    """Field-consensus FACTORISATION of the per-horse 2D path.

    Replaces the two fragile SINGLE-reference per-frame anchors (p96 leader for longitudinal; one
    outer line for lateral scale) with N-horse consensus nuisances + ONE absolute anchor each:

      LATERAL    perp_i(f) = b(f) + lane_px_i(f);  y_lat_i = lane_px_i * scale(f).
                 b(f) = common-mode inner-rail baseline jitter (consensus); scale(f) trusted from the
                 outer rail where the zoom-gate believed it, else FILLED from the field (horses hold
                 their lane -> lane/perp is a per-horse scale; median over horses, clamped to a tight
                 band since zoom is near-constant). Absolute level anchored by the 22.5 m outer sep.
      LONGITUD.  A_i(f) = s_i(f)*scale(f) + C(f).  C(f) = per-frame origin (polyline-origin shift +
                 leader gauge) from the consensus of ALL tracks + a smooth odometer front re-pin.
                 Absolute level anchored by the odometer (leader metres-run).

    S/P are (n_track, n_frame) px matrices (NaN = absent/invalid); odo/scale0/trust are per-frame.
    Returns (A, off, ylat, info) where A=absolute arclen, off=A-odo (position behind leader, <=0),
    ylat>=0 metres, info has the b/scale/C nuisance tracks.
    """
    nt, nfr = S.shape
    L = rail_w / sep_global                        # global scale level (m/px), near-constant

    # ---- LATERAL: remove common-mode rail baseline b(f); scale = trusted-outer / field-filled -----
    b = np.zeros(nfr)
    scale = scale0.copy()
    for _ in range(n_iter):
        lane_m = np.vstack([_nanmedfilt((P[i] - b) * scale, 15) for i in range(nt)])
        lane_px = lane_m / scale[None, :]
        b = b + _consensus(P - (lane_px + b[None, :]), min_cons)
        b = b - np.nanmedian(b[np.isfinite(b)])    # gauge: rail baseline mean = 0
        denom = P - b[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            sc_field = _consensus(np.where(np.abs(denom) > 3.0, lane_m / denom, np.nan), min_cons)
        sc_field = np.clip(sc_field, L * (1 - scale_tol), L * (1 + scale_tol))
        scale = np.where(trust, scale0, sc_field)
        bad = ~np.isfinite(scale)
        if bad.any():
            scale[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], scale[~bad])
    ylat = (P - b[None, :]) * scale[None, :]
    ylat[ylat > ylat_max] = np.nan
    ylat = np.clip(ylat, 0.0, None)

    # ---- LONGITUDINAL: consensus origin C(f) + smooth odometer front anchor -----------------------
    g = S * scale[None, :]
    p96g = np.array([np.nanpercentile(g[:, k], 96) if np.isfinite(g[:, k]).any() else np.nan
                     for k in range(nfr)])
    C = odo - p96g
    fin = np.isfinite(C)
    C = np.interp(np.arange(nfr), np.where(fin)[0], C[fin]) if fin.any() else np.zeros(nfr)
    for _ in range(n_iter):
        A = g + C[None, :]
        Abar = np.vstack([_nanmedfilt(A[i], 9) for i in range(nt)])
        C = _consensus(Abar - g, min_cons)
        front = np.array([np.nanpercentile(Abar[:, k], 96) if np.isfinite(Abar[:, k]).any()
                          else np.nan for k in range(nfr)])
        corr = _nanmedfilt(odo - front, 31)
        cf = np.isfinite(corr)
        corr = np.interp(np.arange(nfr), np.where(cf)[0], corr[cf]) if cf.any() else np.zeros(nfr)
        C = C + corr
    A = g + C[None, :]
    off = A - odo[None, :]

    # ---- SAFE per-track robust cleanup: NaN non-physical excursions (association swaps / glitches);
    # never REASSIGN identity. Lateral < ~2 m/s -> >4 m over ~1.2 s is a swap; longitudinal
    # differential > 8 m over 1.2 s is impossible.
    def _lane_clean(M, tol, w=31):
        for i in range(nt):
            tr = _nanmedfilt(M[i], w)
            M[i][np.isfinite(M[i]) & (np.abs(M[i] - tr) > tol)] = np.nan
    _lane_clean(ylat, 4.0)
    _lane_clean(off, 8.0)
    A = off + odo[None, :]
    return A, off, ylat, {"b": b, "scale": scale, "C": C}


def main() -> None:
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars
    from hkjc.vision.feeds.aerial_rail import line_candidates, select_running_rail
    from hkjc.vision.geometry_cv import project_to_polyline
    os.makedirs(DIAG, exist_ok=True)

    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)
    geom = build_rail_geometry("HV", COURSE)

    def is_bend(f):
        d_from_finish = DIST - float(np.interp(f, FR[fin], ODO[fin]))
        return np.isfinite(local_radius(geom, d_from_finish))

    # ---- PASS 1: per-frame detect + rail + project + lane-bound -------------------------------
    # per frame store: dict(frame -> list of (cx, cy, s, perp)) + the rail curve + the outer rail.
    # Cached to PASS1_CACHE (the 12-min decode) so the association/recon (PASS 2+) iterate instantly.
    per_frame = {}
    curves = {}
    outers = {}          # the OTHER main rail (outer) per frame -> local 22.5 m metric ruler
    if os.path.exists(PASS1_CACHE):
        c = np.load(PASS1_CACHE, allow_pickle=True)
        keys = list(c["keys"])
        per_frame = {k: [tuple(r) for r in v] for k, v in zip(keys, c["per_frame"], strict=True)}
        curves = {k: np.asarray(v, float) for k, v in zip(keys, c["curves"], strict=True)}
        outers = {k: (None if v is None or len(v) == 0 else np.asarray(v, float))
                  for k, v in zip(keys, c["outers"], strict=True)}
    else:
        prev_tang = None     # carried travel tangent -> keeps the perp SIGN consistent across frames
        prev_curve = None    # carried rail -> temporal continuity disambiguates equidistant-field frames
        container = av.open(AERIAL_VIDEO)
        for i, fr in enumerate(container.decode(video=0)):
            if i % STRIDE:
                continue
            img = fr.to_ndarray(format="bgr24")
            panel = img[:, HALF:]
            if not render_active(panel):     # skip the live-broadcast intro (find_layout is unreliable)
                continue
            ph = panel.shape[0]
            ty0, ty1 = track_band(panel)
            dets, _dbg = detect_avatars(panel, ty0, ty1, field=FIELD)
            if len(dets) < 3:
                continue
            # RUNNING-RAIL selection: of the two main rails pick the one CLOSEST (median dist) to the
            # field. detect_edgefill blobs come from track_band (green turf), NOT the rail -> non-circular;
            # the median is robust to scattered background false-positives. The returned curve is
            # tangent-oriented so project_to_polyline's perp SIGN is stable across frames.
            rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
            cands = line_candidates(panel, rb0, rb1)
            field_pts = np.array([[d.cx, d.cy] for d in dets], float)
            curve, prev_tang, _info = select_running_rail(
                cands, deg=(2 if is_bend(i) else 1), field_pts=field_pts,
                prev_tang=prev_tang, prev_curve=prev_curve)
            if curve is None:
                continue
            prev_curve = curve
            outers[i] = (_info or {}).get("outer")
            proj = [(d.cx, d.cy, *project_to_polyline(curve, d.cx, d.cy)) for d in dets]
            perps = np.array([p[3] for p in proj])
            side = _ry_side(perps)          # the field's running side of the rail (per-frame median sign)
            on_side = np.array([p[3] * side for p in proj])   # signed onto the running side (+ = on track)
            lane = LANE_K * float(np.percentile(np.abs(perps), 98))
            # store y_lat as a POSITIVE distance from the rail (the infield side has no runners, so a
            # signed lateral is unphysical). Keep only running-side blobs within the lane (small rail
            # overlap allowed), clamped to >= 0.
            keep = [(p[0], p[1], p[2], max(0.0, o)) for p, o in zip(proj, on_side) if -8.0 <= o <= lane]
            per_frame[i] = keep
            curves[i] = curve
        container.close()
        keys = sorted(per_frame)
        np.savez(PASS1_CACHE, keys=np.array(keys),
                 per_frame=np.array([np.array(per_frame[k], float) for k in keys], dtype=object),
                 curves=np.array([curves[k] for k in keys], dtype=object),
                 outers=np.array([outers.get(k) if outers.get(k) is not None else np.empty((0, 2))
                                  for k in keys], dtype=object))

    frames = sorted(per_frame)
    print(f"aerial-active processed: {len(frames)} frames")
    in_counts = [len(per_frame[f]) for f in frames]
    print(f"in-lane blobs/frame: median {int(np.median(in_counts))}, "
          f"min {min(in_counts)}, max {max(in_counts)}; "
          f"frames with exactly {FIELD}: {sum(c == FIELD for c in in_counts)}")

    # ---- metric anchor (shared by association + reconstruction): odo + head-relative offset -----
    def odo_arclen(f):
        return float(np.interp(f, FR[fin], ODO[fin]))   # HEAD horse metres-run (1800 - M-to-finish)

    # robust head reference: the 96th percentile of s, NOT max(s) (a single spurious lead detection
    # makes max jitter and snaps the longitudinal baseline). p96 ~ the leader but outlier-resistant.
    s_head = {f: float(np.percentile([p[2] for p in per_frame[f]], 96)) for f in frames}

    # ASSOCIATION uses a STABLE global scale (a varying scale would jitter the matching coordinate and
    # shatter the tracks). metric() below feeds PASS 2 only.
    spreads = [max(p[2] for p in per_frame[f]) - min(p[2] for p in per_frame[f])
               for f in frames if len(per_frame[f]) > 2]
    scale_assoc = 15.0 / max(1.0, float(np.median(spreads)))

    def metric(f, s, perp):
        return odo_arclen(f) - (s_head[f] - s) * scale_assoc, perp * scale_assoc

    # OUTPUT uses a PERSPECTIVE-CORRECT per-frame scale: the inner->outer rail separation is a KNOWN
    # 22.5 m, so local px->m varies with foreshortening. Temporally SMOOTHED (camera distance changes
    # slowly; the raw per-frame sep is noisy because the outer rail is the fainter one).
    RAIL_W = 22.5

    def _sep_px(f):
        inner, outer = curves[f], outers.get(f)
        if outer is None or len(outer) == 0:
            return None
        d2 = (inner[:, None, 0] - outer[None, :, 0]) ** 2 + (inner[:, None, 1] - outer[None, :, 1]) ** 2
        return float(np.median(np.sqrt(d2.min(1))))

    sep_raw = np.array([(lambda s: s if s else np.nan)(_sep_px(f)) for f in frames])
    sep_global = float(np.nanmedian(sep_raw)) if np.isfinite(sep_raw).any() else 200.0
    sep_sm, sep_trust = _robust_zoom_track(sep_raw, return_trust=True)   # + per-frame outer-trusted mask
    sep_by_f = {f: float(sep_sm[k]) for k, f in enumerate(frames)}
    trust_by_f = {f: bool(sep_trust[k]) for k, f in enumerate(frames)}

    def scale_f(f):
        return RAIL_W / max(20.0, sep_by_f.get(f, sep_global))

    # LONGITUDINAL trust: the rail polyline's arc-length L_px swings wildly (the detected span collapses
    # to ~140 px or jumps ~700 px frame-to-frame) and s_px CLAMPS to its ends -> the head reference and
    # every gap step. So gate the longitudinal where the rail is too short to span the field or a point
    # clamps to an endpoint (journal 0105: leader front-clamped 18% of frames, back-clamped 11%).
    def _poly_len(f):
        cv = curves[f]
        return float(np.sum(np.hypot(np.diff(cv[:, 0]), np.diff(cv[:, 1]))))

    len_by_f = {f: _poly_len(f) for f in frames}
    len_med = float(np.median(list(len_by_f.values())))
    CLAMP_PX, SHORT_FRAC = 25.0, 0.6   # a point clamped within CLAMP_PX of a polyline end has a bad s

    # ---- PASS 2: METRIC-SPACE association with the ODOMETER as the motion prior ----------------
    # Match detections in (arclen, y_lat) METRES. Each track is predicted forward by the odometer
    # delta (a horse advances with the field; its residual drift between stride-4 frames is small), so
    # association survives bends, zoom and short holes far better than panel-pixel NN. Holes bridged
    # up to GRACE frames; gates are in metres.
    GATE_ARC, GATE_LAT, GRACE = 6.0, 4.0, 24   # gates in metres; GRACE ~1 s (stride-independent)
    tracks = {}            # tid -> list of (frame, cx, cy, s, perp)
    active = {}            # tid -> dict(last_f, arclen, ylat)
    next_tid = 0
    for f in frames:
        dets = [(cx, cy, s, perp, *metric(f, s, perp)) for (cx, cy, s, perp) in per_frame[f]]
        if active:
            tids = list(active)
            pred = {t: (active[t]["arclen"] + (odo_arclen(f) - odo_arclen(active[t]["last_f"])),
                        active[t]["ylat"]) for t in tids}
            cand = []
            for di, d in enumerate(dets):
                for t in tids:
                    da, dl = abs(d[4] - pred[t][0]), abs(d[5] - pred[t][1])
                    if da < GATE_ARC and dl < GATE_LAT:
                        cand.append((da / GATE_ARC + dl / GATE_LAT, di, t))
            cand.sort()
            used, matched = set(), set()
            for _cost, di, t in cand:
                if di in matched or t in used:
                    continue
                d = dets[di]
                tracks[t].append((f, d[0], d[1], d[2], d[3]))
                active[t] = dict(last_f=f, arclen=d[4], ylat=d[5])
                used.add(t)
                matched.add(di)
            for t in tids:                         # bridge holes; drop only after the grace gap
                if t not in used and active[t]["last_f"] < f - GRACE:
                    active.pop(t, None)
            dets = [d for di, d in enumerate(dets) if di not in matched]
        for d in dets:                             # unmatched -> new track
            tracks[next_tid] = [(f, d[0], d[1], d[2], d[3])]
            active[next_tid] = dict(last_f=f, arclen=d[4], ylat=d[5])
            next_tid += 1

    # ---- FRAGMENT RE-LINK: a horse whose detection drops out for > GRACE restarts as a new tid. Re-join
    # B after A when their ODOMETER-PREDICTED (arclen, y_lat) trajectories are continuous across the gap.
    # STRICT gates + one-successor/one-predecessor greedy-by-cost => no ambiguous fusion of two horses
    # (the user's mis-ID caution). Predicting A forward by the odometer delta makes long holes linkable.
    LINK_GAP, LINK_ARC, LINK_LAT = 250, 6.0, 3.0    # max hole (frames ~10 s); metric gates (m)

    def _endpt(pts, k=5):
        def mp(seg):
            ms = [metric(f, s, perp) for (f, _cx, _cy, s, perp) in seg]
            return float(np.median([m[0] for m in ms])), float(np.median([m[1] for m in ms]))
        return mp(pts[:k]), mp(pts[-k:])

    relinkable = {t: v for t, v in tracks.items() if len(v) >= 12}
    ends = {t: (v[0][0], v[-1][0], *_endpt(v)) for t, v in relinkable.items()}  # t->(sf,ef,startm,endm)
    links = []
    for a, (_saf, eaf, _sam, eam) in ends.items():
        for b, (sbf, _ebf, sbm, _ebm) in ends.items():
            gap = sbf - eaf
            if a == b or not (0 < gap <= LINK_GAP):
                continue
            pred_arc = eam[0] + (odo_arclen(sbf) - odo_arclen(eaf))   # A advances with the odometer
            da, dl = abs(sbm[0] - pred_arc), abs(sbm[1] - eam[1])
            if da <= LINK_ARC and dl <= LINK_LAT:
                links.append((da / LINK_ARC + dl / LINK_LAT, a, b))
    links.sort()
    parent = {t: t for t in tracks}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    succ_used, pred_used, n_link = set(), set(), 0
    for _cost, a, b in links:                       # greedy by cost; each end links at most once
        if a in succ_used or b in pred_used or _find(a) == _find(b):
            continue
        parent[_find(b)] = _find(a)
        succ_used.add(a)
        pred_used.add(b)
        n_link += 1
    merged = {}
    for t, v in tracks.items():
        merged.setdefault(_find(t), []).extend(v)
    tracks = {t: sorted(v) for t, v in merged.items()}
    print(f"re-link: {n_link} joins -> {len(tracks)} tracklets (from {len(ends)} relinkable fragments)")

    # DUMP the substantial-track matrix + per-frame nuisances (DUMP_TRACKS=1) for offline estimator work.
    if os.environ.get("DUMP_TRACKS", "0") == "1":
        sub = [v for v in tracks.values() if len(v) >= 60]
        np.savez(
            f"data_cache/cv_tracks_{RACE}.npz",
            frames=np.array(frames),
            odo=np.array([odo_arclen(f) for f in frames]),
            sep=np.array([sep_by_f[f] for f in frames]),
            trust=np.array([trust_by_f[f] for f in frames]),
            plen=np.array([len_by_f[f] for f in frames]),
            len_med=len_med, sep_global=sep_global, rail_w=RAIL_W, dist=DIST,
            tf=np.array([np.array([p[0] for p in v]) for v in sub], dtype=object),
            ts=np.array([np.array([p[3] for p in v]) for v in sub], dtype=object),
            tp=np.array([np.array([p[4] for p in v]) for v in sub], dtype=object),
        )
        print(f"[dump] data_cache/cv_tracks_{RACE}.npz  ({len(sub)} substantial tracks)")

    lengths = sorted((len(v) for v in tracks.values()), reverse=True)
    long_tracks = [v for v in tracks.values() if len(v) >= 5]
    span = max(frames) - min(frames)
    longest = lengths[0] * STRIDE
    print(f"tracklets: {len(tracks)} total; {len(long_tracks)} with >=5 pts; top lengths {lengths[:FIELD]}")
    print(f"longest tracklet spans {longest} frames of {span} (rail-gated: holes where the rail fit "
          f"fails still fragment the global path; perp now sign-consistent across bends)")

    # ---- OVERLAYS (anonymous: rail + per-frame s-rank front-to-back order) --------------------
    def draw_overlay(f, panel):
        vis = panel.copy()
        curve = curves[f]
        for t in range(1, len(curve)):
            cv2.line(vis, (int(curve[t - 1, 0]), int(curve[t - 1, 1])),
                     (int(curve[t, 0]), int(curve[t, 1])), (0, 255, 0), 2)
        blobs = per_frame[f]
        rank = {k: r for r, k in enumerate(sorted(range(len(blobs)), key=lambda k: blobs[k][2]))}
        for k, (cx, cy, s, perp) in enumerate(blobs):
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis, str(rank[k] + 1), (int(cx) - 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        cv2.putText(vis, f"f{f}: GREEN rail; number = s-rank (front-to-back along rail); "
                    f"{len(blobs)} in-lane", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis

    save_set = {f for f in SAVE_FRAMES if f is not None} | {frames[0]}
    saved_panels = {}    # overlay panels are re-decoded here (PASS 1 no longer keeps frames in memory)
    if os.environ.get("OVERLAYS", "1") == "1":   # OVERLAYS=0 skips the 10-min re-decode while iterating
        for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
            if i in save_set and saved_panels.get(i) is None:
                saved_panels[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if all(saved_panels.get(s) is not None for s in save_set):
                break
        for f in sorted(save_set):
            if f not in per_frame or saved_panels.get(f) is None:
                continue
            out = f"{DIAG}/cv_track_id_f{f:04d}{SFX}.png"
            cv2.imwrite(out, draw_overlay(f, saved_panels[f]))
            print(f"[overlay] {out}")

    # anonymous tracklet perp-vs-frame plot (each line = one anonymous runner segment)
    fig, ax = plt.subplots(figsize=(12, 5))
    for pts in tracks.values():
        if len(pts) < 5:
            continue
        ax.plot([p[0] for p in pts], [p[4] for p in pts], "-", lw=1.0, alpha=0.8)
    ax.set_xlabel("aerial frame")
    ax.set_ylabel("perp from rail (px)  [lateral trip]")
    ax.set_title("Anonymous tracklets (|perp| vs frame) — field-median running rail, positive "
                 "lateral distance (px scale not yet normalised)")
    fig.savefig(f"{DIAG}/cv_track_id_tracklets{SFX}.png", dpi=110, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_track_id_tracklets{SFX}.png")

    # ---- PASS 3: FIELD-CONSENSUS FACTORISATION reconstruction -> metric 2D paths -----------------
    # Build the (track x frame) px matrices for the SUBSTANTIAL tracklets, gate the longitudinal s where
    # a point clamps to a polyline end (corrupt arc-length), then let field_consensus_recon infer
    # b(f)/scale(f)/C(f) from the field. Replaces the old per-frame _recon (single p96 leader + single
    # outer line) with the N-horse consensus estimator.
    MIN_TRACK = 60
    S_MIN, YLAT_MAX = 3.0, 25.0
    sub = [v for v in sorted(tracks.values(), key=len, reverse=True) if len(v) >= MIN_TRACK]
    fcol = {f: k for k, f in enumerate(frames)}
    nt, nfr = len(sub), len(frames)
    Smat = np.full((nt, nfr), np.nan)
    Pmat = np.full((nt, nfr), np.nan)
    for i, pts in enumerate(sub):
        for (f, _cx, _cy, s, perp) in pts:
            k = fcol[f]
            lp = len_by_f.get(f, len_med)
            if not (s < S_MIN or s > lp - CLAMP_PX or lp < SHORT_FRAC * len_med):
                Smat[i, k] = s
            Pmat[i, k] = perp
    odo_arr = np.array([odo_arclen(f) for f in frames])
    scale0_arr = np.array([scale_f(f) for f in frames])
    trust_arr = np.array([trust_by_f[f] for f in frames])
    A_mat, off_mat, ylat_mat, info = field_consensus_recon(
        Smat, Pmat, odo_arr, scale0_arr, trust_arr, RAIL_W, sep_global, ylat_max=YLAT_MAX)
    n_paths = nt

    fig2, (axA, axL, axR) = plt.subplots(1, 3, figsize=(24, 7))
    Ff = np.array(frames)
    for i in range(nt):
        mA = np.isfinite(A_mat[i])
        axA.plot(Ff[mA], A_mat[i][mA], "-", lw=1.0, alpha=0.7)
        off = _medfilt(off_mat[i], w=5)
        mO = np.isfinite(off)
        axL.plot(Ff[mO], off[mO], "-", lw=1.0, alpha=0.7)
        yl = _medfilt(ylat_mat[i], w=5)
        mY = np.isfinite(yl) & mA
        axR.plot(A_mat[i][mY], yl[mY], "-", lw=1.0, alpha=0.7)
    axA.set_xlabel("aerial frame")
    axA.set_ylabel("arclen run (m)")
    axA.set_title("Absolute arclen (odometer-anchored, monotonic)")
    axL.set_xlabel("aerial frame")
    axL.set_ylabel("longitudinal offset in field (m)")
    axL.set_title("Per-horse position behind the head (0 = leader) — the race dynamics")
    axL.axhline(0, color="k", lw=0.5)
    axR.set_xlabel("arclen run (m)")
    axR.set_ylabel("y_lat (m)")
    axR.set_title("Reconstructed 2D path: lateral trip vs arclen (field-consensus)")
    fig2.savefig(f"{DIAG}/cv_recon_2d{SFX}.png", dpi=145, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_recon_2d{SFX}.png  ({n_paths} paths, scale~={RAIL_W/sep_global:.3f} m/px; "
          f"b|max {np.nanmax(np.abs(info['b'])):.0f}px, scale-filled {(~trust_arr).mean()*100:.0f}%)")

    # ---- PASS 4: RESOLUTION diagnostic — top tracks' raw perp & s (px) + worst-jump frames -------
    top = sorted(tracks.values(), key=len, reverse=True)[:3]
    fig3, (axP, axS) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    worst = []
    for ti, pts in enumerate(top):
        ff = [p[0] for p in pts]
        pr = [p[4] for p in pts]                       # raw perp (px)
        so = [p[3] - s_head[p[0]] for p in pts]        # raw s offset behind head (px)
        axP.plot(ff, pr, "-o", ms=2, lw=0.8, label=f"track{ti} (n={len(pts)})")
        axS.plot(ff, so, "-o", ms=2, lw=0.8)
        for k in range(1, len(pts)):
            dfr = ff[k] - ff[k - 1]
            if dfr <= 2 * STRIDE and (abs(pr[k] - pr[k - 1]) > 25 or abs(so[k] - so[k - 1]) > 40):
                worst.append((ff[k], ti, round(pr[k] - pr[k - 1]), round(so[k] - so[k - 1])))
    axP.set_ylabel("perp (px) [lateral]")
    axP.legend(fontsize=8)
    axP.set_title("Top-3 tracks RAW resolution — perp (px) and s-offset (px) vs frame (jitter/quant?)")
    axS.set_ylabel("s offset behind head (px) [longitudinal]")
    axS.set_xlabel("aerial frame")
    fig3.savefig(f"{DIAG}/cv_recon_res{SFX}.png", dpi=110, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_recon_res{SFX}.png")
    worst.sort(key=lambda w: -abs(w[2]) - abs(w[3]))
    print(f"worst per-frame jumps (frame, track, dperp_px, ds_px): {worst[:12]}")


if __name__ == "__main__":
    main()

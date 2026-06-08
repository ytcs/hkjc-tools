"""DRONE rail by RAIL-SPACE temporal tracking (the hybrid; data-chosen over pose-space homography).

The honest data->model residual killed the global-homography tracker (median 68-171px, scale drift
7x): a smooth arc under-constrains an 8-DOF homography -> gauge drift. Pixel detection, by contrast,
puts the rail exactly on the rail. So: track each RAIL directly.

Per frame: detect rail pixels -> assign to inner/outer by temporal proximity to the previously
tracked rails (consistency, no gauge freedom) -> fit each rail to ITS pixels (pixel-accurate) ->
fill an OCCLUDED rail from the visible one + the known track width (22.5 m -> px via measured
separation). No global pose, no smoothing lag on visible rails.

Success metric = data->model residual (each detected pixel to the nearest output rail): target
median < 3px, p90 < 10px, across ALL frames incl. the dark bend.
"""
from __future__ import annotations

import importlib.util

import av
import cv2
import numpy as np
from scipy.spatial import cKDTree

_spec = importlib.util.spec_from_file_location("cdr", "studies/cv_drone_rail.py")
cdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdr)

VIDEO = "data_cache/diag/drone_scan/drone_R08_half.mp4"
WIDTH_M = 22.5


def curve(model, n=120):
    return model["sample"](n)


def cdist(a, b):
    return float(np.median(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min(1)))


def fit_pixels(xs, ys):
    return cdr.fit_model(np.asarray(xs), np.asarray(ys))


def _sep_stats(a, b):
    """Median separation + its coefficient of variation between two sampled curves (px). Two rails
    stay ~constant width apart along their length; a rail+distractor pair diverges."""
    pa, pb = a["sample"](40), b["sample"](40)
    d = np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1)).min(1)
    m = float(np.median(d))
    return m, float(np.std(d) / (m + 1e-6))


def select_pair(cands):
    """PREV-INDEPENDENT: the longest pair of candidates with near-CONSTANT separation in a plausible
    range (= the two running rails). Returns (cand_a, cand_b) or (cand, None) or (None, None)."""
    cc = sorted(cands, key=lambda c: -c["len"])[:6]
    lo, hi = 40 * cdr.SCALE, 420 * cdr.SCALE
    best = None
    for i in range(len(cc)):
        for j in range(i + 1, len(cc)):
            sep, cv = _sep_stats(cc[i]["model"], cc[j]["model"])
            if lo < sep < hi and cv < 0.40:
                score = cc[i]["len"] + cc[j]["len"]
                if best is None or score > best[0]:
                    best = (score, cc[i], cc[j])
    if best:
        return best[1], best[2]
    return (cc[0] if cc else None), None


def _green_both_sides(model, band):
    """Inner rail has infield green on BOTH perpendicular sides; outer rail has clutter (non-green)
    on its far side. Returns min green-fraction over the two sides (low => outer)."""
    pts = model["sample"](20)
    H, W = band.shape
    d = np.gradient(pts, axis=0)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
    fr = []
    for s in (+1, -1):
        pr = (pts + s * nrm * 22 * cdr.SCALE).astype(int)
        ok = (pr[:, 0] >= 0) & (pr[:, 0] < W) & (pr[:, 1] >= 0) & (pr[:, 1] < H)
        fr.append(float(np.mean(band[pr[ok, 1], pr[ok, 0]] > 0)) if ok.any() else 0.0)
    return min(fr)


def rail_role(model, band):
    """PER-FRAME ABSOLUTE label: inner rail has infield green on BOTH sides; outer has clutter on its
    far side. Returns ('inner'|'outer', min_green_fraction)."""
    g = _green_both_sides(model, band)
    return ("inner" if g > 0.5 else "outer"), g


def label_pair(ma, mb, prev_in, prev_out, band):
    """Assign the two rail models to inner/outer by the green-side test ALONE (no temporal chaining,
    so a label swap cannot propagate); inner = the rail that is greener on both sides."""
    ga, gb = _green_both_sides(ma, band), _green_both_sides(mb, band)
    return (ma, mb) if ga >= gb else (mb, ma)


def shift_toward(model, width_px, ref):
    """Make the partner rail: offset `model` perpendicular/radially by width_px toward `ref`."""
    base = model["sample"](120)
    if model["type"] == "arc":
        ctr = model["center"]
        rad = np.linalg.norm(base - ctr, axis=1, keepdims=True)
        ur = (base - ctr) / (rad + 1e-9)
        cand = [base + ur * width_px, base - ur * width_px]
    else:
        d = base[-1] - base[0]
        perp = cdr._rot90(d / (np.linalg.norm(d) + 1e-9))
        cand = [base + perp * width_px, base - perp * width_px]
    rp = ref["sample"](40)
    best = min(cand, key=lambda p: cdist(p, rp))
    return {"type": model["type"], "synth": True, "sample": lambda nn=120, _p=best: _p}


def band_outer_curve(band, n=80):
    """The band's outer boundary (green->road/stands) = the OUTER rail, as a fitted curve. Kept
    SEPARATE from the candidate pool (used only as an outer fallback) so it can't pollute the
    high-quality interior-white inner detection."""
    import cv2 as _cv
    bw = (band > 0).astype(np.uint8)
    edge = bw - _cv.erode(bw, np.ones((5, 5), np.uint8))
    m = int(round(10 * cdr.SCALE)) + 1
    edge[:m] = 0
    edge[-m:] = 0
    edge[:, :m] = 0
    edge[:, -m:] = 0
    nn, lab, stats, _ = _cv.connectedComponentsWithStats(edge, 8)
    best = None
    for k in range(1, nn):
        ys, xs = np.where(lab == k)
        if len(xs) < 80 * cdr.SCALE:
            continue
        _, _, ratio, ext, w = cdr._pca(ys, xs)
        if ratio > 3 and ext > 100 * cdr.SCALE and w < 18 * cdr.SCALE:
            if best is None or ext > best[0]:
                best = (ext, cdr.fit_model(xs, ys))
    return best[1]["sample"](n) if best else None


def detect(bgr):
    cands, band, _ = cdr.rail_candidates(bgr)
    if len(cands) < 2:
        c2, _, _ = cdr.rail_candidates(bgr, band=band, tophat=True)
        if len(c2) > len(cands):
            cands = c2
    for c in cands:
        c["model"] = fit_pixels(c["xs"], c["ys"])
    det = np.vstack([np.c_[c["xs"], c["ys"]] for c in cands]) if cands else None
    return cands, band, det


def run(t0=48.0, t1=74.0, stride=3, strip="data_cache/diag/drone_scan/04_rail_track.png"):
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    inner = outer = None
    width = None
    i = -1
    tiles = []
    res_all = []
    loo = []
    cov = {"both": 0, "fill-in": 0, "fill-out": 0, "hold": 0, "seed": 0}
    for fr in cont.decode(st):
        i += 1
        t = i / 25.0
        if t > t1:
            break
        if i % stride or t < t0:
            continue
        bgr = fr.to_ndarray(format="bgr24")
        cands, band, det = detect(bgr)
        kind = "hold"
        sel_px = None
        if band is not None and cands:
            ca, cb = select_pair(cands)          # PREV-INDEPENDENT rail finding
            if ca is not None and cb is not None:
                mi, mo = label_pair(ca["model"], cb["model"], inner, outer, band)
                inner, outer, kind = mi, mo, "both"
                sel_px = np.vstack([np.c_[ca["xs"], ca["ys"]], np.c_[cb["xs"], cb["ys"]]])
                w = cdist(mi["sample"](60), mo["sample"](60))
                # leave-one-out fill accuracy: predict each rail from the OTHER + width, vs truth
                if width is not None:
                    loo.append(cdist(shift_toward(mo, width, mi)["sample"](60), mi["sample"](60)))
                    loo.append(cdist(shift_toward(mi, width, mo)["sample"](60), mo["sample"](60)))
                width = w if width is None else 0.85 * width + 0.15 * w
            elif ca is not None and inner is not None and width is not None:
                # one rail only -> identify it by prev proximity, fill the partner from tracked width
                m = ca["model"]
                if cdist(m["sample"](40), inner["sample"](40)) <= cdist(m["sample"](40), outer["sample"](40)):
                    inner, outer, kind = m, shift_toward(m, width, outer), "fill-out"
                else:
                    outer, inner, kind = m, shift_toward(m, width, inner), "fill-in"
                sel_px = np.c_[ca["xs"], ca["ys"]]
            else:
                kind = "seedwait"
        cov[kind] = cov.get(kind, 0) + 1
        # HONEST residual on the SELECTED rail pixels only (distractors excluded): how well the
        # output rails hug the pixels we committed to.
        med = p90 = float("nan")
        if sel_px is not None and inner is not None:
            rail = np.vstack([inner["sample"](150), outer["sample"](150)])
            d, _ = cKDTree(rail).query(sel_px)
            med, p90 = float(np.median(d)), float(np.percentile(d, 90))
            res_all.append(med)
        if i % 24 == 0 and inner is not None:
            ov = (bgr // 2).copy()
            if det is not None:
                ov[det[:, 1], det[:, 0]] = (0, 140, 255)
            cv2.polylines(ov, [outer["sample"](120).astype(np.int32)], False, (255, 0, 255), 2)
            cv2.polylines(ov, [inner["sample"](120).astype(np.int32)], False, (255, 255, 0), 2)
            cv2.putText(ov, f"t={t:.1f} {kind} resid med={med:.1f} p90={p90:.1f}px", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(cv2.resize(ov, (560, 315)))
    if tiles:
        cols = 4
        rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
                for r in range(0, len(tiles), cols)]
        cv2.imwrite(strip, np.vstack(rows))
    ra = np.array(res_all)
    la = np.array(loo) if loo else np.array([float("nan")])
    print(f"window {t0}-{t1}: frames {sum(cov.values())}  coverage {cov}")
    print(f"  resid(selected px) med {np.median(ra):.1f}px  p90 {np.percentile(ra, 90):.1f}  worst {ra.max():.1f}")
    print(f"  LOO fill error    med {np.median(la):.1f}px  p90 {np.percentile(la, 90):.1f}  worst {la.max():.1f}  (n={len(loo)})")


def _orient(pts, ref):
    if ref is None:
        return pts
    if cdist(pts, ref) > cdist(pts[::-1], ref):
        return pts[::-1]
    return pts


def _clean_series(series, real_idx, thresh, iters=3, bracket_gap=4):
    """Reject outlier detections (the detector locking onto the wrong line) by CLOSE-BRACKET temporal
    consistency: only judge a frame when it has real neighbours within `bracket_gap` on BOTH sides
    (so the interpolated prediction is reliable) — a frame that jumps >> thresh from close neighbours
    is a bad detection. Frames with only far brackets are NOT judged (avoids penalising legitimate
    fast-pan over long gaps). Outliers -> set None; returns the cleaned real-index list."""
    real = list(real_idx)
    for _ in range(iters):
        bad = set()
        for k in real:
            bef = [j for j in real if j < k]
            aft = [j for j in real if j > k]
            if bef and aft and (aft[0] - bef[-1]) <= bracket_gap:
                a, b = bef[-1], aft[0]
                f = (k - a) / (b - a)
                if cdist((1 - f) * series[a] + f * series[b], series[k]) > thresh:
                    bad.add(k)
        if not bad:
            break
        real = [k for k in real if k not in bad]
    for k in list(range(len(series))):
        if series[k] is not None and k not in set(real):
            series[k] = None
    return real


def _fill_series(series, real_idx):
    """Fill None gaps by linear temporal interpolation between bracketing REAL frames (motion is
    ~linear over the brief occlusion); hold nearest at the ends."""
    out = list(series)
    for k in range(len(out)):
        if series[k] is not None:
            continue
        bef = [j for j in real_idx if j < k]
        aft = [j for j in real_idx if j > k]
        if bef and aft:
            a, b = bef[-1], aft[0]
            f = (k - a) / (b - a)
            out[k] = (1 - f) * series[a] + f * series[b]
        elif bef:
            out[k] = series[bef[-1]]
        elif aft:
            out[k] = series[aft[0]]
    return out


CACHE = "data_cache/diag/drone_scan/_rec_pass1.pkl"


def pass1(t0=0.0, t1=113.0, stride=1, video=VIDEO, cache=CACHE):
    """Expensive detection pass (decode + detect + label per frame) -> cache to disk so the cheap
    pass-2 fill experiments run in seconds instead of a 6-min re-decode."""
    import pickle
    cont = av.open(video)
    st = cont.streams.video[0]
    N = 80
    i = -1
    rec = []
    last_in = last_out = None
    for fr in cont.decode(st):
        i += 1
        t = i / 25.0
        if t > t1:
            break
        if i % stride or t < t0:
            continue
        bgr = fr.to_ndarray(format="bgr24")
        cands, band, _ = detect(bgr)
        in_pts = out_pts = sel_px = None
        if band is not None and cands:
            ca, cb = select_pair(cands)
            if ca is not None and cb is not None:
                mi, mo = label_pair(ca["model"], cb["model"], None, None, band)
                in_pts = _orient(mi["sample"](N), last_in)
                out_pts = _orient(mo["sample"](N), last_out)
                sel_px = np.vstack([np.c_[ca["xs"], ca["ys"]], np.c_[cb["xs"], cb["ys"]]])
            elif ca is not None:
                role, _ = rail_role(ca["model"], band)
                if role == "inner":
                    in_pts = _orient(ca["model"]["sample"](N), last_in)
                else:
                    out_pts = _orient(ca["model"]["sample"](N), last_out)
                sel_px = np.c_[ca["xs"], ca["ys"]]
        outb = band_outer_curve(band, N) if band is not None else None  # separate outer fallback
        if in_pts is not None:
            last_in = in_pts
        if out_pts is not None:
            last_out = out_pts
        rec.append({"i": i, "t": t, "in": in_pts, "out": out_pts, "outb": outb, "sel": sel_px,
                    "thumb": cv2.resize(bgr, (560, 315)) if i % 96 == 0 else None})
    with open(cache, "wb") as f:
        pickle.dump(rec, f)
    print(f"pass1 cached {len(rec)} frames -> {cache}")
    return rec


def run_smooth(t0=0.0, t1=113.0, stride=3, strip="data_cache/diag/drone_scan/04_rail_track.png"):
    """LEGACY single-pass entry (kept for reference)."""
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    N = 80
    i = -1
    rec = []                       # per kept-frame record
    last_in = last_out = None
    for fr in cont.decode(st):
        i += 1
        t = i / 25.0
        if t > t1:
            break
        if i % stride or t < t0:
            continue
        bgr = fr.to_ndarray(format="bgr24")
        cands, band, _ = detect(bgr)
        in_pts = out_pts = sel_px = None
        if band is not None and cands:
            ca, cb = select_pair(cands)
            if ca is not None and cb is not None:
                mi, mo = label_pair(ca["model"], cb["model"], None, None, band)  # per-frame green-side
                in_pts = _orient(mi["sample"](N), last_in)
                out_pts = _orient(mo["sample"](N), last_out)
                sel_px = np.vstack([np.c_[ca["xs"], ca["ys"]], np.c_[cb["xs"], cb["ys"]]])
            elif ca is not None:
                role, _ = rail_role(ca["model"], band)              # per-frame absolute role
                if role == "inner":
                    in_pts = _orient(ca["model"]["sample"](N), last_in)
                else:
                    out_pts = _orient(ca["model"]["sample"](N), last_out)
                sel_px = np.c_[ca["xs"], ca["ys"]]
        if in_pts is not None:
            last_in = in_pts
        if out_pts is not None:
            last_out = out_pts
        rec.append({"i": i, "t": t, "in": in_pts, "out": out_pts, "sel": sel_px,
                    "bgr": bgr if i % 96 == 0 else None})
    # PASS 2: temporal fill per rail
    in_series = [r["in"] for r in rec]
    out_series = [r["out"] for r in rec]
    thr = 20.0 * cdr.SCALE
    in_real = _clean_series(in_series, [k for k, p in enumerate(in_series) if p is not None], thr)
    out_real = _clean_series(out_series, [k for k, p in enumerate(out_series) if p is not None], thr)
    in_full = _fill_series(in_series, in_real)
    out_full = _fill_series(out_series, out_real)
    # METRIC: LOO temporal fill error (drop a real rail, interpolate from its OTHER real frames)
    loo = []
    for ser, real in ((in_series, in_real), (out_series, out_real)):
        rset = set(real)
        for k in real:
            bef = [j for j in real if j < k and j in rset]
            aft = [j for j in real if j > k]
            if bef and aft:
                a, b = bef[-1], aft[0]
                f = (k - a) / (b - a)
                pred = (1 - f) * ser[a] + f * ser[b]
                loo.append(cdist(pred, ser[k]))
    res = []
    for k, r in enumerate(rec):
        if r["sel"] is not None and in_full[k] is not None:
            rail = np.vstack([in_full[k], out_full[k]])
            d, _ = cKDTree(rail).query(r["sel"])
            res.append(float(np.median(d)))
    # strip
    tiles = []
    for k, r in enumerate(rec):
        if r["bgr"] is None or in_full[k] is None:
            continue
        ov = (r["bgr"] // 2).copy()
        if r["sel"] is not None:
            ov[r["sel"][:, 1], r["sel"][:, 0]] = (0, 140, 255)
        cv2.polylines(ov, [out_full[k].astype(np.int32)], False, (255, 0, 255), 2)
        cv2.polylines(ov, [in_full[k].astype(np.int32)], False, (255, 255, 0), 2)
        kk = "both" if (k in set(in_real) and k in set(out_real)) else "fill"
        cv2.putText(ov, f"t={r['t']:.1f} {kk}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(cv2.resize(ov, (560, 315)))
    if tiles:
        cols = 4
        rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
                for r in range(0, len(tiles), cols)]
        cv2.imwrite(strip, np.vstack(rows))
    ra, la = np.array(res), np.array(loo)
    nb = len(set(in_real) & set(out_real))
    print(f"smooth {t0}-{t1}: frames {len(rec)}  both-real {nb} ({100*nb/len(rec):.0f}%)  filled {len(rec)-nb}")
    print(f"  resid(real px) med {np.median(ra):.1f}px p90 {np.percentile(ra,90):.1f}")
    print(f"  LOO temporal-fill err med {np.median(la):.1f}px p90 {np.percentile(la,90):.1f} worst {la.max():.1f} (n={len(la)})")


def _model_of(pts):
    if pts is None:
        return None
    return {"type": "line", "sample": lambda nn=80, _p=pts: _p}


def _shift_pts(pts, width, ref):
    """Offset a polyline perpendicular by width toward `ref` (geometry fill of the partner rail)."""
    d = np.gradient(pts, axis=0)
    perp = np.c_[-d[:, 1], d[:, 0]]
    perp /= (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-6)
    a, b = pts + perp * width, pts - perp * width
    return a if cdist(a, ref) < cdist(b, ref) else b


def _temporal_fill(series, real, max_gap):
    """Interpolate Nones from bracketing reals ONLY when the gap <= max_gap (short occlusion);
    longer gaps stay None for the geometry stage."""
    out = list(series)
    for k in range(len(out)):
        if series[k] is not None:
            continue
        bef = [j for j in real if j < k]
        aft = [j for j in real if j > k]
        if bef and aft and (aft[0] - bef[-1]) <= max_gap:
            a, b = bef[-1], aft[0]
            f = (k - a) / (b - a)
            out[k] = (1 - f) * series[a] + f * series[b]
    return out


def pass2(max_gap=24, thr=50.0, cache=CACHE, strip="data_cache/diag/drone_scan/04_rail_track_full.png"):
    """Fast pass-2 on the cached detections: outlier-reject -> HYBRID fill (temporal for short
    occlusions, geometry [other rail + width] for long gaps) -> metrics + strip."""
    import pickle
    with open(cache, "rb") as f:
        rec = pickle.load(f)
    n = len(rec)
    in_s = [r["in"] for r in rec]
    out_s = [r["out"] for r in rec]
    # OUTER fallback: where the white-outer is absent, use the band-boundary outer curve (a real
    # outer-rail observation) -> orient consistently; validated below against the track width.
    nfb = 0
    for k in range(n):
        if out_s[k] is None and rec[k].get("outb") is not None:
            out_s[k] = rec[k]["outb"]
            nfb += 1
    prev = None
    for k in range(n):
        if out_s[k] is not None:
            out_s[k] = _orient(out_s[k], prev)
            prev = out_s[k]
    # WIDTH-CONSISTENCY reject: a both-frame whose rail separation deviates far from the median track
    # width = the detector grabbed a wrong feature (far boundary / verge / wrong rail) -> drop both.
    med_w0 = np.median([cdist(in_s[k], out_s[k]) for k in range(n)
                        if in_s[k] is not None and out_s[k] is not None])
    nbad = 0
    for k in range(n):
        if in_s[k] is not None and out_s[k] is not None:
            if abs(cdist(in_s[k], out_s[k]) - med_w0) > 0.4 * med_w0:
                in_s[k] = out_s[k] = None
                nbad += 1
    in_real = _clean_series(in_s, [k for k, p in enumerate(in_s) if p is not None], thr)
    out_real = _clean_series(out_s, [k for k, p in enumerate(out_s) if p is not None], thr)
    both = sorted(set(in_real) & set(out_real))
    width = float(np.median([cdist(in_s[k], out_s[k]) for k in both])) if both else 50.0
    in_f = _temporal_fill(in_s, in_real, max_gap)
    out_f = _temporal_fill(out_s, out_real, max_gap)
    # geometry fill for long gaps from the partner rail (+ a uncapped temporal last resort)
    in_geo = out_geo = 0
    for k in range(n):
        ref_in = in_s[in_real[np.argmin([abs(j - k) for j in in_real])]] if in_real else None
        ref_out = out_s[out_real[np.argmin([abs(j - k) for j in out_real])]] if out_real else None
        if in_f[k] is None and out_f[k] is not None and ref_in is not None:
            in_f[k] = _shift_pts(out_f[k], width, ref_in)
            in_geo += 1
        if out_f[k] is None and in_f[k] is not None and ref_out is not None:
            out_f[k] = _shift_pts(in_f[k], width, ref_out)
            out_geo += 1
    in_f = _fill_series(in_f, [k for k, p in enumerate(in_f) if p is not None])
    out_f = _fill_series(out_f, [k for k, p in enumerate(out_f) if p is not None])
    # EXPORT the filled per-frame rails (full coverage) for downstream horse lane-bounding
    nanp = np.full((80, 2), np.nan, np.float32)
    np.savez(cache.replace(".pkl", "_rails.npz"),
             idx=np.array([r["i"] for r in rec]),
             inner=np.array([(p if p is not None else nanp) for p in in_f], np.float32),
             outer=np.array([(p if p is not None else nanp) for p in out_f], np.float32))
    # METRICS
    loo, gaps = [], []
    for ser, real in ((in_s, in_real), (out_s, out_real)):
        for idx, k in enumerate(real):
            if 0 < idx < len(real) - 1:
                a, b = real[idx - 1], real[idx + 1]
                f = (k - a) / (b - a)
                loo.append(cdist((1 - f) * ser[a] + f * ser[b], ser[k]))
    res = []
    for k, r in enumerate(rec):
        if r["sel"] is not None and k in set(in_real) | set(out_real) and in_f[k] is not None:
            d, _ = cKDTree(np.vstack([in_f[k], out_f[k]])).query(r["sel"])
            res.append(float(np.median(d)))
    # gap-length histogram for filled frames
    for ser, real in ((in_s, in_real), (out_s, out_real)):
        rset = set(real)
        for k in range(n):
            if k not in rset:
                bef = [j for j in real if j < k]
                aft = [j for j in real if j > k]
                if bef and aft:
                    gaps.append(aft[0] - bef[-1])
    tiles = []
    for k, r in enumerate(rec):
        if r["thumb"] is None or in_f[k] is None:
            continue
        sc = 560 / 960
        ov = (r["thumb"] // 2).copy()
        kk = "both" if (k in set(in_real) and k in set(out_real)) else "fill"
        cv2.polylines(ov, [(out_f[k] * sc).astype(np.int32)], False, (255, 0, 255), 2)
        cv2.polylines(ov, [(in_f[k] * sc).astype(np.int32)], False, (255, 255, 0), 2)
        cv2.putText(ov, f"t={r['t']:.0f} {kk}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        tiles.append(ov)
    if tiles:
        cols = 4
        rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
                for r in range(0, len(tiles), cols)]
        cv2.imwrite(strip, np.vstack(rows))
    ra, la, ga = np.array(res), np.array(loo), np.array(gaps) if gaps else np.array([0])
    nb = len(both)
    print(f"pass2 (max_gap={max_gap}, thr={thr}): frames {n}  both-real {nb} ({100*nb/n:.0f}%)  "
          f"width-rej {nbad}  outer-boundary-fallback {nfb}  geo-fill in {in_geo}/out {out_geo}")
    print(f"  resid(real px) med {np.median(ra):.1f} p90 {np.percentile(ra,90):.1f}")
    print(f"  LOO temporal med {np.median(la):.1f} p90 {np.percentile(la,90):.1f} worst {la.max():.0f}")
    print(f"  fill-gap frames: <={max_gap}f {int((ga<=max_gap).sum())}  >{max_gap}f {int((ga>max_gap).sum())}  max {ga.max()}f")


if __name__ == "__main__":
    import sys
    if "--pass1" in sys.argv:
        pass1(t0=0.0, t1=113.0)
    elif "--pass2" in sys.argv:
        pass2()
    else:
        pass1(t0=0.0, t1=113.0)
        pass2()

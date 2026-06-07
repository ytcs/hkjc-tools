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


def label_pair(ma, mb, prev_in, prev_out, band):
    """Assign the two rail models to inner/outer: by temporal proximity to prev, else green-side."""
    if prev_in is not None:
        da = cdist(ma["sample"](40), prev_in["sample"](40))
        db = cdist(mb["sample"](40), prev_in["sample"](40))
        return (ma, mb) if da < db else (mb, ma)
    ga, gb = _green_both_sides(ma, band), _green_both_sides(mb, band)
    return (ma, mb) if ga >= gb else (mb, ma)        # inner = greener-both-sides


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
    print(f"window {t0}-{t1}: frames {sum(cov.values())}  coverage {cov}")
    print(f"  resid(selected px) med {np.median(ra):.1f}px  mean {ra.mean():.1f}  "
          f"p90 {np.percentile(ra, 90):.1f}  worst {ra.max():.1f}")


if __name__ == "__main__":
    run()

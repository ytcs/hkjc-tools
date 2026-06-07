"""DRONE rail by TEMPORAL POSE TRACKING (the real solution).

Seed the homography H (metric track -> image) on a clean frame, then PROPAGATE it frame-to-frame:
init from the previous H, refine against the visible rail pixels by a GATED chamfer (model points
with no nearby data don't pull) + a reprojection-anchored REGULARISER toward the previous frame
(prevents collapse when a rail is occluded). The rigid known model carries the OCCLUDED inner rail,
so it projects correctly even on the dark far turn where per-frame detection drops it. Output = both
rails every frame + metric scale (px<->m via H).

Demo window: the t~48-74 bend (includes the dark t=68 where single-frame fitting failed).
"""
from __future__ import annotations

import importlib.util

import av
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

_spec = importlib.util.spec_from_file_location("cdr", "studies/cv_drone_rail.py")
cdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdr)

VIDEO = "data_cache/diag/drone_scan/drone_R08_half.mp4"
WIDTH_M = 22.5
HALF_W = WIDTH_M / 2
R_BEND_M = 159.0


def arc_model(span, n=120, R=R_BEND_M):
    a = np.linspace(-span / 2, span / 2, n)
    inner = np.c_[(R - HALF_W) * np.cos(a), (R - HALF_W) * np.sin(a)]
    outer = np.c_[(R + HALF_W) * np.cos(a), (R + HALF_W) * np.sin(a)]
    return inner, outer


def apply_H(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _resample(poly, n):
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return poly[:1].repeat(n, 0)
    g = np.linspace(0, s[-1], n)
    return np.c_[np.interp(g, s, poly[:, 0]), np.interp(g, s, poly[:, 1])]


def detect(bgr):
    cands, band, _ = cdr.rail_candidates(bgr)
    if len(cands) < 2:
        c2, _, _ = cdr.rail_candidates(bgr, band=band, tophat=True)
        if len(c2) > len(cands):
            cands = c2
    det = np.vstack([np.c_[c["xs"], c["ys"]] for c in cands]) if cands else None
    return cands, det


def seed_H(cands):
    """Fit H from the two longest detected arcs (param-matched findHomography), like the POC."""
    cands = sorted(cands, key=lambda c: -c["len"])[:2]
    A = cdr.fit_model(cands[0]["xs"], cands[0]["ys"])
    B = cdr.fit_model(cands[1]["xs"], cands[1]["ys"])
    if A["type"] != "arc" or B["type"] != "arc":
        return None
    n = 80
    Ap, Bp = _resample(A["sample"](n), n), _resample(B["sample"](n), n)
    if np.linalg.norm(Ap[0] - Bp[0]) > np.linalg.norm(Ap[0] - Bp[-1]):
        Bp = Bp[::-1]
    aA, aB = cdr._force_arc(cands[0]), cdr._force_arc(cands[1])
    span = max(aA["span"], aB["span"])
    mi, mo = arc_model(span, n)
    det_out, det_in = (Ap, Bp) if aA["R"] >= aB["R"] else (Bp, Ap)
    model = np.vstack([mo, mi]).astype(np.float64)
    img = np.vstack([det_out, det_in]).astype(np.float64)
    H, _ = cv2.findHomography(model, img, cv2.RANSAC, 5.0)
    return H


def refine(H_prev, det, model, anchors, gate, lam=0.06):
    """Gated chamfer (model->data) + reprojection-anchored regulariser toward H_prev."""
    tree = cKDTree(det)
    anch_prev = apply_H(H_prev, anchors)

    def resid(h):
        H = np.append(h, 1).reshape(3, 3)
        d, _ = tree.query(apply_H(H, model))
        d = np.where(d < gate, d, 0.0)                  # unmatched model pts don't pull
        reg = lam * (apply_H(H, anchors) - anch_prev).ravel()
        return np.concatenate([d, reg])

    sol = least_squares(resid, H_prev.flatten()[:8], loss="soft_l1", f_scale=4.0, max_nfev=80)
    H = np.append(sol.x, 1).reshape(3, 3)
    d, _ = tree.query(apply_H(H, model))
    matched = d < gate
    return H, float(np.median(d[matched])) if matched.any() else 1e9, float(matched.mean())


def draw(bgr, det, H, mi, mo, t, tag):
    ov = (bgr // 2).copy()
    if det is not None:
        ov[det[:, 1], det[:, 0]] = (0, 120, 180)
    if H is not None:
        po, pi = apply_H(H, mo), apply_H(H, mi)
        cv2.polylines(ov, [po.astype(np.int32)], False, (255, 0, 255), 2)
        cv2.polylines(ov, [pi.astype(np.int32)], False, (255, 255, 0), 2)
        spm = float(np.median(np.linalg.norm(pi - po, axis=1))) / WIDTH_M
        cv2.putText(ov, f"t={t:.1f} {tag} {spm:.1f}px/m", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(ov, f"t={t:.1f} no-seed", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return ov


if __name__ == "__main__":
    T0, T1, STRIDE = 48.0, 74.0, 5
    mi, mo = arc_model(1.4, 120)                          # wide model for projection/tracking
    model = np.vstack([mo, mi])
    anchors = np.vstack([mo[::20], mi[::20]])
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    H = None
    i = -1
    tiles = []
    n_seed = n_track = n_fail = 0
    for fr in cont.decode(st):
        i += 1
        t = i / 25.0
        if t > T1:
            break
        if i % STRIDE or t < T0:
            continue
        bgr = fr.to_ndarray(format="bgr24")
        cands, det = detect(bgr)
        gate = 12.0 * cdr.SCALE
        tag = ""
        if H is None:
            if len(cands) >= 2:
                H = seed_H(cands)
                tag = "SEED"
                n_seed += 1
        elif det is not None:
            H, med, frac = refine(H, det, model, anchors, gate)
            tag = f"track m{med:.1f} f{frac:.2f}"
            n_track += 1
        if H is None:
            n_fail += 1
        if i % 5 == 0:
            tiles.append(cv2.resize(draw(bgr, det, H, mi, mo, t, tag), (560, 315)))
    cols = 4
    rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
            for r in range(0, len(tiles), cols)]
    cv2.imwrite("data_cache/diag/drone_scan/pose_track.png", np.vstack(rows))
    print(f"window t={T0}-{T1}  seed {n_seed}  tracked {n_track}  fail {n_fail}")

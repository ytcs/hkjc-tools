"""DRONE rail via HOMOGRAPHY to the known metric track model (POC) — sports-field-registration
approach (TVCalib / No-Bells-Just-Whistles).

The running surface is PLANAR with KNOWN geometry (HV "C": width 22.5 m, bend radius ~159 m). Each
frame relates to the ground plane by a homography H (metric -> image). Fit H so the known model
rails reproject onto the detected rail pixels, then PROJECT THE FULL MODEL back -> both rails,
metric (px<->m) for free.

POC: single-frame fit. (1) detect the two longest rail curves (cv_drone_rail). (2) build a local
metric model (concentric arc pair on a bend / parallel line pair on a straight), param-matched.
(3) H via cv2.findHomography (RANSAC) on the param correspondences. (4) chamfer-refine.
(5) overlay the projected full model. Validate VISUALLY on bends + a straight.
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
N = 80


def _resample(poly, n):
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return poly[:1].repeat(n, 0)
    return np.c_[np.interp(np.linspace(0, s[-1], n), s, poly[:, 0]),
                 np.interp(np.linspace(0, s[-1], n), s, poly[:, 1])]


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def run_frame(bgr, t):
    cands, band, _ = cdr.rail_candidates(bgr)
    if len(cands) < 2:
        c2, _, _ = cdr.rail_candidates(bgr, band=band, tophat=True)
        if len(c2) >= 2:
            cands = c2
    ov = (bgr // 2).copy()
    if cands:
        det = np.vstack([np.c_[c["xs"], c["ys"]] for c in cands])
        ov[det[:, 1], det[:, 0]] = (0, 120, 180)
    if len(cands) < 2:
        cv2.putText(ov, f"t={t:.0f}s <2 rails", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return ov, None
    cands = sorted(cands, key=lambda c: -c["len"])[:2]
    A = cdr.fit_model(cands[0]["xs"], cands[0]["ys"])
    B = cdr.fit_model(cands[1]["xs"], cands[1]["ys"])
    Ap, Bp = _resample(A["sample"](N), N), _resample(B["sample"](N), N)
    if np.linalg.norm(Ap[0] - Bp[0]) > np.linalg.norm(Ap[0] - Bp[-1]):
        Bp = Bp[::-1]                                   # align traversal direction
    # bend if BOTH fit arcs of plausible (image) radius; assign outer = larger radius
    is_bend = A["type"] == "arc" and B["type"] == "arc"
    if is_bend:
        aA = cdr._force_arc(cands[0])
        aB = cdr._force_arc(cands[1])
        span = max(aA["span"], aB["span"])
        a = np.linspace(-span / 2, span / 2, N)
        m_out = np.c_[(R_BEND_M + HALF_W) * np.cos(a), (R_BEND_M + HALF_W) * np.sin(a)]
        m_in = np.c_[(R_BEND_M - HALF_W) * np.cos(a), (R_BEND_M - HALF_W) * np.sin(a)]
        det_out, det_in = (Ap, Bp) if aA["R"] >= aB["R"] else (Bp, Ap)
        geom = f"bend spanRad={span:.2f}"
    else:
        sep = float(np.median(np.linalg.norm(Ap - Bp, axis=1)))
        scale = sep / WIDTH_M
        L = float(np.sqrt(((Ap[-1] - Ap[0]) ** 2).sum())) / scale
        x = np.linspace(-L / 2, L / 2, N)
        m_out = np.c_[x, np.full(N, +HALF_W)]
        m_in = np.c_[x, np.full(N, -HALF_W)]
        det_out, det_in = Ap, Bp                       # straight: outer/inner arbitrary (symmetric)
        geom = f"straight sep={sep:.0f}px"
    model = np.vstack([m_out, m_in]).astype(np.float64)
    img = np.vstack([det_out, det_in]).astype(np.float64)
    H, _ = cv2.findHomography(model, img, cv2.RANSAC, 5.0)
    if H is None:
        cv2.putText(ov, f"t={t:.0f}s H-FAIL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return ov, None
    # chamfer refine H against ALL detected rail pixels (robust)
    tree = cKDTree(det)

    def resid(h):
        d, _ = tree.query(apply_h(np.append(h, 1).reshape(3, 3), model))
        return d
    sol = least_squares(resid, H.flatten()[:8], loss="soft_l1", f_scale=6.0, max_nfev=120)
    H = np.append(sol.x, 1).reshape(3, 3)
    med = float(np.median(resid(sol.x)))
    pi, po = apply_h(H, m_in), apply_h(H, m_out)
    cv2.polylines(ov, [po.astype(np.int32)], False, (255, 0, 255), 2)   # OUTER magenta
    cv2.polylines(ov, [pi.astype(np.int32)], False, (255, 255, 0), 2)   # INNER cyan
    scale_px_m = float(np.median(np.linalg.norm(pi - po, axis=1))) / WIDTH_M
    cv2.putText(ov, f"t={t:.0f}s {geom} chamfer={med:.1f}px {scale_px_m:.1f}px/m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return ov, med


if __name__ == "__main__":
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    times = [30, 54, 68, 96]
    want = {int(round(t * 25)): t for t in times}
    got = {}
    i = -1
    for fr in cont.decode(st):
        i += 1
        if i in want:
            got[want[i]] = fr.to_ndarray(format="bgr24")
        if i > max(want):
            break
    tiles = []
    for t in times:
        ov, med = run_frame(got[t], t)
        cv2.imwrite(f"data_cache/diag/drone_scan/homog_t{t}.png", ov)
        tiles.append(cv2.resize(ov, (600, 338)))
        print(f"t={t}: chamfer_med={med}")
    rows = [np.hstack(tiles[r:r + 2]) for r in range(0, len(tiles), 2)]
    cv2.imwrite("data_cache/diag/drone_scan/homography_poc.png", np.vstack(rows))

"""Pose backbone (task #101) — per-frame homography track-plane->image, the FUNDAMENTAL metric fix.

Replaces the per-frame frame-local-s + jittery head-reference + scalar-scale reconstruction with a
homography H that maps the KNOWN rigid track plane (inner rail b=0, outer rail b=22.5 m) into the
image. Then every horse pixel -> H^-1 -> metric (along, lateral) DIRECTLY: no per-frame origin, no
s_head jitter, and a perspective-correct (depth-varying) scale baked into H.

This module validates the CORE per-frame fit (the temporal optical-flow init + EKF/RTS smoothing are
the next layers). For each test frame: detect the two main rails, build the rigid two-rail template
for the local geometry (straight or arc R), fit H by ICP (affine init -> DLT refine), overlay the
back-projected rails, and report the field's lateral spread in metres.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_pose_backbone.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
DIAG = "data_cache/diag"
HALF = 960
DIST = 1800.0
FIELD = 11
RAIL_W = 22.5
FRAMES = [1080, 1760, 2200, 2440]


def build_template(R, window=120.0, step=3.0):
    """Two-rail template in track-plane metres: inner (b=0) + outer (b=RAIL_W), labelled.

    Straight => R=inf (lines); bend => concentric arcs of radius R and R+RAIL_W. ``a`` is along-track
    metres about the visible centre; returns (plane (M,2), label (M,) 0=inner 1=outer).
    """
    a = np.arange(-window, window + step, step)
    pts, lab = [], []
    if not np.isfinite(R):
        for b, lb in ((0.0, 0), (RAIL_W, 1)):
            pts.append(np.column_stack([a, np.full_like(a, b)]))
            lab.append(np.full(len(a), lb))
    else:
        phi = a / R
        for b, lb in ((0.0, 0), (RAIL_W, 1)):
            rad = R + b
            pts.append(np.column_stack([rad * np.sin(phi), R - rad * np.cos(phi)]))
            lab.append(np.full(len(a), lb))
    return np.vstack(pts), np.concatenate(lab)


def _project_h(plane, H):
    q = np.column_stack([plane, np.ones(len(plane))]) @ H.T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, np.nan, w)     # guard degenerate poses (points at infinity)
    return q[:, :2] / w


def _affine_init(inner_px, outer_px):
    """Affine track-plane->image init from the rails' PCA + the known 22.5 m lateral ruler."""
    p0 = inner_px.mean(0)
    e1 = np.linalg.svd(inner_px - p0, full_matrices=False)[2][0]
    nrm = np.array([-e1[1], e1[0]])
    sep = float((outer_px.mean(0) - p0) @ nrm)
    s_along = float(np.percentile(np.abs((inner_px - p0) @ e1), 95)) / 100.0 + 1e-6
    H = np.eye(3)
    H[:2, :2] = np.column_stack([s_along * e1, (sep / RAIL_W) * nrm])
    H[:2, 2] = p0
    return H


def fit_homography(inner_px, outer_px, R, init_H=None, n_iter=20, trim=0.30):
    """ICP fit of the two-rail template to detected inner/outer rail pixels. Returns (H, resid).

    The render is a tilted/anisotropic TOP-DOWN view, NOT a perspective camera, so we fit a 6-DOF
    AFFINE (estimateAffine2D) — it matches the free homography's accuracy (and beats it on the home
    turn) while its parameters live in a FLAT, SMOOTHABLE space (the free homography did not). Init
    from ``init_H`` (previous pose) if given, else the affine guess from the rails' PCA + RAIL_W ruler.
    """
    import cv2
    plane, label = build_template(R)
    pix = np.vstack([inner_px, outer_px])
    plab = np.r_[np.zeros(len(inner_px)), np.ones(len(outer_px))]
    H = init_H.copy() if init_H is not None else _affine_init(inner_px, outer_px)
    for _ in range(n_iter):
        proj = _project_h(plane, H)
        # nearest template point of the SAME label for each pixel
        keep_src, keep_dst = [], []
        for lb in (0, 1):
            tp = plane[label == lb]
            tq = proj[label == lb]
            px = pix[plab == lb]
            if len(px) == 0 or len(tq) == 0:
                continue
            d2 = ((px[:, None, 0] - tq[None, :, 0]) ** 2 + (px[:, None, 1] - tq[None, :, 1]) ** 2)
            nn = d2.argmin(1)
            res = np.sqrt(d2[np.arange(len(px)), nn])
            ok = res <= np.quantile(res, 1 - trim)
            keep_src.append(tp[nn[ok]])
            keep_dst.append(px[ok])
        if not keep_src:
            break
        src = np.vstack(keep_src).astype(np.float32)
        dst = np.vstack(keep_dst).astype(np.float32)
        if len(src) < 8:                          # too few correspondences -> keep current H
            break
        M, _m = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)   # 6-DOF affine (smoothable)
        if M is None or not np.isfinite(M).all():
            break
        H = np.vstack([M, [0.0, 0.0, 1.0]])
    proj = _project_h(plane, H)
    d2 = ((pix[:, None, 0] - proj[None, :, 0]) ** 2 + (pix[:, None, 1] - proj[None, :, 1]) ** 2)
    resid = float(np.median(np.sqrt(d2.min(1))))
    return H, resid


def main():
    import av
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars
    from hkjc.vision.feeds.aerial_rail import line_candidates, select_running_rail
    os.makedirs(DIAG, exist_ok=True)
    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)
    geom = build_rail_geometry("HV", "C")

    # ACROSS-RACE pass 1: collect a reliable two-rail homography ANCHOR per good frame, plus every
    # frame's field pixels. (Temporal init from the previous anchor speeds convergence.)
    prev_H = None
    anchors = []              # (frame, H normalised so H[2,2]=1)
    frame_fpts = {}           # frame -> field pixels (read y_lat after smoothing)
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        img = fr.to_ndarray(format="bgr24")
        panel = img[:, HALF:]
        if not render_active(panel):
            continue
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        dets, _d = detect_avatars(panel, ty0, ty1, field=FIELD)
        if len(dets) < 3:
            continue
        rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
        cands = line_candidates(panel, rb0, rb1)
        fpts = np.array([[d.cx, d.cy] for d in dets], float)
        d_ff = DIST - float(np.interp(i, FR[fin], ODO[fin]))
        R = local_radius(geom, d_ff)
        inner, _t, info = select_running_rail(cands, deg=(2 if np.isfinite(R) else 1), field_pts=fpts)
        if inner is None:
            continue
        frame_fpts[i] = fpts
        outer = (info or {}).get("outer")
        if outer is None:
            continue
        H, resid = None, 1e9
        for init in (prev_H, None):
            hc, rc = fit_homography(inner, outer, R, init_H=init)
            if np.isfinite(hc).all() and rc < resid:
                H, resid = hc, rc
            if resid < 20:
                break
        if H is not None and resid < 25.0 and abs(H[2, 2]) > 1e-9:
            prev_H = H
            anchors.append((i, H / H[2, 2]))

    # ---- TEMPORAL SMOOTHING (SOTA layer C, batch): smooth the anchor poses, INTERPOLATE H to every
    # frame. Fixes both the per-frame fit jitter and the frozen one-rail stretches (a stuck carry).
    af = np.array([a[0] for a in anchors])
    hp = np.array([a[1].flatten()[:8] for a in anchors])      # 6 affine params (+2 zeros) per anchor
    fw = 30                                                    # frame half-window (~1.2 s)

    def h_at(f):
        m = np.abs(af - f) <= fw                              # FRAME-windowed robust median: outlier-
        if m.sum() < 3:                                       # resistant + no linear-ramp interpolation
            order = np.argsort(np.abs(af - f))[:5]
            m = np.zeros(len(af), bool)
            m[order] = True
        p = np.median(hp[m], axis=0)
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])

    rows = []
    for f in sorted(frame_fpts):
        b = np.abs(_project_h(frame_fpts[f], np.linalg.inv(h_at(f)))[:, 1])
        if not np.isfinite(b).all() or b.max() > 60.0:
            continue
        rows.append((f, float(np.median(b)), float(np.percentile(b, 10)), float(np.percentile(b, 90))))
    rows = np.array(rows)
    print(f"anchors (two-rail)={len(anchors)}; frames reconstructed={len(rows)}")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], alpha=0.3, label="field 10-90 pct")
    ax.plot(rows[:, 0], rows[:, 1], lw=1.0, label="field median y_lat")
    ax.set_xlabel("aerial frame")
    ax.set_ylabel("y_lat from running rail (m)  [homography backbone]")
    ax.set_title("Pose-backbone LATERAL reconstruction — perspective-correct, two-rail metric "
                 "(anchor-smoothed + interpolated pose)")
    ax.set_ylim(0, 30)
    ax.legend()
    out = f"{DIAG}/cv_pose_lateral.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[fig] {out}  median y_lat over race = {np.median(rows[:, 1]):.1f} m")


if __name__ == "__main__":
    main()

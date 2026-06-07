"""Preprocessing-first avatar detection (user idea): SUPPRESS THE TURF, then find horses.

Alternative to silk-patch blobs (which over-detect ~2x). The AVV render has near-uniform green
turf, so we: (1) auto-detect the turf hue (modal hue of saturated greenish pixels -- no tuning),
(2) build a NON-TURF mask = dark horse bodies + saturated non-green silks (excluding the bright
thin rail), (3) morphologically close so each horse's body+silk+shadow joins into ONE blob,
(4) connected components filtered by the known ~50x16 footprint -> one detection per horse, with
ORIENTATION from the blob's own moments (per-horse heading, no global PCA needed).

Renders the turf-suppressed mask + the resulting oriented boxes for review, side by side, to
data_cache/diag/. No per-race tuning: turf hue is detected; footprint is the measured render
constant.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_preproc.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
FIELD = 11
DEBUG_FRAMES = [770, 1400, 2030, 2580]
AVATAR_AREA = (90, 1100)     # px^2 of one horse+shadow blob (~50x16 with slack)


MIN_AREA = 30        # px^2 noise floor for a horse fragment


def _det_from_pts(pts, ty0):
    """(cx,cy,angle,length,area) for a set of blob pixels (pts = Nx2 [x,y] in band coords)."""
    c = pts - pts.mean(0)
    _u, sg, vt = np.linalg.svd(c, full_matrices=False)
    ang = float(np.arctan2(vt[0, 1], vt[0, 0]))
    length = float(4 * np.sqrt(max(sg[0] ** 2, 1.0) / len(pts)))
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean() + ty0), ang, length, len(pts))


def detect_preproc(panel, ty0, ty1, field=FIELD):
    """Turf-suppress -> ONE complete blob per horse -> split merges with k-means (known field size).

    LOOSE mask (user: prior filter too strong): keep everything that is NOT turf and NOT the bright
    rail -> the whole horse comes through (no fragmentation). Merges are then SPLIT: avg horse area =
    total/field, so a blob of area A is k=round(A/avg) horses, partitioned by k-means on its pixels
    (a bunched clump splits into its constituents; one giant finish blob -> field pieces).
    """
    import cv2

    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    # LOOSE: turf = green hue at ANY brightness; horse = not-turf and not bright-rail
    is_turf = (d_hue < 18) & (s > 28)
    white_rail = (v > 180) & (s < 50)
    horse = ((~is_turf) & (~white_rail)).astype(np.uint8)
    horse = cv2.morphologyEx(horse, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    horse = cv2.morphologyEx(horse, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, stats, _cent = cv2.connectedComponentsWithStats(horse, connectivity=8)
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
    if not comps:
        return [], (horse, turf_hue, ty0)
    total = sum(a for _i, a in comps)
    single = max(MIN_AREA, total / field)     # average horse area
    dets = []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    for i, area in comps:
        ys, xs = np.where(lbl == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        k = max(1, int(round(area / single)))
        if k == 1 or len(pts) < 2 * k:
            dets.append(_det_from_pts(pts, ty0))
        else:
            _c, labels, _ctr = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            for j in range(k):
                sub = pts[labels.ravel() == j]
                if len(sub) >= 2:
                    dets.append(_det_from_pts(sub, ty0))
    return dets, (horse, turf_hue, ty0)


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import track_band
    os.makedirs(DIAG, exist_ok=True)

    want = set(DEBUG_FRAMES)
    frames = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(frames) == len(want):
                break

    for f in DEBUG_FRAMES:
        if f not in frames:
            continue
        panel = frames[f]
        ty0, ty1 = track_band(panel)
        dets, (mask, turf_hue, _ty0) = detect_preproc(panel, ty0, ty1)
        vis = panel.copy()
        for (cx, cy, ang, length, area) in dets:
            ux, uy = np.cos(ang), np.sin(ang)
            L = max(20.0, length) / 2
            W = 8.0
            corners = [[int(cx + sx * ux - sy * uy), int(cy + sx * uy + sy * ux)]
                       for sx, sy in [(-L, -W), (L, -W), (L, W), (-L, W)]]
            cv2.polylines(vis, [np.array(corners)], True, (0, 220, 255), 1)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        cv2.putText(vis, f"f{f}: {len(dets)} horse blobs (field={FIELD}); turf-hue {turf_hue}; per-blob orientation",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        # stack the mask (as BGR) under the boxes panel for review
        maskbgr = cv2.cvtColor(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        maskfull = np.zeros_like(panel)
        maskfull[ty0:ty0 + mask.shape[0]] = maskbgr
        combo = np.vstack([vis, maskfull])
        out = f"{DIAG}/cv_avatar_preproc_f{f:04d}.png"
        cv2.imwrite(out, combo)
        print(f"f{f}: {len(dets)} horse blobs (field={FIELD}); turf-hue={turf_hue}  -> {out}")


if __name__ == "__main__":
    main()

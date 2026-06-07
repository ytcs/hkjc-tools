"""Temporal INNER-rail tracker (replaces per-frame greedy selection + the reverted reseed).

The fundamental fix for the outer-rail latch (f772/f2200): the inner running rail is occluded by the
field that runs on it, while the outer rail/boundary is clear — so per-frame 'nearest line' grabs the
wrong one. Instead:
  PASS 1: per frame extract ALL candidate rail lines (each large cyan component -> its own curve) and
          each candidate's image-y where it crosses the field-x (ry).
  PASS 2: TRACK the inner rail's ry as a smooth trajectory — forward+backward, predicting ry and
          selecting the candidate nearest the prediction (the inner rail evolves smoothly as the
          camera pans; the outer rail sits at a different, offset ry). Where no candidate is near the
          prediction (inner fully occluded), CARRY the predicted curve. No per-frame greedy latch.

Validates on the previously-failing frames (752,760,772,2200) + a temporal plot. Iterate against the
outlier overlays (cv_rail_outliers.py) until clean.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_temporal.py
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
STRIDE = 4
SEL_TOL = 30.0       # px: candidate ry within this of the prediction is a match
CHECK = [752, 760, 772, 2200, 400, 910, 1420, 2785]


def candidate_curves(band, ty0, white_xs, white_ys, min_extent=40, deg=2):
    """ALL candidate rail lines: each large connected component -> its own robust curve (panel coords)."""
    import cv2
    if len(white_xs) < 30:
        return []
    h, w = band.shape[:2]
    m = np.zeros((h, w), np.uint8)
    yy, xx = white_ys.astype(int), white_xs.astype(int)
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    m[yy[ok], xx[ok]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        ext = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if ext < min_extent:
            continue
        ys_i, xs_i = np.where(lbl == i)
        P = np.column_stack([xs_i, ys_i]).astype(float)
        horiz = np.ptp(P[:, 0]) >= np.ptp(P[:, 1])
        a, b = (P[:, 0], P[:, 1]) if horiz else (P[:, 1], P[:, 0])
        if len(a) < 12:
            continue
        coeffs = np.polyfit(a, b, min(deg, 2))
        ga = np.linspace(a.min(), a.max(), 120)
        gb = np.polyval(coeffs, ga)
        curve = np.column_stack([ga, gb + ty0]) if horiz else np.column_stack([gb, ga + ty0])
        out.append(curve)
    return out


def _ry(curve, x):
    k = int(np.argmin(np.abs(curve[:, 0] - x)))
    return float(curve[k, 1])


def main() -> None:
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_avatar_edgefill import detect_edgefill
    from cv_s0d_aerial_render import track_band
    from hkjc.vision.geometry_cv import white_rail_pixels
    from hkjc.vision.mode_b.layout import find_layout
    os.makedirs(DIAG, exist_ok=True)

    # PASS 1: per-frame candidates + field-x
    rec = []   # (frame, fx, [(curve, ry), ...])
    panels = {}
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i % STRIDE:
            continue
        img = fr.to_ndarray(format="bgr24")
        if not find_layout(img).aerial_active:
            continue
        panel = img[:, HALF:]
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        rb0 = int(0.28 * ph)
        xs, ys = white_rail_pixels(panel[rb0:int(0.82 * ph)], v_min=165, s_max=70)
        dets, _b, _d = detect_edgefill(panel, ty0, ty1)
        if not dets:
            continue
        fx = float(np.mean([d[0] for d in dets]))
        fy = float(np.mean([d[1] for d in dets]))
        cands = [(c, _ry(c, fx)) for c in candidate_curves(panel[rb0:int(0.82 * ph)], rb0, xs, ys)]
        rec.append([i, fx, cands, fy])
        if i in CHECK:
            panels[i] = panel.copy()
    container.close()

    frames = [r[0] for r in rec]
    fy_arr = np.array([r[3] for r in rec])
    n = len(rec)

    def robust_smooth(y, win=9):
        """Moving median (nan-aware) — rejects outlier picks, keeps the dominant trajectory."""
        s = np.full(n, np.nan)
        for k in range(n):
            w = y[max(0, k - win):min(n, k + win + 1)]
            w = w[np.isfinite(w)]
            if len(w):
                s[k] = float(np.median(w))
        return s

    # INIT: per frame, the candidate nearest the FIELD (inner rail is the band nearest the horses).
    est = np.array([
        min(cands, key=lambda c: abs(c[1] - fy_arr[k]))[1] if cands else np.nan
        for k, (_fr, _fx, cands, _fy) in enumerate(rec)
    ])
    # GLOBAL robust smoothing + re-selection, iterated -> lock the dominant inner band, reject jumps
    for _ in range(5):
        sm = robust_smooth(est)
        sm = np.interp(np.arange(n), np.where(np.isfinite(sm))[0], sm[np.isfinite(sm)])
        for k, (_fr, _fx, cands, _fy) in enumerate(rec):
            if cands:
                est[k] = min(cands, key=lambda c: abs(c[1] - sm[k]))[1]   # noqa: B023
    inner = robust_smooth(est)
    inner = np.interp(np.arange(n), np.where(np.isfinite(inner))[0], inner[np.isfinite(inner)])

    # OVERLAY check frames: draw ALL candidates (grey) + the SELECTED inner ry's curve (green)
    fidx = {fr: k for k, fr in enumerate(frames)}
    for f in CHECK:
        if f not in panels:
            continue
        vis = panels[f]
        k = fidx[f]
        cands = rec[k][2]
        target = inner[k]
        best = None
        for (c, ry) in cands:
            col = (110, 110, 110)
            for t in range(1, len(c)):
                cv2.line(vis, (int(c[t - 1, 0]), int(c[t - 1, 1])), (int(c[t, 0]), int(c[t, 1])), col, 1)
            if best is None or abs(ry - target) < abs(best[1] - target):
                best = (c, ry)
        if best is not None:
            c = best[0]
            for t in range(1, len(c)):
                cv2.line(vis, (int(c[t - 1, 0]), int(c[t - 1, 1])), (int(c[t, 0]), int(c[t, 1])), (0, 255, 0), 2)
        cv2.putText(vis, f"f{f}: grey=all candidates, GREEN=tracked inner rail (ry={target:.0f})",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{DIAG}/cv_rail_temporal_f{f:04d}.png", vis)

    # temporal plot: all candidate ry (grey) + tracked inner (green)
    fig, ax = plt.subplots(figsize=(11, 4))
    for _k, (fr, _fx, cands, _fy) in enumerate(rec):
        for (_c, ry) in cands:
            ax.plot(fr, ry, ".", color="0.75", ms=3)
    ax.plot(frames, inner, "-", color="C2", lw=1.5, label="tracked inner rail")
    ax.set_xlabel("aerial frame")
    ax.set_ylabel("rail image-y @ field-x (px)")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Temporal inner-rail tracking — grey=ALL candidate lines, green=tracked inner")
    fig.savefig(f"{DIAG}/cv_rail_temporal_plot.png", dpi=110, bbox_inches="tight")
    print(f"frames {n}; overlays for {[f for f in CHECK if f in panels]}; plot cv_rail_temporal_plot.png")


if __name__ == "__main__":
    main()

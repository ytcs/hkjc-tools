"""Dial-in RAIL RECON — broad-coverage diagnostic across the race.

Goal: localize the RUNNING rail (the inner rail the horses run along) per frame, as a rigid
straight/arc, so it can bound detection (lane band) + anchor the longitudinal recon. This first
diagnostic just SHOWS, over ~16 frames spanning the race:
  * white-rail-pixel CANDIDATES (cyan)  -- every white line: running rail, far rail, markers, edges
  * the HORSE cluster (red dots + magenta centroid) -- where horses actually are
  * a naive robust line fit (green) -- to see where it currently locks (often the WRONG band)

so the discriminator for "which line is the running rail" can be designed against reality.
Montage -> data_cache/diag/cv_rail_recon_montage.png (+ per-frame for zoom).

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_recon.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
SAMPLE = [0, 400, 570, 740, 910, 1080, 1250, 1420, 1590, 1760, 1930, 2100, 2270, 2440, 2610, 2785]


def fit_rail_curve(band, x_min, ty0, white_xs, white_ys, field_ctr=None, min_extent=40, inlier=12.0,
                   deg=2):
    """Rigid CURVE fit to the rail candidates. The field is used ONLY to SELECT which coherent line is
    the running rail (the one nearest the field — there are several: running rail, outer/foreground
    boundary, markers); the curve POSITION then comes entirely from that line's own pixels. Steps:
    components (extent≥min_extent drops dashes) → pick the SEED line nearest the field → fit quadratic
    → gather collinear inliers (within ``inlier`` px of the seed curve, so a rail broken by occlusion
    re-joins, but the far boundary/markers are excluded) → refit. Returns a panel polyline or None."""
    import cv2
    if len(white_xs) < 50:
        return None
    h, w = band.shape[:2]
    m = np.zeros((h, w), np.uint8)
    yy, xx = white_ys.astype(int), white_xs.astype(int)
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    m[yy[ok], xx[ok]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        ext = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if ext >= min_extent:
            ys_i, xs_i = np.where(lbl == i)
            comps.append(np.column_stack([xs_i, ys_i]).astype(float))
    if not comps:
        return None
    # SELECT seed line: nearest the field (band coords), else the largest
    if field_ctr is not None:
        fx, fy = field_ctr[0], field_ctr[1] - ty0
        seed = min(comps, key=lambda P: float(np.min((P[:, 0] - fx) ** 2 + (P[:, 1] - fy) ** 2)))
    else:
        seed = max(comps, key=len)
    horiz = np.ptp(seed[:, 0]) >= np.ptp(seed[:, 1])

    def axes(P):
        return (P[:, 0], P[:, 1]) if horiz else (P[:, 1], P[:, 0])
    a, b = axes(seed)
    if len(a) < 12:
        return None
    deg = min(deg, 2)                                          # RIGID: deg=1 straight, 2 arc (by geometry)
    coeffs = np.polyfit(a, b, deg)
    allP = np.vstack(comps)
    aa, bb = axes(allP)
    for _ in range(3):
        kp = np.abs(bb - np.polyval(coeffs, aa)) < inlier      # collinear with the seed curve
        if kp.sum() < 20:
            break
        coeffs = np.polyfit(aa[kp], bb[kp], deg)
    kp = np.abs(bb - np.polyval(coeffs, aa)) < inlier
    if kp.sum() < 20:
        return None
    ga = np.linspace(aa[kp].min(), aa[kp].max(), 120)
    gb = np.polyval(coeffs, ga)
    return (np.column_stack([ga, gb + ty0]) if horiz else np.column_stack([gb, ga + ty0]))


def main() -> None:
    import av
    import cv2

    from cv_avatar_edgefill import detect_edgefill
    from cv_s0d_aerial_render import track_band
    from hkjc.vision.geometry_cv import white_rail_pixels
    os.makedirs(DIAG, exist_ok=True)

    want = set(SAMPLE)
    frames = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(frames) == len(want):
                break

    tiles = []
    for f in SAMPLE:
        if f not in frames:
            continue
        panel = frames[f]
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)            # horse-detection band (centred on the field)
        # RAIL band must be WIDER: the rail can sit ABOVE or BELOW the field, outside the horse band.
        # Span below the top text/widget, above the bottom legend strip.
        rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
        rband = panel[rb0:rb1]
        xs, ys = white_rail_pixels(rband, v_min=165, s_max=70)   # candidates, rband coords
        vis = panel.copy()
        # mark candidate rail pixels cyan
        if len(xs):
            yy = (ys + rb0).astype(int)
            xx = xs.astype(int)
            ok = (yy >= 0) & (yy < vis.shape[0]) & (xx >= 0) & (xx < vis.shape[1])
            vis[yy[ok], xx[ok]] = (255, 255, 0)
        # horses — used ONLY to SELECT which line is the running rail, not to position the curve
        dets_a, _b, _dbg = detect_edgefill(panel, ty0, ty1)
        for d in dets_a:
            cv2.circle(vis, (int(d[0]), int(d[1])), 3, (0, 0, 255), -1)
        fctr = (float(np.mean([d[0] for d in dets_a])), float(np.mean([d[1] for d in dets_a]))) \
            if dets_a else None
        if fctr:
            cv2.drawMarker(vis, (int(fctr[0]), int(fctr[1])), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        # RAIL = rigid CURVE; field selects WHICH cyan line is the running rail, curve from rail pixels.
        curve = fit_rail_curve(rband, x_min=0, ty0=rb0, white_xs=xs, white_ys=ys, field_ctr=fctr)
        if curve is not None:
            for k in range(1, len(curve)):
                cv2.line(vis, (int(curve[k - 1, 0]), int(curve[k - 1, 1])),
                         (int(curve[k, 0]), int(curve[k, 1])), (0, 255, 0), 2)
        cv2.putText(vis, f"f{f}  cand={len(xs)}  horses={len(dets_a)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # save per-frame full-res for zoom
        cv2.imwrite(f"{DIAG}/cv_rail_recon_f{f:04d}.png", vis)
        tiles.append(cv2.resize(vis, (480, 270)))

    # 4x4 montage
    while len(tiles) < 16:
        tiles.append(np.zeros((270, 480, 3), np.uint8))
    rows = [np.hstack(tiles[r * 4:r * 4 + 4]) for r in range(4)]
    montage = np.vstack(rows)
    out = f"{DIAG}/cv_rail_recon_montage.png"
    cv2.imwrite(out, montage)
    print(f"wrote {out} (+ {len(tiles)} per-frame). CYAN=rail candidates, RED=horses, "
          f"MAGENTA=cluster centroid, GREEN=naive line fit.")


if __name__ == "__main__":
    main()

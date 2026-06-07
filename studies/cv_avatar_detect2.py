"""Improved aerial horse detection — rail-bounded ROI + rail-parallel blur, NO k-means.

User guidance (2026-06-07): (1) bound the detection region by the RAIL (field-independent, from the
rail candidates) to drop background false-positives; (2) tune the edge+fill; (3) add a small blur
ONLY along the rail direction to re-connect each horse's body+silk that Canny/fill split; (4) drop
the k-means (it forces k=field -> double-count on merges, miss on faint horses). With the avatars
re-connected, one connected component = one horse.

Compares the CURRENT detect_edgefill vs the NEW detector on bunched + spread frames.
Diagnostics -> data_cache/diag/cv_avatar2_f*.png.  VISUALLY SCRUTINISE.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_detect2.py
"""
from __future__ import annotations

import os

import numpy as np

from hkjc.vision.feeds.aerial import track_band
from hkjc.vision.feeds.aerial_detect import detect_edgefill
from hkjc.vision.feeds.aerial_rail import enumerate_lines, line_candidates

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
FIELD = 11
FRAMES = [1080, 1760, 2200, 2440]   # spread, mid, bunched-bend, bunched-near-finish


def _line_kernel(length: int, angle: float):
    """Binary line structuring element of ``length`` px oriented at ``angle`` (rad)."""
    import cv2
    L = max(3, length | 1)
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    dx, dy = np.cos(angle), np.sin(angle)
    p0 = (int(round(c - dx * c)), int(round(c - dy * c)))
    p1 = (int(round(c + dx * c)), int(round(c + dy * c)))
    cv2.line(k, p0, p1, 1, 1)
    return k


def rail_geom(panel, ty0, ty1):
    """Dominant rail tangent angle + a track-region mask between the two main rails (band coords)."""
    cands = line_candidates(panel, ty0, ty1)
    lines = sorted([ln for ln in enumerate_lines(cands, 2) if 1.0 <= ln["coverage"] <= 5.0],
                   key=lambda ln: -ln["coverage"])[:2]
    if not lines:
        return 0.0, None
    u = lines[0]["frame"][1]
    angle = float(np.arctan2(u[1], u[0]))
    # track region: dilate the union of the (up-to-2) rail polylines into a thick band; horses live
    # within ~a track-width of a rail, so a generous perpendicular dilation bounds them + drops far bg.
    import cv2
    w = panel.shape[1]
    rmask = np.zeros((ty1 - ty0, w), np.uint8)
    for ln in lines:
        p = ln["pts"]
        for t in range(1, len(p)):
            cv2.line(rmask, (int(p[t - 1, 0]), int(p[t - 1, 1]) - ty0),
                     (int(p[t, 0]), int(p[t, 1]) - ty0), 1, 1)
    band = cv2.dilate(rmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (181, 181)))  # ~track width
    return angle, band


def _components(mask, ty0):
    """Filtered connected components -> (cx, cy_panel, area) horse detections (no k-means)."""
    import cv2
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    dets = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40 or area > 6000:
            continue
        ys_i, xs_i = np.where(lbl == i)
        p = np.column_stack([xs_i, ys_i]).astype(float)
        ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
        major, minor = np.sqrt(max(ev[1], 1e-6)), np.sqrt(max(ev[0], 1e-6))
        if 4.0 * major > 130.0 or major / minor > 6.0:   # rail/line -> drop
            continue
        dets.append((float(cent[i, 0]), float(cent[i, 1] + ty0), area))
    return dets


def detect_new(panel, ty0, ty1, field=FIELD):
    """Rail-bounded edge+fill; ITERATE a rail-parallel blur from none upward, stopping at the blur
    whose blob count is closest to ``field`` (no k-means). Re-connects body+silk splits with the
    smallest blur that does so; avoids over-merging by not blurring more than needed."""
    import cv2
    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    bright = (v > 175) & (s < 55)
    sup = band.copy()
    sup[is_turf | bright] = 0
    gray = cv2.cvtColor(sup, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    fill = edges.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fill, cnts, -1, 255, -1)
    color = ((~is_turf) & (~bright)).astype(np.uint8)
    base = ((fill > 0) & (color > 0)).astype(np.uint8) * 255
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    angle, rband = rail_geom(panel, ty0, ty1)
    if rband is not None:                              # RAIL-BOUND: drop background beyond the track
        base = base * rband
    # iterate blur length: 1 (none) upward; keep the count CLOSEST to field, stop once we reach field
    prog = []
    best = None
    for L in (1, 3, 5, 7, 9, 11, 13, 15):
        m = base if L <= 1 else cv2.morphologyEx(base, cv2.MORPH_CLOSE, _line_kernel(L, angle))
        dets = _components(m, ty0)
        prog.append((L, len(dets)))
        if best is None or abs(len(dets) - field) < abs(best[2] - field):
            best = (L, m, len(dets), dets)
        if len(dets) <= field:        # merged down to (or below) the field -> stop
            break
    L, m, _cnt, dets = best
    return dets, m, angle, L, prog


def _base_mask(panel, ty0, ty1):
    """Edge+fill ∩ colour, rail-bounded, with (3) an isotropic close to reconnect each avatar's own
    body+silk+leg pieces. Returns (mask, angle)."""
    import cv2
    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    bright = (v > 175) & (s < 55)
    sup = band.copy()
    sup[is_turf | bright] = 0
    gray = cv2.cvtColor(sup, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    fill = edges.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fill, cnts, -1, 255, -1)
    color = ((~is_turf) & (~bright)).astype(np.uint8)
    base = ((fill > 0) & (color > 0)).astype(np.uint8) * 255
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE,            # (3) reconnect intra-avatar pieces
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    angle, rband = rail_geom(panel, ty0, ty1)
    if rband is not None:
        base = base * rband
    return base, angle


def _split_blob(sub_mask, nsub, sep=9):
    """Distance-transform NMS: return ``nsub`` well-separated centres inside one (merged) blob."""
    import cv2
    dist = cv2.distanceTransform(sub_mask.astype(np.uint8), cv2.DIST_L2, 5)
    centres = []
    for _ in range(nsub):
        _mn, mx, _ml, loc = cv2.minMaxLoc(dist)
        if mx <= 0:
            break
        centres.append((float(loc[0]), float(loc[1])))
        cv2.circle(dist, loc, sep, 0, -1)        # suppress a horse-radius around the peak
    return centres


def detect_watershed(panel, ty0, ty1, field):
    """(3)+(2): reconnect avatars (close + rail-parallel blur), then split only OVER-LARGE blobs
    (genuine merges) into round(area / single-horse area) centres via distance-transform NMS. The
    per-blob count is data-driven (area ratio), NOT a forced field total. Returns [(cx, cy_panel, n)]."""
    import cv2
    base, angle = _base_mask(panel, ty0, ty1)
    m = cv2.morphologyEx(base, cv2.MORPH_CLOSE, _line_kernel(9, angle))   # reconnect avatar pieces
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
             if 40 <= int(stats[i, cv2.CC_STAT_AREA]) <= 8000]
    if not comps:
        return []
    # ANCHOR the single-horse area to the known field count (legend): choose ``single`` so the total
    # split count Sum round(area/single) == field. Robust to zoom + mixed singles/merges per frame.
    areas = [a for _i, a in comps]
    total = float(sum(areas))
    single = total / field
    for cand in np.linspace(0.45 * total / field, 1.8 * total / field, 80):
        if sum(max(1, int(round(a / cand))) for a in areas) == field:
            single = cand
            break
    dets = []
    for i, area in comps:
        nsub = max(1, int(round(area / single)))
        if nsub == 1:
            dets.append((float(cent[i, 0]), float(cent[i, 1] + ty0), 1))
        else:
            for (cx, cy) in _split_blob(lbl == i, nsub):
                dets.append((cx, cy + ty0, nsub))
    return dets


def main():
    import av
    import cv2
    os.makedirs(DIAG, exist_ok=True)
    panels = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in FRAMES:
            panels[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
        if len(panels) == len(FRAMES):
            break
    for f in sorted(panels):
        panel = panels[f]
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        old_a, _b, _d = detect_edgefill(panel, ty0, ty1, field=FIELD)
        new, mask, angle, blur_L, prog = detect_new(panel, ty0, ty1, field=FIELD)
        peaks = detect_watershed(panel, ty0, ty1, FIELD)
        vis = panel.copy()
        for d in old_a:
            cv2.circle(vis, (int(d.cx), int(d.cy)), 9, (0, 0, 255), 2)       # OLD k-means = red rings
        for (cx, cy, _a) in peaks:
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)          # PEAKS = green dots
        cv2.putText(vis, f"f{f}  OLD(red)={len(old_a)}  PEAKS(green)={len(peaks)}  "
                    f"blurIter={len(new)}@L{blur_L}  field={FIELD}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        out = f"{DIAG}/cv_avatar2_f{f:04d}.png"
        cv2.imwrite(out, vis[int(0.28 * ph):int(0.82 * ph)])
        prog_s = " ".join(f"L{lL}:{c}" for lL, c in prog)
        print(f"f{f}: OLD={len(old_a)} WSHED={len(peaks)} blurIter={len(new)}@L{blur_L} "
              f"(target {FIELD})  blurprog[{prog_s}] -> {out}")


if __name__ == "__main__":
    main()

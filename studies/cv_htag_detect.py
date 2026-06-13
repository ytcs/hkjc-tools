"""R&D study — CLASS-SEGMENTATION horse detection for the AVV render (horse-tagging milestone).

Pixel study finding (2026-06-12): the render palette separates per-pixel into
  turf (green hue, lit or shadow-dark)  |  body (brown/tan hue)  |  white (bright desat)
  vivid (saturated non-green silk)      |  dark (near-black: shadow cores + outlines)
and the AVATAR CORE = body + vivid + small-white is unique to horses: a cast shadow carries
ZERO core-class pixels, so segmentation rejects shadows BY CONSTRUCTION (no shape heuristics).
White furniture (rails / disclaimer text / painted markers) is excluded by GEOMETRY (thin or
long or large components), never by count-forcing.

Detector v1 (count-aware, never count-forced):
  1. class maps (turf hue read per frame);
  2. TRACK-SURFACE ROI = the filled largest green region (kills grandstand/apron/road furniture
     on rotated views — v0 leaked 54 dets on R08 f800);
  3. white_kept = white components that are compact AND not on a RAIL CURVE (per-frame rail fits
     reused from the rail milestone — the pooled structure that separates rail dashes from
     jockey whites, which are locally identical compact blobs);
  4. core = vivid | body | white_kept, closed lightly -> clusters inside the ROI;
  5. clusters SPLIT by silk markers (one jockey = one merged vivid/white blob; >=2 markers in
     one cluster = a bunched merge -> one det per marker, core px assigned to nearest marker).
Comparisons: detect_avatars (count-forcing baseline) + the 0113 silk-anchored shape detector.

Run:  PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_detect.py [R08 ...]
Overlays -> data_cache/diag/htag/<race>_fNNNNN.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "studies")

RACES = {
    "R08": ("data_cache/video/aerial_20260603_08.mp4", 11,
            [336, 360, 406, 600, 745, 800, 1080, 1400, 1700, 2000]),
    "R02A": ("data_cache/video/aerial_20260520_02.mp4", 12,
             [340, 400, 500, 630, 700, 900, 1100, 1400, 1800, 2200, 2500]),
    "R03": ("data_cache/video/aerial_20260603_03.mp4", 12, [300, 700, 1200, 1700]),
}
DIAG = "data_cache/diag/htag"

# ---- render-graphics constants (palette/geometry of the AVV product, not per-race tunables)
GREEN_TOL = 16        # hue distance from the per-frame modal turf hue that still counts as turf
SHADOW_V = 90         # turf darker than this = turf-in-shadow
WHITE_V, WHITE_S = 170, 60
VIVID_S, VIVID_V = 90, 80
BODY_S, BODY_V_LO, BODY_V_HI = 45, 35, 200
WHITE_MAX_EXT = 26    # a jockey-white blob is compact; longer = rail dash / text / furniture
WHITE_MAX_ELONG = 3.5
RAIL_EXCL = 5.0       # px: a white comp whose centroid sits this close to a rail curve = dash
CORE_MIN_PX = 14      # a real avatar core has at least this many core px
DET_AREA_MAX = 700    # px: avatar cores measure ~150-280 (max ~400 zoomed); bigger =
#                       furniture (the odometer minimap box segments at 985-3551 px)
CORE_CLOSE = 5        # px, light closing to join body+silk fragments of ONE avatar
MARKER_MIN_PX = 5     # a silk marker (jockey) blob is at least this big
ROI_CLOSE = 17        # px: closing that absorbs avatars/shadows as holes in the turf region
# The avatar is an ORIENTED ellipse along the travel direction (the local rail tangent):
# merging/assignment distances are elliptical — generous ALONG the body (helmet<->torso,
# head fragment<->jockey), strict ACROSS track (the adjacent lane is ~0.4 scale away).
SCALE_DEFAULT = 38.0  # avatar body length fallback (px)
TRACK_W = 215.0       # px: render track width; horses live within this of the racing rail,
#                       on the side OPPOSITE the fixed/infield rail (venue model, HKJC_SPECS
#                       §1) — the rendered grandstand starts just beyond the outer fence
MM_ALONG = 0.45       # marker-merge semi-axis, x scale (same jockey's helmet+torso)
MM_ACROSS = 0.22
AS_ALONG = 1.00       # fragment-assignment semi-axis, x scale (head fragment -> its jockey)
AS_ACROSS = 0.34


# The on-render odometer/minimap box is an OPAQUE fixed HUD element (aerial_layout's odo
# ROI covers its OCR text portion; the full box is a bit wider). Horses behind it are
# occluded — zeroing it before detection is honest (no horse is visible there) and kills
# its green-arrow/digit fragments, which carry dark adjacency and dodge the area cap.
# Panel-relative fractions of the 960-px render panel (box observed x~640-920, y~260-380).
ODO_BOX = (0.66, 0.97, 0.24, 0.36)   # x0/x1 fractions of panel width, y0/y1 of height


def mask_hud(band: np.ndarray) -> np.ndarray:
    """Return ``band`` with the opaque odometer/minimap HUD box zeroed."""
    x0, x1 = int(ODO_BOX[0] * band.shape[1]), int(ODO_BOX[1] * band.shape[1])
    y0, y1 = int(ODO_BOX[2] * 1080), int(ODO_BOX[3] * 1080)
    if y0 >= band.shape[0]:
        return band
    out = band.copy()
    out[y0:min(y1, band.shape[0]), x0:x1] = 0
    return out


def class_maps(band: np.ndarray):
    """Per-pixel class booleans for a track-band BGR image."""
    import cv2

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    turf_sel = (s > 50) & (v > 60)
    thue = int(np.bincount(np.asarray(h)[turf_sel].ravel(), minlength=180).argmax()) \
        if turf_sel.any() else 45
    d_hue = np.minimum(np.abs(h - thue), 180 - np.abs(h - thue))
    green = d_hue < GREEN_TOL
    white = (v > WHITE_V) & (s < WHITE_S)
    vivid = (~green) & (~white) & (s > VIVID_S) & (v > VIVID_V)
    body = (~green) & (~white) & (~vivid) & (((h <= 25) | (h >= 165)) & (s > BODY_S)) \
        & (v > BODY_V_LO) & (v < BODY_V_HI)
    dark = v < 45
    return dict(green=green, white=white, vivid=vivid, body=body, dark=dark, thue=thue)


def white_kept(white: np.ndarray, rail_pts: np.ndarray | None = None):
    """White components that are jockey-sized: compact, not rail dashes / text / furniture.

    ``rail_pts`` (Nx2 band coords) = the frame's fitted rail curves; a compact white comp ON a
    rail curve is a rail dash — locally indistinguishable from a jockey white, separated only by
    the pooled rail structure (the rail-milestone meta-law applied here).
    """
    import cv2

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    keep = np.zeros_like(white, bool)
    for i in range(1, n):
        w0, h0 = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w0, h0) > WHITE_MAX_EXT:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        if max(w0, h0) / max(1, min(w0, h0)) > WHITE_MAX_ELONG and area < 12:
            continue                                    # thin sliver = rail dash fragment
        if rail_pts is not None and len(rail_pts):
            cx, cy = float(cent[i, 0]), float(cent[i, 1])
            d2 = (rail_pts[:, 0] - cx) ** 2 + (rail_pts[:, 1] - cy) ** 2
            if float(d2.min()) < RAIL_EXCL ** 2:
                continue                                # white dash on a rail curve
        keep |= lbl == i
    return keep


def turf_roi(green: np.ndarray):
    """Track-surface ROI: the filled largest green region (lit + shadow turf).

    Avatars, shadows and rail dashes are holes in the turf; closing + hole-filling absorbs
    them. Grandstand / apron / road furniture is OUTSIDE the turf region and drops out.
    """
    import cv2

    g = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_CLOSE, ROI_CLOSE)))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(g, 8)
    if n <= 1:
        return np.ones_like(green, bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = (lbl == big).astype(np.uint8)
    # fill holes: flood from the border on the complement
    inv = (1 - roi).astype(np.uint8)
    ff = inv.copy()
    mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 0)
    return (roi | ff).astype(bool)


def _avatar_scale(footprint: np.ndarray, in_track=None) -> float:
    """Provisional avatar length (px): low-percentile of footprint-blob extents. A LOW
    percentile (not median) resists furniture/merge blobs dominating when avatars are tiny
    in a zoomed-out bend (the median there reads the rail/text furniture, not the horses)."""
    import cv2

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), 8)
    lens = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12 or area > 4000:
            continue
        if in_track is not None and not in_track(float(cent[i, 0]), float(cent[i, 1])):
            continue
        ys, xs = np.where(lbl == i)
        p = np.column_stack([xs, ys]).astype(float)
        ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
        length = 4.0 * float(np.sqrt(max(ev[1], 1e-6)))
        if length / max(4.0 * float(np.sqrt(max(ev[0], 1e-6))), 1e-6) > 6.0:
            continue
        lens.append(length)
    if len(lens) < 3:
        return SCALE_DEFAULT
    return float(np.clip(np.percentile(lens, 35), 12.0, 60.0))


def _scale_from_markers(closed: np.ndarray, marks) -> float | None:
    """Refined avatar length: median PCA-extent of the core components that CONTAIN a marker
    (those are real avatars — furniture has no jockey marker, so it cannot inflate the scale
    even when the camera pulls back and the avatars shrink to a few px)."""
    import cv2

    if len(marks) < 3:
        return None
    n, lbl, _stats, _c = cv2.connectedComponentsWithStats(closed.astype(np.uint8), 8)
    seen, lens = set(), []
    for mx, my, _a in marks:
        ix, iy = int(mx), int(my)
        if not (0 <= iy < lbl.shape[0] and 0 <= ix < lbl.shape[1]):
            continue
        lid = int(lbl[iy, ix])
        if lid == 0 or lid in seen:
            continue
        seen.add(lid)
        ys, xs = np.where(lbl == lid)
        p = np.column_stack([xs, ys]).astype(float)
        ev = np.linalg.eigvalsh(np.cov((p - p.mean(0)).T))
        lens.append(4.0 * float(np.sqrt(max(ev[1], 1e-6))))
    if len(lens) < 3:
        return None
    return float(np.clip(np.median(lens), 12.0, 60.0))


def _angle_at(rail_pts: np.ndarray | None, x: float, y: float, k: int = 15) -> float:
    """Local travel direction = PCA tangent of the k nearest rail points (0.0 without rails)."""
    if rail_pts is None or len(rail_pts) < k:
        return 0.0
    d2 = (rail_pts[:, 0] - x) ** 2 + (rail_pts[:, 1] - y) ** 2
    near = rail_pts[np.argsort(d2)[:k]].astype(float)
    c = near - near.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0]))


class InTrack:
    """Venue-model in-track test: within TRACK_W of the racing rail, on the HORSE side.

    Side orientation, in preference order, all PER-QUERY-LOCAL (an SVD normal's raw sign
    is arbitrary and cannot be compared across queries):
      1. away from the nearest FIXED/infield rail point (config C: both rails seen);
      2. the TURF-RICH side of the racing rail (config A merged band: the rendered
         grandstand/crowd abuts the rail on the non-turf side — f400's crowd dots pass
         every local appearance gate, only this geometry separates them);
      3. both sides turf (track + turf infield) -> distance-only. Honest degradation.
    """

    def __init__(self, racing: np.ndarray | None, fixed: np.ndarray | None,
                 green: np.ndarray | None = None):
        self.racing = racing if racing is not None and len(racing) >= 15 else None
        self.fixed = fixed if fixed is not None and len(fixed) else None
        self.green = green

    def _local(self, x: float, y: float):
        """Nearest racing point j and the local unit normal (sign unoriented)."""
        r = self.racing
        d2 = (r[:, 0] - x) ** 2 + (r[:, 1] - y) ** 2
        j = int(np.argmin(d2))
        near = r[np.argsort(d2)[:15]].astype(float)
        c = near - near.mean(0)
        _u, _s, vt = np.linalg.svd(c, full_matrices=False)
        return j, -float(vt[0, 1]), float(vt[0, 0])

    def _turf_frac(self, j: int, nx: float, ny: float, sgn: float) -> float:
        """Turf fraction sampled 25-55 px off the rail on side ``sgn`` near point j."""
        g = self.green
        h, w = g.shape
        r = self.racing
        hits, tot = 0, 0
        for jj in range(max(0, j - 20), min(len(r), j + 21), 10):
            for d in (25.0, 40.0, 55.0):
                px = int(r[jj, 0] + sgn * nx * d)
                py = int(r[jj, 1] + sgn * ny * d)
                if 0 <= px < w and 0 <= py < h:
                    tot += 1
                    hits += bool(g[py, px])
        return hits / tot if tot else 0.0

    def __call__(self, x: float, y: float) -> bool:
        if self.racing is None:
            if self.fixed is None:
                return True
            d2 = (self.fixed[:, 0] - x) ** 2 + (self.fixed[:, 1] - y) ** 2
            return bool(d2.min() < (1.15 * TRACK_W) ** 2)
        j, nx, ny = self._local(x, y)
        s = float((x - self.racing[j, 0]) * nx + (y - self.racing[j, 1]) * ny)
        if abs(s) > TRACK_W:
            return False
        if abs(s) <= 14.0:
            return True                        # hugging the rail: side is moot
        if self.fixed is not None:             # (1) infield = toward the fixed rail
            fd2 = (self.fixed[:, 0] - self.racing[j, 0]) ** 2 \
                + (self.fixed[:, 1] - self.racing[j, 1]) ** 2
            fj = int(np.argmin(fd2))
            dot = nx * (self.racing[j, 0] - self.fixed[fj, 0]) \
                + ny * (self.racing[j, 1] - self.fixed[fj, 1])
            return s * np.sign(dot) > -14.0
        if self.green is not None:             # (2) horse side = the turf-rich side
            f_q = self._turf_frac(j, nx, ny, np.sign(s))
            f_o = self._turf_frac(j, nx, ny, -np.sign(s))
            if f_q < 0.5 <= f_o:
                return False                   # query sits on the non-turf (stands) side
            if f_o < 0.5 <= f_q:
                return True
        return True                            # (3) ambiguous: distance-only


def _silk_sig(lab_px: np.ndarray):
    """Silk colour signature: up to 2 dominant LAB colours of the det's jockey px (k-means;
    the MEAN of a halved silk is useless — red+white -> pink, journal 0112)."""
    import cv2

    px = np.asarray(lab_px, np.float32).reshape(-1, 3)
    if len(px) < 4:
        return px.mean(0)[None, :] if len(px) else np.zeros((0, 3), np.float32)
    kk = 2 if len(px) >= 14 else 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _c, lbls, ctr = cv2.kmeans(px, kk, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbls = lbls.ravel()
    order = np.argsort([-(lbls == j).sum() for j in range(kk)])
    return np.asarray([ctr[j] for j in order], np.float32)


def sig_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Min pairwise LAB distance between two silk signatures (tolerant to the 2nd colour)."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 1e9
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def _ell_d2(dx: float, dy: float, ang: float, a: float, b: float) -> float:
    """Squared elliptical distance: rotate (dx,dy) into the travel frame, scale axes."""
    ca, sa = np.cos(ang), np.sin(ang)
    u = dx * ca + dy * sa
    w = -dx * sa + dy * ca
    return (u / a) ** 2 + (w / b) ** 2


def _line_kernel(length: int, angle: float):
    """Binary line structuring element of ``length`` px oriented at ``angle`` (rad)."""
    import cv2

    nn = max(3, int(length) | 1)
    k = np.zeros((nn, nn), np.uint8)
    c = nn // 2
    dx, dy = np.cos(angle), np.sin(angle)
    cv2.line(k, (int(round(c - dx * c)), int(round(c - dy * c))),
             (int(round(c + dx * c)), int(round(c + dy * c))), 1, 1)
    return k


def _dominant_tangent(rail_pts: np.ndarray | None) -> float:
    """Travel-direction angle (rad) = PCA major axis of the rail points (0.0 without rails)."""
    if rail_pts is None or len(rail_pts) < 15:
        return 0.0
    c = rail_pts.astype(float) - rail_pts.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0]))


def _farthest_k(pts, k: int) -> list[int]:
    """Indices of k mutually-farthest points (greedy farthest-point seeding), starting
    from the largest. Used to pick which markers anchor a genuine multi-horse merge."""
    if k >= len(pts):
        return list(range(len(pts)))
    xy = np.asarray([(p[0], p[1]) for p in pts], float)
    chosen = [int(np.argmax([p[2] for p in pts]))]   # largest marker first
    while len(chosen) < k:
        d = np.min([((xy[:, 0] - xy[c, 0]) ** 2 + (xy[:, 1] - xy[c, 1]) ** 2)
                    for c in chosen], axis=0)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    return chosen


def _markers(vivid: np.ndarray, wk: np.ndarray, scale: float,
             rail_pts: np.ndarray | None, in_track=None):
    """Per-jockey silk markers: vivid/white blobs merged ELLIPTICALLY (along-axis generous:
    helmet+torso of one jockey; across-axis strict: the adjacent lane stays separate)."""
    import cv2

    m = (vivid | wk).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    raw = [(float(cent[i, 0]), float(cent[i, 1]), int(stats[i, cv2.CC_STAT_AREA]))
           for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= MARKER_MIN_PX]
    if in_track is not None:
        raw = [r for r in raw if in_track(r[0], r[1])]
    raw.sort(key=lambda r: -r[2])
    aa, bb = MM_ALONG * scale, MM_ACROSS * scale
    merged: list[list] = []
    for cx, cy, a in raw:
        ang = _angle_at(rail_pts, cx, cy)
        hit = next((g for g in merged
                    if _ell_d2(g[0] / g[2] - cx, g[1] / g[2] - cy, ang, aa, bb) < 1.0), None)
        if hit is None:
            merged.append([cx * a, cy * a, a])
        else:
            hit[0] += cx * a
            hit[1] += cy * a
            hit[2] += a
    return [(g[0] / g[2], g[1] / g[2], g[2]) for g in merged]


def detect_core(band: np.ndarray, rails: dict | None = None, scale_override: float | None = None):
    """Segment avatar-core detections in a track band. Returns (dets, debug).

    Markers (jockeys) are the instance authority; core clusters are evidence pools:
      * cluster with k>=2 markers  -> bunched merge, split px by nearest marker;
      * cluster with one marker    -> one detection;
      * cluster with none (a head/body fragment or a dark-silk horse) -> ATTACH to the
        nearest marker within the avatar ellipse, else stand alone (count-aware fallback).
    ``rails`` = dict(racing=Nx2, fixed=Mx2) band-coord rail curves (venue-model gating).
    dets = dict(cx, cy, area, n_vivid, n_white, n_body, marker, split) in BAND coords.
    """
    import cv2

    band = mask_hud(band)
    racing = rails.get("racing") if rails else None
    fixed = rails.get("fixed") if rails else None
    parts = [p for p in (racing, fixed) if p is not None and len(p)]
    rail_pts = np.vstack(parts) if parts else None

    m = class_maps(band)
    in_track = InTrack(racing, fixed, m["green"])
    roi = turf_roi(m["green"])
    wk = white_kept(m["white"], rail_pts)
    vivid_r, body_r, wk_r = m["vivid"] & roi, m["body"] & roi, wk & roi
    core = (vivid_r | body_r | wk_r).astype(np.uint8)
    # connect ONE avatar's fragments (silk + blaze + tail, split by the dark saddle) along
    # the TRAVEL axis before clustering — within-horse fragments align along travel; adjacent
    # horses are offset ACROSS-track, so an oriented close does not bridge lanes. A small
    # isotropic close then fills. (Replaces the isotropic close that left split-silk doubles.)
    scale0 = _avatar_scale(cv2.morphologyEx(
        core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (CORE_CLOSE, CORE_CLOSE))), in_track)
    ang0 = _dominant_tangent(rail_pts)
    closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _line_kernel(int(0.55 * scale0), ang0))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (CORE_CLOSE, CORE_CLOSE)))
    # provisional scale (low-percentile, furniture-resistant) locates markers; then refine
    # from marker-bearing blobs (the real avatars) — robust to the zoomed-out bend collapse.
    # A temporally-smoothed scale_override (the global, the zoom is smooth) overrides both.
    marks = _markers(vivid_r, wk_r, scale0, rail_pts, in_track)
    raw_scale = _scale_from_markers(closed, marks) or scale0
    scale = scale_override if scale_override is not None else raw_scale
    if abs(scale - scale0) > 0.25 * scale0:    # scale moved -> re-merge markers + close at it
        ang0 = _dominant_tangent(rail_pts)
        closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _line_kernel(int(0.55 * scale), ang0))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            (CORE_CLOSE, CORE_CLOSE)))
        marks = _markers(vivid_r, wk_r, scale, rail_pts, in_track)
    n, lbl, _stats, _cent = cv2.connectedComponentsWithStats(closed, 8)

    # AREA is the count authority, markers only LOCATE (a single white horse fragments into
    # 2-3 vivid/white blobs — marker count over-segments at the bunched start). Per cluster:
    # k = round(area / single_horse_area), clamped to [1, #inside markers].
    clusters = []                              # (pts, inside-marker indices, area)
    orphans = []
    for i in range(1, n):
        sel = lbl == i
        ys, xs = np.where(sel & (core > 0))
        if len(xs) < 6:
            continue
        pts = np.column_stack([xs, ys]).astype(float)
        inside = [j for j, (mx, my, _a) in enumerate(marks)
                  if 0 <= int(my) < sel.shape[0] and 0 <= int(mx) < sel.shape[1]
                  and sel[int(my), int(mx)]]
        if inside:
            clusters.append((pts, inside, len(xs)))
        else:
            orphans.append(pts)
    singles = [a for _p, ins, a in clusters if len(ins) == 1]
    single_area = (float(np.median(singles)) if len(singles) >= 3
                   else max(float(CORE_MIN_PX), 0.22 * scale * scale))

    slots = [dict(mx=mx, my=my, px=[], split=False) for mx, my, _a in marks]
    used = set()
    for pts, inside, area in clusters:
        k = int(round(area / single_area))
        k = max(1, min(k, len(inside)))
        if k < len(inside):                    # area authorizes fewer horses than markers
            chosen = _farthest_k([(marks[j][0], marks[j][1], marks[j][2]) for j in inside], k)
            chosen = [inside[c] for c in chosen]
        else:
            chosen = inside
        used.update(inside)                    # every inside marker is consumed
        if len(chosen) == 1:
            slots[chosen[0]]["px"].append(pts)
        else:
            mk = np.asarray([[marks[j][0], marks[j][1]] for j in chosen], float)
            owner = np.argmin(((pts[:, None, :] - mk[None, :, :]) ** 2).sum(-1), axis=1)
            for jj, j in enumerate(chosen):
                slots[j]["px"].append(pts[owner == jj])
                slots[j]["split"] = True
    n_marker_slots = len(marks)
    aa, bb = AS_ALONG * scale, AS_ACROSS * scale
    for pts in orphans:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ang = _angle_at(rail_pts, cx, cy)
        best, bd = None, 1.0
        for j, s in enumerate(slots):
            if "mx" not in s:
                continue
            d2 = _ell_d2(s["mx"] - cx, s["my"] - cy, ang, aa, bb)
            if d2 < bd:
                best, bd = j, d2
        if best is not None:
            slots[best]["px"].append(pts)
        elif len(pts) >= CORE_MIN_PX and in_track(cx, cy):
            orphan_det = dict(cx=cx, cy=cy, area=int(len(pts)), marker=False, split=False)
            orphan_det["pts"] = pts
            slots.append(orphan_det)

    dets = []
    core_b = core.astype(bool)
    for si, s in enumerate(slots):
        if "mx" in s:
            if not s["px"]:
                if si < n_marker_slots and si in used:
                    continue                   # absorbed into an over-segmented cluster
                # small/far avatar: its cluster fell under the comp floor, but the jockey
                # marker is real evidence — gather core px in a disc around it
                r = max(8, int(0.6 * scale))
                x0, x1 = max(0, int(s["mx"]) - r), min(band.shape[1], int(s["mx"]) + r + 1)
                y0, y1 = max(0, int(s["my"]) - r), min(band.shape[0], int(s["my"]) + r + 1)
                ys, xs = np.where(core_b[y0:y1, x0:x1])
                if len(xs) < 6:
                    continue
                pts = np.column_stack([xs + x0, ys + y0]).astype(float)
                d2 = (pts[:, 0] - s["mx"]) ** 2 + (pts[:, 1] - s["my"]) ** 2
                pts = pts[d2 < r * r]
                if len(pts) < 6:
                    continue
                s["px"].append(pts)
            pts = np.vstack(s["px"])
            if len(pts) < max(6, CORE_MIN_PX // 2):   # marker-backed floor is lower
                continue
            dets.append(dict(cx=float(pts[:, 0].mean()), cy=float(pts[:, 1].mean()),
                             area=int(len(pts)), marker=True, split=s["split"], pts=pts))
        else:
            dets.append(s)
    # fold sub-avatar fragment dets (nose blaze / split cloth) into their parent det:
    # elliptical agglomeration, smaller-into-larger; a nose-to-tail neighbour is ~1.0 scale
    # along and survives, a fragment is <=0.6 scale away and folds.
    dets.sort(key=lambda d: -d["area"])
    kept: list[dict] = []
    for d in dets:
        ang = _angle_at(rail_pts, d["cx"], d["cy"])
        parent = next((k for k in kept
                       if _ell_d2(k["cx"] - d["cx"], k["cy"] - d["cy"], ang,
                                  0.75 * scale, 0.30 * scale) < 1.0), None)
        if parent is None:
            kept.append(d)
        else:
            pts = np.vstack([parent["pts"], d["pts"]])
            parent.update(cx=float(pts[:, 0].mean()), cy=float(pts[:, 1].mean()),
                          area=int(len(pts)), pts=pts,
                          marker=parent["marker"] or d["marker"])
    dets = kept
    lab_img = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
    for d in dets:
        pts = d.pop("pts")
        ii = pts.astype(int)
        d["n_vivid"] = int(vivid_r[ii[:, 1], ii[:, 0]].sum())
        d["n_white"] = int(wk_r[ii[:, 1], ii[:, 0]].sum())
        d["n_body"] = int(body_r[ii[:, 1], ii[:, 0]].sum())
        silk_sel = vivid_r[ii[:, 1], ii[:, 0]] | wk_r[ii[:, 1], ii[:, 0]]
        d["sig"] = _silk_sig(lab_img[ii[silk_sel, 1], ii[silk_sel, 0]])
        # horse-evidence gate: a det that is PURE white must abut the dark/brown horse
        # (painted "50M"-style turf glyphs are white strokes on clean turf — no horse).
        d["big"] = d["area"] > DET_AREA_MAX
        if d["n_vivid"] == 0 and d["n_body"] == 0:
            x0 = max(0, int(pts[:, 0].min()) - 4)
            x1 = min(band.shape[1], int(pts[:, 0].max()) + 5)
            y0 = max(0, int(pts[:, 1].min()) - 4)
            y1 = min(band.shape[0], int(pts[:, 1].max()) + 5)
            nb = m["dark"][y0:y1, x0:x1] | m["body"][y0:y1, x0:x1]
            d["horseless"] = int(nb.sum()) < 8
        else:
            d["horseless"] = False
    dets = [d for d in dets if not d["horseless"] and not d["big"]]
    # text-ROW filter: >=3 PURE-WHITE dets sharing a y-line = a disclaimer/HUD text row
    # (glyphs organize in rows; horses are never simultaneously pure-white AND collinear).
    # Drop every pure-white det within the row band (the second interleaved line too).
    pure = [d for d in dets if d["n_vivid"] == 0 and d["n_body"] == 0]
    row_ys = []
    for d in pure:
        peers = [e for e in pure if abs(e["cy"] - d["cy"]) <= 8.0]
        if len(peers) >= 3:
            row_ys.append(float(np.median([e["cy"] for e in peers])))
    if row_ys:
        dets = [d for d in dets
                if not (d["n_vivid"] == 0 and d["n_body"] == 0
                        and any(abs(d["cy"] - ry) <= 28.0 for ry in row_ys))]
    return dets, dict(core=core, closed=closed, roi=roi, marks=marks, scale=scale,
                      raw_scale=raw_scale, **m)


def _overlay(panel, ty0, ty1, dets, av_dets, sh_dets, tag, field):
    import cv2

    vis = panel.copy()
    cv2.line(vis, (0, ty0), (vis.shape[1], ty0), (255, 255, 0), 1)
    cv2.line(vis, (0, ty1), (vis.shape[1], ty1), (255, 255, 0), 1)
    for d in av_dets:
        cv2.drawMarker(vis, (int(d.cx), int(d.cy)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 1)
    for d in sh_dets:
        cv2.drawMarker(vis, (int(d.cx), int(d.cy)), (255, 0, 255), cv2.MARKER_SQUARE, 12, 1)
    for d in dets:
        x, y = int(d["cx"]), int(d["cy"] + ty0)
        col = ((0, 255, 255) if d.get("split")
               else (0, 255, 0) if d.get("marker") else (255, 255, 0))
        cv2.circle(vis, (x, y), 11, col, 2)
        cv2.putText(vis, str(d["area"]), (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.putText(vis, f"{tag} field={field} seg={len(dets)} shape={len(sh_dets)} "
                f"avat={len(av_dets)}  (green=seg, magenta=shape, red=detect_avatars)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out = f"{DIAG}/{tag}.png"
    cv2.imwrite(out, vis)
    return out


def main():
    import av

    from cv_avatar_shape import _detect_shape
    from hkjc.vision.aerial_layout import legend_top, render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars

    os.makedirs(DIAG, exist_ok=True)
    only = sys.argv[1:] or list(RACES)
    for tag in only:
        vid, field, frames = RACES[tag]
        want, mx = set(frames), max(frames)
        rails = {}
        rp = f"/tmp/htag_rails_{tag}.npz"
        if os.path.exists(rp):
            z = np.load(rp)
            for k in z.files:
                f, name = k.rsplit("_", 1)
                rails.setdefault(int(f), {})[name] = z[k]
        print(f"[{tag}] field={field} rail-frames={len(rails)}")
        for i, fr in enumerate(av.open(vid).decode(video=0)):
            if i in want:
                panel = fr.to_ndarray(format="bgr24")[:, 960:]
                if not render_active(panel):
                    print(f"  f{i:05d} INACTIVE")
                else:
                    # full panel down to the legend strip when rails bound the track
                    # (venue-model side test); without rails fall back to the fixed band
                    # (a fixed cut clips the start-phase field — R03 f300 sits ABOVE
                    # 0.42h — but an open panel without rails admits the backdrop).
                    rd = rails.get(i)
                    if rd and sum(len(v) for v in rd.values()):
                        ty0, ty1 = 0, legend_top(panel) - 2
                        off = np.array([0.0, ty0], np.float32)
                        rd = {k: (v - off if len(v) else v) for k, v in rd.items()}
                    else:
                        ty0, ty1 = track_band(panel)
                        rd = None
                    dets, _dbg = detect_core(panel[ty0:ty1], rd)
                    tb0, tb1 = track_band(panel)   # baselines keep their own band
                    try:
                        av_dets, _ = detect_avatars(panel, tb0, tb1, field)
                    except Exception:
                        av_dets = []
                    try:
                        sh_dets, _ = _detect_shape(panel, tb0, tb1, field)
                    except Exception:
                        sh_dets = []
                    out = _overlay(panel, ty0, ty1, dets, av_dets, sh_dets,
                                   f"{tag}_f{i:05d}", field)
                    print(f"  f{i:05d} seg={len(dets):2d} shape={len(sh_dets):2d} "
                          f"avat={len(av_dets):2d} field={field} "
                          f"scale={_dbg['scale']:.0f} marks={len(_dbg['marks'])}  -> {out}")
            if i >= mx:
                break


if __name__ == "__main__":
    main()

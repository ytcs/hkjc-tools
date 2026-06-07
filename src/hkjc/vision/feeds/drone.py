"""Drone feed — real moving high camera, lateral ORDERING + cross-feed redundancy.

A panning/zooming real camera has no global longitudinal axis, but its near-static STRAIGHT
windows admit the stable-rail recipe: fit ONE rail line per window (not per frame), measure
each horse's ground-contact perpendicular offset, scale by the corridor width. Anchored on the
shared clock by two real-time events (gate jump, photo-finish cut) -> affine vid->race_t.

Rail primitives come from ``geometry_cv`` (pooled-window stable rail). cv2 + av lazy.
See ``docs/CV_RECON.md`` §2 (feeds), §3 (clock).
"""
from __future__ import annotations

import numpy as np

from . import FeedContext
from ..geometry_cv import fit_stable_rail, rail_perp


# --------------------------------------------------------------------------- shared clock
def vid_to_race_t(vid_t, gate_vid_t: float, finishcut_vid_t: float,
                  winner_finish_race_t: float):
    """Affine drone video-time -> ``race_t`` from the gate-jump and photo-finish-cut anchors."""
    v0, v1 = gate_vid_t, finishcut_vid_t
    r0, r1 = 0.0, winner_finish_race_t
    return r0 + (np.asarray(vid_t, float) - v0) * (r1 - r0) / (v1 - v0)


def race_t_to_arclen(race_t, section_times: np.ndarray, section_bounds: np.ndarray):
    """Shared ``race_t`` -> arclen (m run) via the official cumulative-section curve."""
    return np.interp(np.asarray(race_t, float), section_times, section_bounds)


# --------------------------------------------------------------------------- turf / rail / detect
def turf_mask(img: np.ndarray):
    """Largest connected turf region (boolean uint8) + its modal hue, or ``(None, hue)``."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sel = (ss > 40) & (vv > 40) & (hh > 30) & (hh < 80)
    if sel.sum() < 1000:
        return None, None
    turf_hue = int(np.median(hh[sel]))
    green = ((np.abs(hh.astype(int) - turf_hue) <= 15) & (ss > 30) & (vv > 40)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lbl, st, _ct = cv2.connectedComponentsWithStats(green, 8)
    if n < 2:
        return None, turf_hue
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lbl == big).astype(np.uint8), turf_hue


def rail_points(turf: np.ndarray, field_xy, side_band: int = 260):
    """Inner-rail boundary points near the field (turf-mask edge). ``(N, 2)`` or None."""
    import cv2

    cnts, _ = cv2.findContours(turf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if not field_xy:
        return None
    fx = np.array([p[0] for p in field_xy])
    fy = np.array([p[1] for p in field_xy])
    d2 = ((cnt[:, None, 0] - fx[None, :]) ** 2 + (cnt[:, None, 1] - fy[None, :]) ** 2).min(axis=1)
    near = cnt[d2 < side_band ** 2]
    return near if len(near) >= 12 else None


def corridor_px(turf: np.ndarray):
    """Perpendicular turf-corridor width in px via the distance-transform ridge (== track width)."""
    import cv2

    dt = cv2.distanceTransform(turf, cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    if vals.size < 50:
        return None
    half = float(np.percentile(vals, 99.0))
    return 2.0 * half if half > 5 else None


def detect_field(img: np.ndarray, turf_lock):
    """Field horse-sprites + their GROUND-CONTACT (feet/shadow base, not body centroid).

    Returns ``(keep, turf_lock, turf)`` where each kept sprite is
    ``(cx, cy, gc_x, gc_y, area, silk_lab)`` and the silk LAB is the high-chroma mean (the
    identity colour anchor). ``turf_lock`` is the EMA-smoothed turf hue carried across frames.
    """
    import cv2

    turf, turf_hue = turf_mask(img)
    if turf is None:
        return [], turf_lock, None
    turf_lock = turf_hue if turf_lock is None else int(0.7 * turf_lock + 0.3 * turf_hue)
    field_region = cv2.erode(turf, np.ones((9, 9), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (np.abs(hh.astype(int) - turf_lock) <= 13) & (ss > 35) & (vv > 40)
    nong = ((~green) & (vv > 30)).astype(np.uint8)
    sprites = cv2.bitwise_and(nong, field_region)
    sprites = cv2.morphologyEx(sprites, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sprites = cv2.morphologyEx(sprites, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _l2, s2, c2 = cv2.connectedComponentsWithStats(sprites, 8)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cand = []
    for i in range(1, n):
        a = int(s2[i, cv2.CC_STAT_AREA])
        bw, bh = int(s2[i, cv2.CC_STAT_WIDTH]), int(s2[i, cv2.CC_STAT_HEIGHT])
        if not (18 <= a <= 2000 and bw < 80 and bh < 80):
            continue
        cx, cy = float(c2[i][0]), float(c2[i][1])
        bx, by = int(s2[i, cv2.CC_STAT_LEFT]), int(s2[i, cv2.CC_STAT_TOP])
        gc_x, gc_y = cx, float(by + bh)   # ground contact = sprite's lowest extent
        patch = lab_img[by:by + bh, bx:bx + bw].reshape(-1, 3).astype(float)
        chroma = np.abs(patch[:, 1] - 128.0) + np.abs(patch[:, 2] - 128.0)
        k = max(3, int(0.3 * len(patch)))
        mlab = patch[np.argsort(chroma)[-k:]].mean(axis=0)
        cand.append((cx, cy, gc_x, gc_y, a, (float(mlab[0]), float(mlab[1]), float(mlab[2]))))
    if not cand:
        return [], turf_lock, turf
    xy = np.array([(c[0], c[1]) for c in cand])
    d = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
    keep = [cand[i] for i in range(len(cand)) if d[i] < 400]
    return keep, turf_lock, turf


# --------------------------------------------------------------------------- identity by silk LAB
def assign_number(lab, anchors: dict, max_d: float = 26.0) -> tuple:
    """Assign a sprite's silk-LAB to the nearest number anchor; ``(num|None, dist)``."""
    best, bd = None, 1e9
    for num, al in anchors.items():
        d = float(np.linalg.norm(np.array(lab) - al))
        if d < bd:
            bd, best = d, num
    return (best, bd) if bd <= max_d else (None, bd)


# --------------------------------------------------------------------------- window extraction
def collect_frames(ctx: FeedContext, window: tuple[float, float, int]):
    """Decode the drone video, collect ``(vid_t, BGR)`` over ``[t0, t1]`` at the given stride."""
    import av

    t0, t1, stride = window
    fps = ctx.fps or 25.0
    c = av.open(ctx.video)
    out = []
    for i, fr in enumerate(c.decode(video=0)):
        t = i / fps
        if t < t0:
            continue
        if t > t1:
            break
        if i % stride == 0:
            out.append((t, fr.to_ndarray(format="bgr24")))
    c.close()
    return out


def extract_window(frames, anchors: dict, track_width_m: float) -> tuple[list, dict]:
    """Numbered ground-contact ``y_lat`` for a clean window via a stable per-window rail.

    Pools rail-boundary points across the window, fits ONE rail, then per frame projects each
    sprite's ground contact perpendicular to it, scaled by the per-frame corridor == track width,
    and assigns a number by nearest silk-LAB anchor. Returns ``(rows, diag)``.
    """
    turf_lock = None
    per_frame = []
    rail_pool = []
    for vid_t, img in frames:
        blobs, turf_lock, turf = detect_field(img, turf_lock)
        if not blobs or turf is None:
            continue
        field_xy = [(b[0], b[1]) for b in blobs]
        rp = rail_points(turf, field_xy)
        cpx = corridor_px(turf)
        if rp is not None:
            rail_pool.append(rp)
        per_frame.append((vid_t, blobs, rp, cpx))
    if len(rail_pool) < 3 or not per_frame:
        return [], {"error": "insufficient rail/field"}
    rail = fit_stable_rail(rail_pool)
    rows = []
    for vid_t, blobs, _rp, cpx in per_frame:
        if cpx is None or cpx < 50:
            continue
        mpp = track_width_m / cpx
        for (cx, cy, gcx, gcy, _area, lab) in blobs:
            num, labd = assign_number(lab, anchors)
            if num is None:
                continue
            perp_px = rail_perp(rail, gcx, gcy)
            y_lat = min(max(abs(perp_px) * mpp, 0.0), track_width_m)
            perp_body = rail_perp(rail, cx, cy)
            rows.append({
                "vid_t": round(vid_t, 3), "number": num, "y_lat": round(float(y_lat), 3),
                "y_lat_body": round(float(min(max(abs(perp_body) * mpp, 0.0), track_width_m)), 3),
                "lab_d": round(labd, 1), "m_per_px": round(mpp, 4),
                "silk_l": round(lab[0], 1), "silk_a": round(lab[1], 1), "silk_b": round(lab[2], 1),
            })
    return rows, {"rail_straight": round(rail["straight"], 3),
                  "rail_across_std": round(rail["across_std"], 1),
                  "n_rail_frames": len(rail_pool), "n_rows": len(rows)}


def census_straightness(ctx: FeedContext, stride: int = 25,
                        finishcut_vid_t: float | None = None) -> list[tuple]:
    """Per sampled frame, classify STRAIGHT vs BEND via the rail-points straightness score.

    A coarse single-frame pass returning ``[(vid_t, straight|None, across_std|None), ...]`` up to
    the photo-finish cut (after which the geometry is discontinuous). Place on the shared clock
    with :func:`vid_to_race_t`.
    """
    import av

    fps = ctx.fps or 25.0
    turf_lock = None
    out = []
    c = av.open(ctx.video)
    for i, fr in enumerate(c.decode(video=0)):
        if i % stride:
            continue
        t = i / fps
        if finishcut_vid_t is not None and t > finishcut_vid_t:
            break
        img = fr.to_ndarray(format="bgr24")
        blobs, turf_lock, turf = detect_field(img, turf_lock)
        if not blobs or turf is None:
            out.append((t, None, None))
            continue
        rp = rail_points(turf, [(b[0], b[1]) for b in blobs])
        if rp is None or len(rp) < 12:
            out.append((t, None, None))
            continue
        r = fit_stable_rail([rp])
        out.append((t, r["straight"], r["across_std"]))
    c.close()
    return out

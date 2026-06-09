"""R&D study — SHAPE / APPEARANCE-TEMPLATE avatar detection for the Aerial Virtual View (AVV).

Hypothesis (user's idea, journal refs 0111/0112): the AVV horse avatar has a DISTINCTIVE,
visually-consistent SHAPE across frames and races — a dark/tan elongated horse body (~25-45 px
long at this zoom) with a small BRIGHT, SATURATED jockey-silk patch on top, oriented along the
direction of travel, on bright green turf. We try to detect avatars by matching this shape /
appearance DIRECTLY (oriented matched-template over the rail tangent), WITHOUT the legend strip.

Why it might beat the alternatives:
  * the current ``detect_avatars`` (turf-suppress -> Canny -> fill -> rail ROI -> count-FORCING)
    MISSES horses when its ROI is mis-sized and FABRICATES detections by splitting blobs to hit
    the field size (right count, wrong positions);
  * a naive dark-body detector EXPLODES on shadows (R02A: cast shadows look like dark bodies);
  * silk-only (journal 0112) is shadow-invariant but only ~33-64% recall (dark/low-chroma silks
    missed) and the legend only lists ~8 runners.

A whole-avatar SHAPE score that REQUIRES a horse-shaped dark/tan body AND rewards a bright silk
patch on top can (a) reject shadows (a shadow has a dark streak but no silk and the wrong, longer
fill), and (b) still fire on dark-silk horses (the body term carries them even when the silk is
faint).

STUDY ONLY (disposable scratch). Does NOT touch production code: imports ``detect_avatars`` /
``track_band`` / ``render_active`` for a side-by-side comparison; everything else is local and
derived ADAPTIVELY from the frame (turf hue read from data, avatar scale estimated from the blob
size distribution, orientation from the rail tangent) so it degrades gracefully under style drift.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_shape.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "studies")

# --------------------------------------------------------------------------- test set
# (video, field size, [frame indices]). Decode once per race, break after the max index.
RACES = {
    "R08": ("data_cache/video/aerial_20260603_08.mp4", 11, [336, 406, 600, 745, 1080]),
    "R03": ("data_cache/video/aerial_20260603_03.mp4", 12, [700]),
    "R02A": ("data_cache/video/aerial_20260520_02.mp4", 12, [900]),
}
DIAG = "data_cache/diag"

# --------------------------------------------------------------------------- graphics constants (NOT per-race)
# A rendered avatar body is ~25-45 px long / ~14-22 px wide at this zoom; the silk patch ~6-12 px.
# Cast shadows are LONGER (often >1.4x body) and carry no vivid silk. All thresholds are graphics
# properties of the render, read against the turf hue measured per frame (never a per-race table).
SILK_SAT_MIN = 95       # a silk patch is vivid ...
SILK_VAL_MIN = 80       # ... and bright (rules out the dark body AND the cast shadow)
SILK_AREA_MIN = 3       # px^2
SILK_AREA_MAX = 110     # px^2 (bigger = merged silks / furniture)
NMS_FRAC = 0.55         # NMS radius as a fraction of avatar scale


@dataclass
class ShapeDet:
    """One avatar detected by the shape-template route (panel-relative px)."""

    cx: float
    cy: float
    angle: float        # body orientation (rad), == local rail tangent
    score: float        # combined body+silk shape score in [0, ~2]
    has_silk: bool      # did a vivid silk patch anchor / confirm this detection


# --------------------------------------------------------------------------- turf / channel maps
def _turf_hue(hsv: np.ndarray) -> int:
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    if not turf.any():
        return 45
    return int(np.bincount(h[turf].ravel(), minlength=180).argmax())


def _channels(band: np.ndarray):
    """Return (is_turf, body, silk) boolean maps for a track band.

    body = non-turf, non-bright-furniture pixels (the dark/tan horse + ALSO cast shadows);
    silk = vivid bright non-turf pixels (the jockey patch ONLY — never a shadow).
    """
    import cv2

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    thue = _turf_hue(hsv)
    d_hue = np.minimum(np.abs(h - thue), 180 - np.abs(h - thue))
    is_turf = (d_hue < 16) & (s > 30) & (v > 45)
    bright = (v > 180) & (s < 55)                       # rail bar / painted markers / overlay
    body = (~is_turf) & (~bright)
    silk = (~is_turf) & (s > SILK_SAT_MIN) & (v > SILK_VAL_MIN)
    return is_turf, body.astype(np.uint8), silk.astype(np.uint8)


# --------------------------------------------------------------------------- adaptive scale + orientation
def _rail_tangent(panel: np.ndarray, ty0: int, ty1: int) -> float:
    """Local rail tangent angle (rad) = the orientation prior for the avatars (they travel along it)."""
    try:
        from hkjc.vision.feeds.aerial_rail import enumerate_lines, line_candidates
    except Exception:
        return 0.0
    lines = sorted(
        (ln for ln in enumerate_lines(line_candidates(panel, ty0, ty1), 2)
         if 1.0 <= ln["coverage"] <= 5.0),
        key=lambda ln: -ln["coverage"],
    )[:2]
    if not lines:
        return 0.0
    u = lines[0]["frame"][1]
    a = float(np.arctan2(u[1], u[0]))
    # fold to the travel direction (avatars run roughly left->right): keep |angle| < pi/2
    if a > np.pi / 2:
        a -= np.pi
    elif a < -np.pi / 2:
        a += np.pi
    return a


def _avatar_scale(body: np.ndarray, angle: float = 0.0) -> float:
    """Estimate the avatar body LENGTH (px) adaptively from the body-blob size distribution.

    The dark/tan body fragments under turf suppression, so first CLOSE the map a little along the
    travel axis to reconnect each avatar before measuring. Bodies + shadows both survive; a single
    avatar body is among the SHORTER coherent components, so the lower-central principal-axis
    length is a robust scale. Falls back to a graphics default if the band is degenerate.
    """
    import cv2

    m = cv2.morphologyEx(body, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    lens = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 30 or area > 4000:
            continue
        ys, xs = np.where(lbl == i)
        p = np.column_stack([xs, ys]).astype(float)
        cov = np.cov((p - p.mean(0)).T)
        ev = np.linalg.eigvalsh(cov)
        length = 4.0 * float(np.sqrt(max(ev[1], 1e-6)))
        minor = 4.0 * float(np.sqrt(max(ev[0], 1e-6)))
        if length / max(minor, 1e-6) > 6.0:           # skip obvious rail/shadow streaks
            continue
        lens.append(length)
    if len(lens) < 3:
        return 34.0
    # bodies cluster shorter than body+shadow merges; take the lower-central tendency
    return float(np.clip(np.median(lens), 26.0, 60.0))


# --------------------------------------------------------------------------- oriented shape score
def _body_components(body: np.ndarray, scale: float, angle: float):
    """Reconnect each avatar's fragments into horse-shaped blobs and return their shape descriptors.

    The dark/tan body fragments under turf suppression; closing along the RAIL TANGENT (the travel
    axis) glues each avatar's head/body/tail back into one oriented blob without bridging neighbours
    across-track. Returns a list of dicts with centroid, principal length/width, aspect, area.
    """
    import cv2

    closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, _line_kernel(int(scale * 0.7), angle))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(closed, 8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(18, int(0.25 * scale * scale * 0.4)):
            continue
        ys, xs = np.where(lbl == i)
        p = np.column_stack([xs, ys]).astype(float)
        cov = np.cov((p - p.mean(0)).T)
        ev = np.linalg.eigvalsh(cov)
        major = float(np.sqrt(max(ev[1], 1e-6)))
        minor = float(np.sqrt(max(ev[0], 1e-6)))
        comps.append({
            "cx": float(cent[i, 0]), "cy": float(cent[i, 1]), "area": area,
            "length": 4.0 * major, "width": 4.0 * minor,
            "aspect": major / max(minor, 1e-6), "mask": (lbl == i),
        })
    return comps


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


def _silk_blobs(silk: np.ndarray):
    """Vivid silk patch centroids (the shadow-immune anchor). Returns list of (cx, cy)."""
    import cv2

    m = cv2.morphologyEx(silk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if SILK_AREA_MIN <= a <= SILK_AREA_MAX:
            out.append((float(cent[i, 0]), float(cent[i, 1])))
    return out


def _detect_shape(panel: np.ndarray, ty0: int, ty1: int, field: int):
    """Whole-avatar SHAPE-descriptor detection. Returns (dets, debug).

    Each candidate is scored on three SHAPE/APPEARANCE cues that together describe the avatar and
    separate it from a cast shadow:
      * SILK  — a vivid bright silk patch inside the body footprint (a shadow has none);
      * LENGTH — a horse body is ~`scale` px long; a cast shadow is markedly LONGER (penalty);
      * ASPECT — a horse body is a stubby ellipse (aspect ~1.5-3.5); the rail-shadow band and
        long streaks are very elongated (rejected).
    Silk patches act as high-recall anchors (recall bright silks even when their body fragments);
    horse-shaped body components with no silk are kept at a lower score (recall DARK-silk runners);
    long/thin shadow components are rejected. Greedy NMS then yields one detection per avatar; the
    field count caps low-score extras (count-AWARE, never count-forced — no peaks are invented).
    """
    band = panel[ty0:ty1]
    _is_turf, body, silk = _channels(band)
    angle = _rail_tangent(panel, ty0, ty1)
    scale = _avatar_scale(body, angle)

    comps = _body_components(body, scale, angle)
    silks = _silk_blobs(silk)
    half = 0.6 * scale

    cands = []
    used_silk = [False] * len(silks)
    for comp in comps:
        # is this component a plausible horse body (not the rail-shadow band / a long streak)?
        too_long = comp["length"] > 1.9 * scale
        too_thin = comp["aspect"] > 5.0
        too_big = comp["area"] > 6.0 * scale * scale
        # any silk patch sitting inside the body footprint -> real horse, shadow-immune
        sidx = _nearest_silk(comp["cx"], comp["cy"], silks, used_silk, half)
        has_silk = sidx is not None
        if has_silk:
            used_silk[sidx] = True
            cx, cy = silks[sidx]                       # silk centre is the crisp avatar anchor
        else:
            cx, cy = comp["cx"], comp["cy"]
        if (too_long or too_thin or too_big) and not has_silk:
            continue                                   # cast shadow / rail band -> reject
        # shape score: horse-length bonus, over-length penalty, silk bonus
        len_term = np.exp(-((comp["length"] - scale) / (0.7 * scale)) ** 2)
        asp_term = np.exp(-((comp["aspect"] - 2.3) / 1.6) ** 2)
        score = 0.5 * len_term + 0.3 * asp_term + (0.9 if has_silk else 0.0)
        cands.append((cx, cy, score, has_silk))
    # silk patches with NO body component (bunched / fragmented body) are still real horses
    for j, (sx, sy) in enumerate(silks):
        if not used_silk[j]:
            cands.append((sx, sy, 0.7, True))

    dets = _nms(cands, scale, ty0, angle)
    if len(dets) > field:
        dets = sorted(dets, key=lambda d: -d.score)[:field]
    debug = (band, body, silk, scale, angle, comps)
    return dets, debug


def _nearest_silk(cx, cy, silks, used, radius):
    """Index of the nearest unused silk patch within ``radius`` of (cx, cy), else None."""
    best, bd = None, radius ** 2
    for j, (sx, sy) in enumerate(silks):
        if used[j]:
            continue
        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
        if d2 < bd:
            best, bd = j, d2
    return best


def _nms(cands, scale, ty0, angle):
    """Greedy non-max suppression; silk-confirmed candidates win ties. Returns ShapeDet (panel)."""
    rad = max(8.0, scale * NMS_FRAC)
    cands = sorted(cands, key=lambda c: (-c[2], not c[3]))
    kept: list[tuple] = []
    for cx, cy, sc, silked in cands:
        hit = next((k for k in range(len(kept))
                    if (cx - kept[k][0]) ** 2 + (cy - kept[k][1]) ** 2 < rad ** 2), None)
        if hit is not None:
            if silked and not kept[hit][3]:
                kx, ky, ks, _ = kept[hit]
                kept[hit] = (kx, ky, ks, True)
            continue
        kept.append((cx, cy, sc, silked))
    return [ShapeDet(float(cx), float(cy + ty0), angle, float(sc), bool(k))
            for cx, cy, sc, k in kept]


# --------------------------------------------------------------------------- overlay + drivers
def _draw(panel, ty0, ty1, dets, av_dets, scale, tag):
    """Annotate: shape dets (green=silk-confirmed, orange=body-only) vs detect_avatars (red x)."""
    import cv2

    vis = panel.copy()
    cv2.line(vis, (0, ty0), (vis.shape[1], ty0), (255, 255, 0), 1)
    cv2.line(vis, (0, ty1), (vis.shape[1], ty1), (255, 255, 0), 1)
    for d in av_dets:
        x, y = int(d.cx), int(d.cy)
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    for d in dets:
        x, y = int(d.cx), int(d.cy)
        col = (0, 220, 0) if d.has_silk else (0, 165, 255)
        cv2.circle(vis, (x, y), int(scale * 0.5), col, 2)
        cv2.circle(vis, (x, y), 2, col, -1)
    cv2.putText(vis, f"{tag}  shape={len(dets)}  silk={sum(d.has_silk for d in dets)}"
                f"  detect_avatars={len(av_dets)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out = f"{DIAG}/shape_{tag}.png"
    cv2.imwrite(out, vis)
    return out


def _run_race(tag, vid, field, frames):
    import av

    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars

    want = set(frames)
    mx = max(frames)
    rows = []
    for i, fr in enumerate(av.open(vid).decode(video=0)):
        if i in want:
            panel = fr.to_ndarray(format="bgr24")[:, 960:]
            if not render_active(panel):
                rows.append((i, "no-render", 0, 0, 0))
                if i >= mx:
                    break
                continue
            ty0, ty1 = track_band(panel)
            dets, dbg = _detect_shape(panel, ty0, ty1, field)
            av_dets, _ = detect_avatars(panel, ty0, ty1, field)
            scale = dbg[3]
            out = _draw(panel, ty0, ty1, dets, av_dets, scale, f"{tag}_f{i:05d}")
            nsilk = sum(d.has_silk for d in dets)
            rows.append((i, f"scale={scale:.0f} ang={np.degrees(dbg[4]):.0f}",
                         len(dets), nsilk, len(av_dets)))
            print(f"  [{tag} f{i:05d}] field={field} shape={len(dets)} "
                  f"silk-confirmed={nsilk} detect_avatars={len(av_dets)}  -> {out}")
        if i >= mx:
            break
    return rows


def main():
    only = os.environ.get("RACE")
    print("=== SHAPE-TEMPLATE avatar detection (study) ===")
    print("legend: green circle = silk-confirmed, orange = body-template only, red x = detect_avatars\n")
    for tag, (vid, field, frames) in RACES.items():
        if only and tag != only:
            continue
        if not os.path.exists(vid):
            print(f"  [{tag}] missing {vid}, skipping")
            continue
        print(f"[{tag}] field={field} frames={frames}")
        _run_race(tag, vid, field, frames)
        print()


if __name__ == "__main__":
    main()

"""Prep cache for the temporal rail tracker: per consecutive stride-2 frame, the candidate rail
pixels + field centroids + odometer + a downscaled panel. One slow decode -> fast tracker iteration.

Candidate pixels are extracted from a GENEROUS vertical band (not the fixed track_band, which crops a
curving rail at a bend) and kept model-free; the temporal tracker decides what is rail. Letterbox
(black border) is trimmed so 'spans the frame' means spans the CONTENT.

    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_prep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACES = {"R08": ("aerial_20260603_08", 1800.0, 11), "R03": ("aerial_20260603_03", 1200.0, 12),
         "R02A": ("aerial_20260520_02", 1500.0, 12)}
RACE = os.environ.get("RACE", "R08")
_vid, DIST, FIELD = RACES[RACE]
VID = f"data_cache/video/{_vid}.mp4"
OBS = f"data_cache/cv_joint_aerial_{RACE}_obs.npz"
LO, HI, STRIDE = int(os.environ.get("LO", 400)), int(os.environ.get("HI", 900)), 2
OUT = f"data_cache/_railprep_{RACE}_{LO}_{HI}.npz"
HALF = 960
DS = float(os.environ.get("DS", 0.5))   # downscale factor for cached overlay panels (memory)


def _content_box(panel):
    import cv2
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    cols = np.where(g.max(0) > 25)[0]
    rows = np.where(g.max(1) > 25)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, panel.shape[1], 0, panel.shape[0]
    return int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1


def _legend_top(panel):
    """Top row of the bottom SILK-LEGEND strip (a dense row of saturated boxes low in the panel).
    Returns the panel-y above which there is no legend, so callers can exclude its bright white numbers
    (which otherwise latch as rail candidates). Falls back to 0.80*h if no clear strip."""
    import cv2
    h = panel.shape[0]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    vivid = ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 110)).astype(np.uint8)
    rowsum = vivid.sum(1).astype(float)
    lo = int(0.65 * h)
    band = rowsum.copy()
    band[:lo] = 0
    if band.max() < 0.10 * panel.shape[1]:
        return int(0.80 * h)
    rows = np.where(band > 0.30 * band.max())[0]
    return int(rows.min()) if len(rows) else int(0.80 * h)


def _rail_pixels(panel, y0, y1):
    """Bright THIN rail pixels (model-free) in the vertical band [y0,y1). Panel coords.

    Brightness gate is TURF-RELATIVE (rail is brighter than turf by a margin) so a lighting-angle change
    that dims the whole scene (e.g. f812) doesn't lose the rail; floored so dark clutter can't pass."""
    import cv2
    band = panel[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    turf_g = float(np.median(gray[s > 40])) if (s > 40).any() else 120.0
    thr = max(150.0, turf_g + 28.0)
    white = ((gray > thr) & (s < 75)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, st, _c = cv2.connectedComponentsWithStats(white, 8)
    keep = np.zeros_like(white)
    for i in range(1, n):
        ext = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        if ext >= 8 and st[i, cv2.CC_STAT_AREA] / (ext + 1e-6) <= 6.0:
            keep[lbl == i] = 1
    ys, xs = np.where(keep)
    return np.column_stack([xs.astype(np.float32), ys.astype(np.float32) + y0])


def main() -> None:
    import av
    import cv2

    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars

    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)

    keys, railpx, field, odo, panels, boxes = [], [], [], [], [], []
    for i, fr in enumerate(av.open(VID).decode(video=0)):
        if i > HI:
            break
        if i < LO or (i % STRIDE):
            continue
        panel = fr.to_ndarray(format="bgr24")[:, HALF:]
        if not render_active(panel):
            continue
        ph = panel.shape[0]
        cx0, cx1, cy0, cy1 = _content_box(panel)
        y0, y1 = max(cy0, int(0.15 * ph)), min(cy1, _legend_top(panel) - 8)
        rp = _rail_pixels(panel, y0, y1)
        ty0, ty1 = track_band(panel)
        dets, _ = detect_avatars(panel, ty0, ty1, field=FIELD)
        fp = np.array([[d.cx, d.cy] for d in dets], np.float32) if dets else np.zeros((0, 2), np.float32)
        keys.append(i)
        railpx.append(rp)
        field.append(fp)
        odo.append(float(np.interp(i, FR[fin], ODO[fin])))
        boxes.append([cx0, cx1, cy0, cy1])
        small = cv2.resize(panel, None, fx=DS, fy=DS)
        panels.append(small)
    np.savez(OUT, keys=np.array(keys), railpx=np.array(railpx, dtype=object),
             field=np.array(field, dtype=object), odo=np.array(odo),
             boxes=np.array(boxes), panels=np.array(panels, dtype=object), ds=DS)
    print(f"{RACE}: {len(keys)} frames [{LO},{HI}] stride {STRIDE} -> {OUT} "
          f"(railpx median {int(np.median([len(r) for r in railpx]))})")


if __name__ == "__main__":
    main()

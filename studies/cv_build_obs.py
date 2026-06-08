"""Minimal odometer obs builder (frames + odo arclen) for the aerial-tracker validation.

The tracker only needs ``frames`` + ``odo`` (arclen-run, m) from the obs npz; the old
``cv_joint_aerial.py`` builder imports the retired ``mode_b``, so this rebuilds just that subset with
the PROMOTED ``aerial_ocr.read_odometer`` + ``aerial_layout.render_active``. ``RACE`` env selects.

A robust monotonic clean-up rejects OCR outliers: the field only advances, so arclen must rise ~linearly
in frame — fit a robust line, drop reads far off it, and enforce non-decreasing.

    RACE=R03 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_build_obs.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACES = {                                   # race -> (aerial video, distance m)
    "R08": ("data_cache/video/aerial_20260603_08.mp4", 1800.0),
    "R03": ("data_cache/video/aerial_20260603_03.mp4", 1200.0),
    "R06": ("data_cache/video/aerial_20260603_06.mp4", 1200.0),
    "R01": ("data_cache/video/aerial_20260603_01.mp4", 1650.0),
    "R02A": ("data_cache/video/aerial_20260520_02.mp4", 1500.0),   # A-config (single-rail-dominant)
}
RACE = os.environ.get("RACE", "R03")
VIDEO, DIST = RACES[RACE]
OUT = f"data_cache/cv_joint_aerial_{RACE}_obs.npz"
HALF = 960


def _odo_crop(panel):
    """Locate the odometer by DETECTING the green minimap box (top-right), not a fixed ROI.

    The "###M" text's position relative to the minimap varies per race (right of the box in R08, inside
    it on the left in R03), but the bright-green minimap box is always the largest green blob in the
    top-right — crop it (with margin) and let read_odometer find the digit+M pattern within.
    """
    import cv2
    h, w = panel.shape[:2]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    g = ((hh > 40) & (hh < 90) & (ss > 80) & (vv > 70)).astype(np.uint8)
    g[:int(0.10 * h), :] = 0
    g[int(0.5 * h):, :] = 0
    g[:, :int(0.55 * w)] = 0
    n, _lbl, st, _c = cv2.connectedComponentsWithStats(g, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax([st[k, cv2.CC_STAT_AREA] for k in range(1, n)]))
    if st[i, cv2.CC_STAT_AREA] < 1500:
        return None
    x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
    bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    return panel[max(0, y - int(0.4 * bh)):y + bh + 5, max(0, x - int(0.15 * bw)):x + bw + 5]


def _clean(frames, odo, dist):
    """Robust monotonic clean-up: drop OCR outliers off the linear arclen-vs-frame trend, enforce rise."""
    f = np.asarray(frames, float)
    v = np.asarray(odo, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return v
    for _ in range(3):                      # robust line fit (Theil-sen-ish via lstsq + MAD reject)
        a, b = np.polyfit(f[ok], v[ok], 1)
        resid = v - (a * f + b)
        mad = np.median(np.abs(resid[ok] - np.median(resid[ok]))) + 1e-6
        ok = np.isfinite(v) & (np.abs(resid) <= 4.0 * 1.4826 * mad)
    out = np.where(ok, v, np.nan)
    # enforce non-decreasing among survivors (a backwards jump is a mis-read)
    last = -np.inf
    for j in range(len(out)):
        if np.isfinite(out[j]):
            if out[j] < last - 5.0:
                out[j] = np.nan
            else:
                last = max(last, out[j])
    return out


def main():
    import av

    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.aerial_ocr import read_odometer
    frames, odo = [], []
    for i, fr in enumerate(av.open(VIDEO).decode(video=0)):
        panel = fr.to_ndarray(format="bgr24")[:, HALF:]
        if not render_active(panel):
            continue
        crop = _odo_crop(panel)
        m = read_odometer(crop) if crop is not None else None
        frames.append(i)
        odo.append(DIST - m if m is not None else np.nan)
    frames = np.array(frames)
    odo = _clean(frames, np.array(odo, float), DIST)
    np.savez(OUT, frames=frames, odo=odo)
    fin = np.isfinite(odo)
    rng = f"{np.nanmin(odo):.0f}..{np.nanmax(odo):.0f}" if fin.any() else "none"
    print(f"{RACE}: {len(frames)} render frames, {fin.sum()} clean odo reads ({100*fin.mean():.0f}%); "
          f"arclen range {rng} m of {DIST:.0f} -> {OUT}")


if __name__ == "__main__":
    main()

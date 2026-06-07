"""Compare a few PREPROCESSING variants for single-frame avatar isolation (user request).

Goal: ONE solid blob per horse (no fragmentation, no merges). Shows, per frame, the original
beside several candidate masks so the approach can be judged by eye:

  A. COLOR        - turf-suppress (non-green, non-rail), moderate morph-close.
  B. COLOR+CLOSE  - same, AGGRESSIVE close to fuse a horse's fragments into one blob.
  C. EDGES        - turf-suppress -> Canny edges -> dilate+close -> fill -> solid silhouettes.
  D. SALIENCY     - continuous distance-from-turf-hue, blurred, thresholded -> smooth blobs.

No k-means / no splitting (that was a misread). turf hue auto-detected. Writes a side-by-side
montage to data_cache/diag/. Judge: which gives ~11 clean, whole, separated horse blobs?

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_preproc_variants.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
DEBUG_FRAMES = [1400, 2580]   # bend (strung) + finish (bunched)


def _hsv(band):
    import cv2
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    return h, s, v, d_hue, turf_hue


def _ncomp(mask):
    import cv2
    n, _l, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 30)


def var_color(band, close_k):
    import cv2
    _h, s, v, d_hue, _t = _hsv(band)
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    white = (v > 180) & (s < 50)
    horse = ((~is_turf) & (~white)).astype(np.uint8)
    horse = cv2.morphologyEx(horse, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    horse = cv2.morphologyEx(horse, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return horse * 255


def var_edges(band):
    import cv2
    _h, s, v, d_hue, _t = _hsv(band)
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    sup = band.copy()
    sup[is_turf] = 0                       # blank the turf, keep horses
    gray = cv2.cvtColor(sup, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # fill enclosed silhouettes
    fill = edges.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fill, cnts, -1, 255, -1)
    return fill


def var_saliency(band):
    import cv2
    _h, s, v, d_hue, _t = _hsv(band)
    # continuous "non-turf" score: far hue from turf, OR dark, OR vivid
    sal = np.clip(d_hue / 25.0, 0, 1) * (s.astype(float) / 255) + (1 - v.astype(float) / 255) * 0.6
    sal = (sal / sal.max() * 255).astype(np.uint8)
    sal = cv2.GaussianBlur(sal, (5, 5), 0)
    _t2, m = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return m


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
        band = panel[ty0:ty1]
        variants = [
            ("ORIGINAL", cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) * 0 + band[:, :, 1]),  # placeholder
            ("A color close5", var_color(band, 5)),
            ("B color close11", var_color(band, 11)),
            ("C edges->fill", var_edges(band)),
            ("D saliency-otsu", var_saliency(band)),
        ]
        tiles = []
        for name, m in variants:
            if name == "ORIGINAL":
                tile = band.copy()
            else:
                tile = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                cnt = _ncomp(m)
                name = f"{name}  ({cnt} blobs)"
            cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
            tiles.append(tile)
        montage = np.vstack(tiles)
        z = 2
        montage = cv2.resize(montage, (montage.shape[1] * z, montage.shape[0] * z),
                             interpolation=cv2.INTER_NEAREST)
        out = f"{DIAG}/cv_preproc_variants_f{f:04d}.png"
        cv2.imwrite(out, montage)
        counts = {name: _ncomp(m) for name, m in variants if name != "ORIGINAL"}
        print(f"f{f}: blob counts {counts} (field=11) -> {out}")


if __name__ == "__main__":
    main()

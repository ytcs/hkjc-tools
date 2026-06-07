"""Ground-contact extraction — the horse's TRUE ground projection, not its body centroid.

At an oblique angle a horse's bright body sits ABOVE its ground position, so the body
centroid carries a camera-angle lateral bias. The two feeds that read lateral from a
real/rendered 3D scene recover the ground contact two ways, with a shared signature:

- :func:`shadow_base` — the dark shadow under a rendered/aerial 3D model (aerial feed).
- :func:`plumb_to_hoof` — a plumb dropped from a head-on silk down to where the body meets
  turf (the fixed-mount patrol feed).

See ``docs/CV_RECON.md``. cv2 is imported lazily.
"""
from __future__ import annotations

import numpy as np


def shadow_base(panel: np.ndarray, cx: float, cy: float, ty0: int, ty1: int,
                r: int = 18) -> tuple[float, float, bool]:
    """Shadow-base ground contact for a 3D-rendered/aerial figure near ``(cx, cy)``.

    Crops a window around the figure, finds the dark on-turf shadow pixels (turf-hued but
    darker than the local turf median value), and returns the shadow centroid. Falls back to
    the body centroid if no clean shadow is found.

    Returns ``(gx, gy, used_shadow)``.
    """
    import cv2

    x0 = max(0, int(cx - r))
    x1 = min(panel.shape[1], int(cx + r))
    y0 = max(ty0, int(cy - r))
    y1 = min(ty1, int(cy + r))
    win = panel[y0:y1, x0:x1]
    if win.size == 0:
        return float(cx), float(cy), False
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    turf = h_[(s_ > 40) & (v_ > 40)]
    if turf.size < 10:
        return float(cx), float(cy), False
    thue = int(np.median(turf))
    turf_band = np.abs(h_.astype(int) - thue) <= 18
    vturf = v_[turf_band]
    vmed = float(np.median(vturf)) if vturf.size else 128.0
    shadow = turf_band & (v_ < 0.72 * vmed) & (v_ > 25)
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sy, sx = np.where(shadow)
    if len(sx) >= 6:
        return float(np.mean(sx)) + x0, float(np.mean(sy)) + y0, True
    return float(cx), float(cy), False


def plumb_to_hoof(turf_mask: np.ndarray, cx: float, silk_bottom: float,
                  max_drop: int = 220, run_min: int = 6) -> float:
    """Drop a plumb from a head-on silk down to the horse's hoof/shadow base.

    Walks down the ``cx`` column from ``silk_bottom`` through the (non-turf) horse body until a
    SUSTAINED run of ``run_min`` turf pixels begins — the point just above that is the ground
    contact. ``turf_mask`` is a boolean/uint8 turf mask the size of the frame. Returns ``y_ground``.
    """
    h, w = turf_mask.shape[:2]
    col_x = int(np.clip(cx, 0, w - 1))
    y_ground = silk_bottom
    run = 0
    for y in range(int(silk_bottom), min(h, int(silk_bottom) + max_drop)):
        if turf_mask[y, col_x]:
            run += 1
            if run >= run_min:
                y_ground = y - run
                break
        else:
            run = 0
            y_ground = y
    return float(y_ground)

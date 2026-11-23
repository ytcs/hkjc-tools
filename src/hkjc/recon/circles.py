"""Numbered-circle render front end (2026/27 replays): each runner is a blue disc (radius ~25 px at 1080p) with
its saddlecloth number in white; no silk avatars and no legend strip.

``centres`` finds disc centres as peaks of the disc-template fill fraction of the blue mask, which splits
overlapping discs down to ~20 px (about 1 m) apart. ``read`` OCRs the number inside each disc."""

from __future__ import annotations

import numpy as np

R = 25  # disc radius, panel px
CROP = 36  # number crop (px square around the centre)
FILL_MIN = 0.80  # fraction of the disc template that must be blue at a centre
NMS = 18  # px: minimum centre separation


def blue_mask(panel) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (100, 121, 111), (125, 255, 255)) & 1  # 100 <= h <= 125, s > 120, v > 110 as 0/1


_DISC = None


def centres(panel) -> np.ndarray:
    """(N, 6) [cx, cy, w, h, fill, 0] disc detections in panel px (the avatar detector's row layout)."""
    import cv2

    global _DISC
    if _DISC is None:
        d = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
        cv2.circle(d, (R, R), R, 1.0, -1)
        _DISC = d / d.sum()
    fill = cv2.morphologyEx(blue_mask(panel), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    s = cv2.filter2D(fill.astype(np.float32), -1, _DISC)
    peak = (s >= FILL_MIN) & (s == cv2.dilate(s, np.ones((19, 19), np.uint8)))
    ys, xs = np.nonzero(peak)
    out = []
    for y, x in sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: -s[p[0], p[1]]):
        if all((x - a) ** 2 + (y - b) ** 2 > NMS**2 for a, b, *_ in out):
            out.append((x, y, 2 * R, 2 * R, float(s[y, x]), 0.0))
    return np.array(out, float).reshape(-1, 6)


def crop(panel, x, y) -> np.ndarray:
    h = CROP // 2
    return panel[max(0, int(y) - h) : int(y) + h, max(0, int(x) - h) : int(x) + h].copy()


def read(crops) -> list[tuple[int, float]]:
    """[(number or -1, confidence)] per number crop (batched recognition-only OCR)."""
    from . import ocr

    out = []
    for t, c in ocr.rec_batch(crops, scale=3) if crops else []:
        d = "".join(ch for ch in t if ch in "0123456789")
        out.append((int(d) if d else -1, float(c)))
    return out


def count(panel) -> int:
    return len(centres(panel))

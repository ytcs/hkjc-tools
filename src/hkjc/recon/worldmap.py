"""World map of the AVV course: mosaic the de-duplicated render frames through their poses.

Averaging hundreds of registered frames erases the moving horses and leaves the STATIC world
(turf, rails, painted markers) — rails are then extracted ONCE on a static image instead of per
frame. ``Mosaic`` accumulates incrementally so a single decode pass can fill several maps
(e.g. per-pass maps that test drift vs a genuine fixed/movable rail pair).
"""

from __future__ import annotations

import numpy as np


class Mosaic:
    def __init__(self, bounds, scale: float = 1.0):
        """bounds = (x0, y0, x1, y1) in world px; scale = mosaic px per world px."""
        self.x0, self.y0, x1, y1 = bounds
        self.sc = scale
        self.W = int(np.ceil((x1 - self.x0) * scale))
        self.H = int(np.ceil((y1 - self.y0) * scale))
        self.acc = np.zeros((self.H, self.W, 3), np.float32)
        self.cnt = np.zeros((self.H, self.W), np.float32)
        self.A = np.array([[scale, 0, -self.x0 * scale], [0, scale, -self.y0 * scale], [0, 0, 1]])

    def add(self, panel, mask, G):
        import cv2

        h, w = panel.shape[:2]
        T = self.A @ G
        cn = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ T.T
        x0, y0 = np.floor(cn[:, :2].min(0)).astype(int)
        x1, y1 = np.ceil(cn[:, :2].max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
        if x1 <= x0 or y1 <= y0:
            return
        T2 = T.copy()
        T2[0, 2] -= x0
        T2[1, 2] -= y0
        m = mask.astype(np.float32)
        self.acc[y0:y1, x0:x1] += cv2.warpAffine(
            panel.astype(np.float32) * m[..., None],
            T2[:2],
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
        )
        self.cnt[y0:y1, x0:x1] += cv2.warpAffine(m, T2[:2], (x1 - x0, y1 - y0))

    def image(self) -> np.ndarray:
        return (self.acc / np.maximum(self.cnt, 1e-3)[..., None]).clip(0, 255).astype(np.uint8)

    def to_px(self, xy):
        """world px -> mosaic px"""
        xy = np.atleast_2d(xy)
        return (xy - [self.x0, self.y0]) * self.sc

    def to_world(self, uv):
        return np.atleast_2d(uv) / self.sc + [self.x0, self.y0]


def bounds_of(G, shape, pad: float = 50) -> tuple:
    h, w = shape
    cn = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float)
    P = np.einsum("kij,nj->kni", G, cn)[..., :2].reshape(-1, 2)
    lo, hi = P.min(0) - pad, P.max(0) + pad
    return (lo[0], lo[1], hi[0], hi[1])

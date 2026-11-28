"""Turf-texture registration: ORB features + RANSAC similarity.

The render camera is near-nadir, so world->image is a 2D similarity (rotation, translation, slow
zoom; a homography fits no better). Turf/rail/marker texture is static world; screen overlays and
avatars are masked, residual horse features are rejected by RANSAC.
"""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np


@dataclass
class Feats:
    pts: np.ndarray  # (N,2) float32 panel px
    des: np.ndarray  # (N,32) uint8 ORB descriptors


def world_mask(panel, hud) -> np.ndarray:
    """HUD-free pixels, minus non-turf blobs (horses/shadows) dilated, as a 0/255 ORB mask."""
    import cv2

    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    # avatars are small vivid / dark blobs on green; keep turf, rails (white), stands (static)
    dark = (hsv[..., 2] < 45).astype(np.uint8)
    vivid = ((hsv[..., 1] > 120) & ((hsv[..., 0] < 22) | (hsv[..., 0] > 95))).astype(np.uint8)
    blob = cv2.dilate(dark | vivid, np.ones((9, 9), np.uint8))
    return ((hud > 0) & (blob == 0)).astype(np.uint8) * 255


def mask_boxes(mask, dets, grow: float = 1.6) -> np.ndarray:
    """Zero ``mask`` inside each avatar box (cx, cy, w, h, ...) grown by ``grow`` (covers the shadow).
    The colour mask misses brown-bodied / pastel-silk avatars, which the camera keeps static in the
    image, so step matches lock onto them unless their boxes are masked."""
    m = mask.copy()
    H, W = m.shape[:2]
    for d in dets:
        cx, cy, w, h = d[:4]
        hw, hh = grow * w / 2, grow * h / 2
        x0, x1 = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
        y0, y1 = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
        m[y0:y1, x0:x1] = 0
    return m


_ORB = {}


def extract(panel, mask, n: int = 3000) -> Feats:

    import cv2

    if n not in _ORB:
        _ORB[n] = cv2.ORB_create(nfeatures=n, fastThreshold=7)
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    kp, des = _ORB[n].detectAndCompute(g, mask)
    if des is None:
        return Feats(np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8))
    return Feats(np.float32([k.pt for k in kp]), des)


def umeyama(A, B) -> np.ndarray:
    """Closed-form least-squares similarity A -> B (3x3)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, Sg, Vt = np.linalg.svd(b.T @ a)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    sc = (Sg * np.diag(D)).sum() / (a**2).sum()
    M = np.eye(3)
    M[:2, :2] = sc * R
    M[:2, 2] = mb - sc * R @ ma
    return M


@dataclass
class Match:
    S: np.ndarray  # 3x3 similarity mapping a-pts -> b-pts
    n_inl: int
    resid: float  # median inlier residual (px)


def match(
    a: Feats,
    b: Feats,
    *,
    prior: np.ndarray | None = None,
    gate: float = 40.0,
    thr: float = 1.5,
    min_inl: int = 30,
    nn=None,
) -> Match | None:
    """Similarity a->b. With ``prior`` (3x3 a->b guess) candidate matches are restricted to those
    whose prior-predicted position is within ``gate`` px — kills aliasing on repetitive turf."""
    import cv2

    if len(a.pts) < 10 or len(b.pts) < 10:
        return None
    # nn: a precomputed bfmatch table for this pair (cross-check (ia, ib), or the knn train-index table)
    bf = None if nn is not None else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=prior is None)
    if prior is None:
        if nn is not None:
            ia, ib = nn
        else:
            ms = bf.match(a.des, b.des)
            ia = np.array([m.queryIdx for m in ms])
            ib = np.array([m.trainIdx for m in ms])
    else:
        pa = a.pts @ prior[:2, :2].T + prior[:2, 2]
        if nn is not None:
            ti = np.asarray(nn)
            qi = np.repeat(np.arange(len(ti))[:, None], ti.shape[1], axis=1)
        else:
            knn = bf.knnMatch(a.des, b.des, k=4)
            # vectorised (identical result): per query, the FIRST of its k candidates within the gate
            L = max((len(q) for q in knn), default=0)
            qi = np.full((len(knn), L), -1)
            ti = np.full((len(knn), L), -1)
            for r, q in enumerate(knn):
                for c, m in enumerate(q):
                    qi[r, c], ti[r, c] = m.queryIdx, m.trainIdx
        ok = ti >= 0
        dist = np.full(qi.shape, np.inf)
        dist[ok] = np.hypot(*(pa[qi[ok]] - b.pts[ti[ok]]).T)
        hit = dist < gate
        rows = np.where(hit.any(1))[0]
        first = hit[rows].argmax(1)
        ia, ib = qi[rows, first], ti[rows, first]
    if len(ia) < min_inl:
        return None
    A, B = a.pts[ia], b.pts[ib]
    M, inl = cv2.estimateAffinePartial2D(
        A,
        B,
        method=cv2.RANSAC,
        ransacReprojThreshold=thr,
        maxIters=4000,
        confidence=0.999,
    )
    if M is None or inl.sum() < min_inl:
        return None
    k = inl.ravel() > 0
    # least-squares refit on inliers (Umeyama): a median refit would snap integer FAST keypoints to
    # whole-pixel steps (an accumulating speed-dependent bias)
    S = umeyama(A[k], B[k])
    r = np.linalg.norm(A[k] @ S[:2, :2].T + S[:2, 2] - B[k], axis=1)
    return Match(S, int(k.sum()), float(np.median(r)))

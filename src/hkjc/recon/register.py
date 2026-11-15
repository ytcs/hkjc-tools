"""Turf-texture registration: ORB features + RANSAC similarity.

The render camera is near-nadir with no zoom, so world->image is a 2D similarity (rotation,
translation, slow zoom). Measured on R08: similarity residual == homography residual
(0.61 vs 0.63 px median) and per-step scale 1.000 +- 0.001. The turf/rail/marker texture is static
world, the horses + screen overlays are masked (overlays) or rejected by RANSAC (horses).
"""

from __future__ import annotations

from dataclasses import dataclass

import os

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


def world_mask_v2(panel, hud, *, blob_max=6000) -> np.ndarray:
    """HUD-free pixels minus COMPACT NON-TURF blobs (avatars + shadows), dilated, as a 0/255 ORB mask.
    Turf = the panel's dominant hue mode (per frame, so Happy Valley's yellower turf is not clipped) or
    white (rail, markings); shadows = turf much darker than the local turf. Non-turf connected regions
    smaller than ``blob_max`` px are masked; LARGER ones (stands, the all-weather track, paths: static
    world structure) are kept.
    MEASURED (pixels, ST 0419 R4 frames 1365-1373): the v1 blacklist (V < 45 or vivid hue) left
    brown-bodied / pastel-silk avatars unmasked; the camera follows the field, so those avatars are
    static in the image, and the step match locked onto them (111 inliers on horses): the camera
    "stopped" for 3 ticks (image-centre step < 1 px vs 36), the field's world speed dropped to ~0 and
    the tracker rotated identities. With this mask the steps there are (34-36, -11..-13) px."""
    import cv2

    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    h, s_, v = (hsv[..., i].astype(np.int16) for i in range(3))
    world = hud > 0
    cand = world & (s_ >= 35) & (v >= 60)
    hist = np.bincount(h[cand].ravel(), minlength=180) if cand.any() else np.zeros(180)
    mode = int(np.argmax(np.convolve(hist, np.ones(7), "same")))
    dh = np.minimum(np.abs(h - mode), 180 - np.abs(h - mode))
    turf = (dh <= 15) & (s_ >= 30)
    white = (s_ < 45) & (v > 165)
    vloc = cv2.blur(v.astype(np.float32), (61, 61))
    turf &= v >= 0.72 * vloc  # shadows: turf much darker than the local turf
    non = (world & ~(turf | white)).astype(np.uint8)
    non = cv2.morphologyEx(non, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(non)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < blob_max
    bad = cv2.dilate(small[lab].astype(np.uint8), np.ones((9, 9), np.uint8))
    return (world & (bad == 0)).astype(np.uint8) * 255


def mask_boxes(mask, dets, grow: float = 1.6) -> np.ndarray:
    """Zero ``mask`` inside each avatar box (cx, cy, w, h, ...) grown by ``grow`` (covers the shadow).
    MEASURED (pixels, ST 0419 R4 frames 1365-1373): the colour blacklist leaves brown-bodied /
    pastel-silk avatars unmasked; the camera follows the field, so they are static in the image and
    the step match locked onto them (camera "stopped" 3 ticks). The avatar detector finds them; a
    colour WHITELIST (``world_mask_v2``) also fixed the stall but dropped small static world features
    (posts, markers): HV 0429 R3 loop closures 335 -> 0."""
    m = mask.copy()
    H, W = m.shape[:2]
    for d in dets:
        cx, cy, w, h = d[:4]
        hw, hh = grow * w / 2, grow * h / 2
        x0, x1 = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
        y0, y1 = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
        m[y0:y1, x0:x1] = 0
    return m


def world_mask_sel(panel, hud) -> np.ndarray:
    """``world_mask`` (default; the features stage adds the avatar boxes unless HKJC_WORLD_MASK=v1) or
    ``world_mask_v2`` (env HKJC_WORLD_MASK=v2)."""
    if os.environ.get("HKJC_WORLD_MASK", "v1") == "v2":
        return world_mask_v2(panel, hud)
    return world_mask(panel, hud)


_ORB = {}


def orb_scale() -> float:
    """Feature-extraction scale (env HKJC_ORB_SCALE; 1.0 = full-res panel). Keypoints are always
    returned in FULL-RES panel pixels; the RANSAC threshold scales with 1/scale.
    REJECTED as default (10 dev races): scale 0.5 / 1500 features is 5x faster (features 72 -> 15 s,
    pose 88 -> 18 s) but only 2 races pass every timing gate vs 5 at full res — the long-baseline and
    loop-closure graph thins out (R08 d24 edges 212 -> 31, loops 171 -> 0; R02A sectional 0.18 ->
    0.42 s). Scale 0.5 with 3000 features keeps R08 but saves only ~35 %."""

    return float(os.environ.get("HKJC_ORB_SCALE", "1.0"))


def extract(panel, mask, n: int = 3000) -> Feats:

    import cv2

    sc = orb_scale()
    if sc != 1.0:  # ~1/sc^2 fewer pixels; FAST/ORB on the downscaled panel
        panel = cv2.resize(panel, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST)
        n = int(os.environ.get("HKJC_ORB_N", n))  # feature count (default unchanged)
    if n not in _ORB:
        _ORB[n] = cv2.ORB_create(nfeatures=n, fastThreshold=7)
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    kp, des = _ORB[n].detectAndCompute(g, mask)
    if des is None:
        return Feats(np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8))
    return Feats(np.float32([k.pt for k in kp]) / sc, des)


def sim_matrix(s, th_rad, tx, ty) -> np.ndarray:
    c, n = s * np.cos(th_rad), s * np.sin(th_rad)
    return np.array([[c, -n, tx], [n, c, ty], [0, 0, 1.0]])


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

    @property
    def scale(self):
        return float(np.hypot(self.S[0, 0], self.S[1, 0]))


def match(
    a: Feats,
    b: Feats,
    *,
    prior: np.ndarray | None = None,
    gate: float = 40.0,
    thr: float = 1.5,
    min_inl: int = 30,
) -> Match | None:
    """Similarity a->b. With ``prior`` (3x3 a->b guess) candidate matches are restricted to those
    whose prior-predicted position is within ``gate`` px — kills aliasing on repetitive turf.
    Speed alternatives tried and REJECTED (R08, 2026-11-13): (1) projection-guided search (KD-tree
    candidates within the gate, Hamming only for those): no gain — ~30 keypoints per 40 px gate at this
    density, so ~47k candidate pairs cost as much as the brute-force kNN (~25 ms/pair); (2) XFeat
    (2404.19174) learned features: CPU extraction 137 vs 34 ms/frame, matching not faster (16 ms), and
    long-baseline (d=24) pairs fail on the repetitive mowing stripes (24 vs 90 inliers, transforms
    ~680 px off). ORB + brute-force Hamming stays; a CUDA GPU (cv2.cuda / torch Hamming) is the lever."""
    import cv2

    if len(a.pts) < 10 or len(b.pts) < 10:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=prior is None)
    if prior is None:
        ms = bf.match(a.des, b.des)
        ia = np.array([m.queryIdx for m in ms])
        ib = np.array([m.trainIdx for m in ms])
    else:
        pa = a.pts @ prior[:2, :2].T + prior[:2, 2]
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
        ransacReprojThreshold=thr / min(orb_scale(), 1.0),
        maxIters=4000,
        confidence=0.999,
    )
    if M is None or inl.sum() < min_inl:
        return None
    k = inl.ravel() > 0
    # least-squares refit on inliers (Umeyama). NOT LMEDS: FAST keypoints are integer-located, and
    # a median refit snaps the step to whole pixels (speed-dependent bias that accumulates).
    S = umeyama(A[k], B[k])
    r = np.linalg.norm(A[k] @ S[:2, :2].T + S[:2, 2] - B[k], axis=1)
    # residual in scale-normalised units (keypoints localise ~1/orb_scale coarser at reduced scale),
    # so the callers' acceptance thresholds (median resid <= 1.3) keep their meaning
    return Match(S, int(k.sum()), float(np.median(r)) * min(orb_scale(), 1.0))

"""Patrol head-on lateral reconstruction (study only — task #83).

Fixed-camera head-on patrol feed, ST 2026-06-03 R1. Recovers per-detection GROUND-CONTACT
lateral offset from the LEFT running rail and computes the within/between dispersion ratio
that nulled aerial/drone (0071: within 3.8m / between 3.3m, ratio 0.87; drone >1). The
thesis: a FIXED camera + HEAD-ON view gives a CONSTANT homography (no per-frame rail
jitter — the exact 0071 root cause) and lateral = ~the horizontal image coordinate after
perspective rectification, so the recovered lateral should be STABLE (ratio << 1).

Pipeline:
  1. head-on window t=83-96s (verified by eye: field runs toward the FIXED camera, white
     rail static on the LEFT; before ~83s the camera still pans off the bend, after ~96s
     the near horses occlude the rail and the feed cuts to the finish camera at ~t=101).
  2. FIXED rail: fit the white left rail per-frame, take the MEDIAN line ONCE (camera is
     fixed => ONE homography). Slope CV ~3% demonstrates stability.
  3. ground-plane homography from the rail line + its vanishing point + a metric width
     anchor (perspective-correct; the lateral axis is metres-from-rail). The within/between
     RATIO is scale-invariant, so it is the robust headline; the metric scale is a sanity
     readout cross-checked against the "C" width 21.3 m.
  4. detect JOCKEY SILKS (saturated, non-turf), cluster fragments per jockey, drop a plumb
     to the horse's GROUND CONTACT (hoof/shadow base) below the silk, project to lateral.
  5. link detections into anonymous per-horse tracks; within/between lateral dispersion.

Official data (finish order, stewards' trip notes) is VALIDATION only, never an input.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

VIDEO = "data_cache/video/patrol_20260603_01.mp4"
COURSE_WIDTH_M = 21.3
T0, T1 = 83.0, 96.0  # verified fixed head-on window (rail slope steady, CV ~3%)


@dataclasses.dataclass
class FixedRail:
    a: float  # x = a*y + b (image)
    b: float
    vp_x: float
    vp_y: float
    slope_cv: float
    intercept_std: float
    n_frames: int
    per_frame: list  # (t, a, b)


def _fit_white_rail(fr: np.ndarray) -> tuple[float, float] | None:
    """Left white running-rail line x = a*y + b from one frame (robust inlier fit)."""
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = ((v > 175) & (s < 55)).astype(np.uint8)
    white[: int(0.10 * h), : int(0.30 * w)] = 0  # 'Patrol Replay' legend box
    pts = []
    for y in range(int(0.24 * h), h, 2):
        row = np.where(white[y, : int(0.45 * w)] > 0)[0]
        if len(row):
            pts.append((row.min(), y))
    if len(pts) < 30:
        return None
    pts = np.array(pts, float)
    xs, ys = pts[:, 0], pts[:, 1]
    a, b = np.polyfit(ys, xs, 1)
    for _ in range(3):
        r = xs - (a * ys + b)
        s_ = np.std(r) + 1e-6
        keep = np.abs(r) < 2.0 * s_
        if keep.sum() < 30:
            break
        a, b = np.polyfit(ys[keep], xs[keep], 1)
    return float(a), float(b)


def fit_fixed_rail(cap: cv2.VideoCapture, fps: float) -> FixedRail:
    """Fit the rail ONCE as the median line across the head-on window (fixed camera)."""
    A, B, per = [], [], []
    for t in np.arange(T0, T1 + 0.01, 0.5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        r = _fit_white_rail(fr)
        if r:
            A.append(r[0])
            B.append(r[1])
            per.append((float(t), r[0], r[1]))
    A, B = np.array(A), np.array(B)
    a_med, b_med = float(np.median(A)), float(np.median(B))
    # vanishing point: the rail's far convergence. The straight's VP lies on the rail line;
    # we take it at the image-top horizon (y=0) as a stable along-track foreshortening pole.
    vp_y = 0.0
    vp_x = a_med * vp_y + b_med
    cv = float(np.std(A) / abs(a_med)) if a_med else float("nan")
    return FixedRail(a_med, b_med, vp_x, vp_y, cv, float(np.std(B)), len(A), per)


def _turf_hue(fr: np.ndarray) -> int:
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = (h > 30) & (h < 90) & (s > 30) & (v > 40)
    return int(np.median(h[g])) if g.any() else 55


def _silk_mask(fr: np.ndarray, turf_lock: int) -> np.ndarray:
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (np.abs(hh.astype(int) - turf_lock) <= 16) & (ss > 30) & (vv > 40)
    silk = ((ss > 110) & (vv > 90) & (~turf)).astype(np.uint8)
    silk[: int(0.10 * h), : int(0.30 * w)] = 0  # legend box
    return silk


def detect_jockeys(fr: np.ndarray, turf_lock: int) -> list[dict]:
    """Detect per-jockey silk clusters and the horse GROUND-CONTACT point below each.

    Silks fragment into colour patches; dilate-then-CC merges a jockey's patches. The
    horse's ground contact is found by dropping a plumb from the silk centre down through
    the dark horse body to where it meets turf (hoof/shadow base)."""
    h, w = fr.shape[:2]
    m = _silk_mask(fr, turf_lock)
    md = cv2.dilate(m, np.ones((11, 11), np.uint8))
    n, _lbl, st, ct = cv2.connectedComponentsWithStats(md, 8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (np.abs(hh.astype(int) - turf_lock) <= 16) & (ss > 30) & (vv > 40)
    out = []
    for i in range(1, n):
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        core = int(m[by : by + bh, bx : bx + bw].sum())
        if core < 60 or bw > 140 or bh > 160:
            continue
        cx = float(ct[i][0])
        silk_bottom = by + bh
        # plumb down to ground: walk down the cx column from silk bottom; the horse body is
        # non-turf; stop where turf begins for a sustained run (hoof/shadow base).
        col_x = int(np.clip(cx, 0, w - 1))
        y_ground = silk_bottom
        run = 0
        for y in range(silk_bottom, min(h, silk_bottom + 220)):
            if turf[y, col_x]:
                run += 1
                if run >= 6:  # sustained turf -> we passed the hooves
                    y_ground = y - run
                    break
            else:
                run = 0
                y_ground = y
        out.append(
            {
                "cx": cx,
                "cy": float(ct[i][1]),
                "y_ground": float(y_ground),
                "bw": bw,
                "bh": bh,
                "core": core,
            }
        )
    return out


def build_homography(rail: FixedRail, h: int, w: int, mpp_near: float) -> np.ndarray:
    """Ground-plane homography image(x,y) -> track(off_m, depth) using ONE rail + VP.

    The rail is off=0, a constant-lateral ground line through the VP. A parallel line at
    off=W metres shares the VP; at scanline y its horizontal offset from the rail is
    W*mpp_near scaled by the convergence toward the VP. Four coplanar correspondences =>
    an exact perspective homography. `mpp_near` = px per metre at the near scanline,
    calibrated from horse spacing / width (the RATIO output is scale-free regardless).
    """
    y_near = 0.90 * h
    y_far = 0.45 * h
    xL_near = rail.a * y_near + rail.b
    xL_far = rail.a * y_far + rail.b
    W = COURSE_WIDTH_M
    dx_near = W * mpp_near
    conv = (y_near - rail.vp_y) / (y_far - rail.vp_y)  # near wider than far
    dx_far = dx_near / conv
    src = np.array(
        [
            [xL_near, y_near],
            [xL_near + dx_near, y_near],
            [xL_far, y_far],
            [xL_far + dx_far, y_far],
        ],
        np.float32,
    )
    dst = np.array([[0.0, 0.0], [W, 0.0], [0.0, 1.0], [W, 1.0]], np.float32)
    Hmat, _ = cv2.findHomography(src, dst)
    return Hmat


def project(Hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.empty((0, 2))
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = (Hmat @ p.T).T
    return q[:, :2] / q[:, 2:3]

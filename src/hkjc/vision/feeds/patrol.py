"""Patrol feed — fixed head-on mount, the geometrically cleanest lateral METRIC.

A fixed stewards' mount gives a CONSTANT homography (one rail fit serves the whole head-on
window — no per-frame jitter) and, head-on, lateral ≈ the rectified horizontal image coord.
This module ports the patrol PERCEPTION: detect jockey silks + their ground contact (plumb to
hoof), build the fixed-rail homography, read per-runner lateral, and the silk<->number bridge
that gives the anonymous head-on tracks an identity from the finish-cam.

Rail/homography primitives come from ``geometry_cv`` (fixed rail), the silk descriptor from
``vision.silk``, the saddlecloth OCR from ``vision.ocr``. cv2 is imported lazily.
See ``docs/CV_RECON.md`` §2 (feeds), §4 (identity).
"""
from __future__ import annotations

import numpy as np

from ..geometry_cv import (
    FixedRail,
    fit_fixed_rail,
    fit_homography_from_rail,
    lateral_from_homography,
)
from ..ground_contact import plumb_to_hoof
from ..silk import silk_signature


# --------------------------------------------------------------------------- turf / silk masks
def turf_hue(fr: np.ndarray) -> int:
    """Modal turf hue (OpenCV 0..180) of a frame — the per-frame green lock for masking."""
    import cv2

    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = (h > 30) & (h < 90) & (s > 30) & (v > 40)
    return int(np.median(h[g])) if g.any() else 55


def silk_mask(fr: np.ndarray, turf_lock: int,
              legend_box: tuple[float, float] = (0.10, 0.30)) -> np.ndarray:
    """Saturated non-turf silk mask, with the top-left replay-legend box blanked."""
    import cv2

    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (np.abs(hh.astype(int) - turf_lock) <= 16) & (ss > 30) & (vv > 40)
    silk = ((ss > 110) & (vv > 90) & (~turf)).astype(np.uint8)
    silk[: int(legend_box[0] * h), : int(legend_box[1] * w)] = 0
    return silk


def detect_jockeys(fr: np.ndarray, turf_lock: int) -> list[dict]:
    """Detect per-jockey silk clusters + the horse GROUND CONTACT (hoof/shadow base) below each.

    Silks fragment into colour patches; a dilate-then-CC merges a jockey's patches. The ground
    contact is found by dropping a plumb from the silk centre through the dark body to the turf
    (``ground_contact.plumb_to_hoof``). Returns ``{cx, cy, y_ground, bw, bh, core}`` per jockey.
    """
    import cv2

    h, w = fr.shape[:2]
    m = silk_mask(fr, turf_lock)
    md = cv2.dilate(m, np.ones((11, 11), np.uint8))
    n, _lbl, st, ct = cv2.connectedComponentsWithStats(md, 8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (np.abs(hh.astype(int) - turf_lock) <= 16) & (ss > 30) & (vv > 40)
    out = []
    for i in range(1, n):
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        core = int(m[by: by + bh, bx: bx + bw].sum())
        if core < 60 or bw > 140 or bh > 160:
            continue
        cx = float(ct[i][0])
        y_ground = plumb_to_hoof(turf, cx, by + bh)
        out.append({"cx": cx, "cy": float(ct[i][1]), "y_ground": y_ground,
                    "bw": bw, "bh": bh, "core": core})
    return out


# --------------------------------------------------------------------------- homography
def build_homography(rail: FixedRail, h: int, w: int, mpp_near: float,
                     track_width_m: float) -> np.ndarray:
    """Fixed-mount ground-plane homography from the one rail + its VP (see ``geometry_cv``)."""
    return fit_homography_from_rail(rail, h, w, mpp_near, track_width_m)


def project(hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply the homography to ``(N, 2)`` image points -> ``(N, 2)`` track coords."""
    return lateral_from_homography(hmat, pts)


def calibrate_mpp_near(rail: FixedRail, jockeys: list[dict], field_span_m: float = 15.0) -> float:
    """Near-scanline px/m from the field's rectified rail-perp span over a known span (m).

    The mid-window field fans across ~``field_span_m``; the perpendicular pixel span of its
    ground contacts to the rail sets the near px/m. A within/between RATIO read is scale-free,
    so this only fixes the metric readout.
    """
    if not jockeys:
        return 60.0
    cxs = np.array([j["cx"] for j in jockeys])
    ygs = np.array([j["y_ground"] for j in jockeys])
    perp = cxs - (rail.a * ygs + rail.b)
    span_px = float(np.max(perp) - np.min(perp)) if len(perp) else 1.0
    return (span_px / field_span_m) if span_px > 0 else 60.0


# --------------------------------------------------------------------------- silk descriptors
def _silk_hsv_patch(fr: np.ndarray, mask: np.ndarray, j: dict, w: int, h: int):
    """HSV pixels of a head-on jockey's silk bbox (masked to saturated silk)."""
    import cv2

    x0 = int(max(0, j["cx"] - j["bw"]))
    x1 = int(min(w, j["cx"] + j["bw"]))
    y0 = int(max(0, j["cy"] - j["bh"]))
    y1 = int(min(h, j["cy"] + j["bh"] * 0.6))
    sm = mask[y0:y1, x0:x1] > 0
    if sm.sum() < 8:
        return None
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    return hsv[y0:y1, x0:x1][sm].reshape(-1, 3)


def head_on_silk(fr: np.ndarray, turf_lock: int, jockeys: list[dict]) -> list[tuple]:
    """Per detected jockey, ``(cx, y_ground, silk_signature)`` — the head-on identity colour read."""
    h, w = fr.shape[:2]
    m = silk_mask(fr, turf_lock)
    out = []
    for j in jockeys:
        patch = _silk_hsv_patch(fr, m, j, w, h)
        sig = silk_signature(patch) if patch is not None else None
        if sig is not None:
            out.append((j["cx"], j["y_ground"], sig))
    return out


def finish_cam_silk_blobs(fr: np.ndarray) -> list[tuple]:
    """Saturated non-turf silk blobs on a finish-cam frame: ``(cx, cy, silk_signature, area)``.

    Used to bind each OCR'd saddlecloth number (``vision.ocr``) to its nearest ABOVE silk blob
    when building the number<->silk LUT from the finish camera.
    """
    import cv2

    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (hh > 30) & (hh < 90) & (s > 25) & (v > 40)
    silk = ((s > 105) & (v > 75) & (~turf)).astype(np.uint8)
    silk = cv2.morphologyEx(silk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, st, ct = cv2.connectedComponentsWithStats(silk, 8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < 35 or area > 3500:
            continue
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        mask = lbl[by: by + bh, bx: bx + bw] == i
        hp = hsv[by: by + bh, bx: bx + bw][mask]
        sig = silk_signature(hp)
        if sig is None:
            continue
        out.append((float(ct[i][0]), float(ct[i][1]), sig, area))
    return out


# --------------------------------------------------------------------------- shared clock
def vid_to_race_t(vid_t: float, offset: float) -> float:
    """Patrol video time -> shared ``race_t`` (``race_t = vid_t - offset``)."""
    return vid_t - offset


def race_t_to_arclen(race_t: float, section_times: np.ndarray, section_bounds: np.ndarray) -> float:
    """Shared ``race_t`` -> arclen (m run) via the official cumulative-section curve."""
    return float(np.interp(race_t, section_times, section_bounds))


__all__ = [
    "turf_hue", "silk_mask", "detect_jockeys", "build_homography", "project",
    "calibrate_mpp_near", "head_on_silk", "finish_cam_silk_blobs",
    "vid_to_race_t", "race_t_to_arclen", "fit_fixed_rail",
]

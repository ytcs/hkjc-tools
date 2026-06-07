"""Aerial-render horse detector — EDGE+FILL → exactly one solid blob per runner.

The dialed-in replacement for the legacy ``detect_blobs`` on the AVV green render. From the
preprocessing bake-off, *suppress turf+rail → Canny edges → fill contours → ∩ non-turf colour →
field-ROI → PCA rail-rejection → k-means-to-field* yields the cleanest whole-horse silhouettes
(no silk-colour fragmentation, no rail-bar leak). See ``docs/CV_RECON.md`` §2 and the study
``studies/cv_avatar_edgefill.py`` it was dialled in from.

Two centroid variants are returned per horse (the choice is the caller's; ``A`` is the validated
default):
  * ``A`` = PIXEL centroid of the cluster (pixel-weighted; leans to the larger fragment);
  * ``B`` = mean of the per-connected-component piece centroids (= the midpoint of a silk-split).

Pure numpy at import; cv2 is lazy. Nothing here is hand-tuned per race — the turf hue is read
from the frame and every threshold is a physical/graphics constant of the render.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Render-graphics constants (NOT per-race tunables): a rendered avatar is ~50 px long, a two-horse
# merge ~100 px; the rail bar is a long thin line. Areas/extents below bound horse-vs-rail-vs-noise.
FIELD_DEFAULT = 11      # one-per-runner k-means target; callers pass the real field size
AREA_MIN = 40          # px^2: drop noise specks
AREA_MAX = 1500        # px^2: drop the rail bar / huge merges (×4 guard in the component loop)
RAIL_LEN = 130.0       # px: principal-axis length above this = rail/line (horse ~50, 2-merge ~100)
RAIL_ELONG = 6.0       # major/minor axis ratio above this = thin line (horse ~2-3)


@dataclass(frozen=True)
class AerialBlob:
    """One detected runner avatar in PANEL-relative pixel coords.

    ``cx, cy`` centroid; ``angle`` principal-axis orientation (rad, avatar aligns with the rail);
    ``area`` cluster pixel count. ``cx`` is panel-relative (add ``half`` for full-frame).
    """

    cx: float
    cy: float
    angle: float
    area: int


def _field_roi(color: np.ndarray, shape, margin: int = 20):
    """Bbox (0/1 mask) of the horse CLUSTER. OPENING removes the thin rail (~3-5 px) but keeps horse
    bodies (~16 px); the robust (1-99 pct) bbox of what remains is the field, rail excluded."""
    import cv2

    opened = cv2.morphologyEx(color, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ys, xs = np.where(opened > 0)
    if len(xs) < 20:
        return None
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    roi = np.zeros(shape, np.uint8)
    roi[max(0, int(y0) - margin):int(y1) + margin, max(0, int(x0) - margin):int(x1) + margin] = 1
    return roi


def _kmeans_field_both(mask: np.ndarray, lbl: np.ndarray, field: int, ty0: int):
    """Partition foreground into EXACTLY ``field`` horses (k-means on pixel coords). Returns two
    centroid variants per horse: ``A`` = pixel centroid, ``B`` = mean of per-component centroids.
    Both lists hold :class:`AerialBlob` (cy is band-shifted back to panel coords by ``ty0``)."""
    import cv2

    ys, xs = np.where(mask > 0)
    if len(xs) < field * 3:
        return [], []
    pts = np.column_stack([xs, ys]).astype(np.float32)
    comp = lbl[ys, xs]
    k = min(field, len(pts) // 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _c, labels, _ctr = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    dets_a: list[AerialBlob] = []
    dets_b: list[AerialBlob] = []
    for j in range(k):
        m = labels == j
        sub = pts[m]
        cj = comp[m]
        if len(sub) < 3:
            continue
        c = sub - sub.mean(0)
        _u, _sg, vt = np.linalg.svd(c, full_matrices=False)
        ang = float(np.arctan2(vt[0, 1], vt[0, 0]))
        n_px = len(sub)
        dets_a.append(AerialBlob(float(sub[:, 0].mean()), float(sub[:, 1].mean() + ty0), ang, n_px))
        ccents = [[sub[cj == c2, 0].mean(), sub[cj == c2, 1].mean()] for c2 in np.unique(cj)]
        cc = np.array(ccents, float)
        dets_b.append(AerialBlob(float(cc[:, 0].mean()), float(cc[:, 1].mean() + ty0), ang, n_px))
    return dets_a, dets_b


def detect_edgefill(panel: np.ndarray, ty0: int, ty1: int, field: int = FIELD_DEFAULT):
    """Field-ROI → turf+rail suppress → Canny → fill → (∩ colour) → shape-filtered horse blobs.

    Returns ``(dets_a, dets_b, debug)`` where ``dets_a``/``dets_b`` are lists of :class:`AerialBlob`
    (variant A = pixel centroid is the validated default) and ``debug = (mask, turf_hue, ty0)``.
    ``ty0, ty1`` bound the render track band (see ``feeds.aerial.track_band``).
    """
    import cv2

    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    bright = (v > 175) & (s < 55)                 # rail + painted markers + light overlay (desat)
    sup = band.copy()
    sup[is_turf | bright] = 0
    gray = cv2.cvtColor(sup, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    fill = edges.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fill, cnts, -1, 255, -1)
    color = ((~is_turf) & (~bright)).astype(np.uint8)
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = ((fill > 0) & (color > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    roi = _field_roi(color, mask.shape)
    if roi is not None:
        mask = (mask * roi).astype(np.uint8)
    # REJECT rail/line components by PCA length+elongation (a diagonal rail has a square-ish bbox so
    # bbox-aspect misses it; its principal axis is long+thin).
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < AREA_MIN or area > AREA_MAX * 4:
            continue
        ys_i, xs_i = np.where(lbl == i)
        p = np.column_stack([xs_i, ys_i]).astype(float)
        cov = np.cov((p - p.mean(0)).T)
        ev = np.linalg.eigvalsh(cov)              # ascending [minor, major]
        major = float(np.sqrt(max(ev[1], 1e-6)))
        minor = float(np.sqrt(max(ev[0], 1e-6)))
        length = 4.0 * major                      # ~full extent along principal axis
        elong = major / minor
        if length > RAIL_LEN or elong > RAIL_ELONG:   # rail / line -> drop
            continue
        keep[lbl == i] = 255
    dets_a, dets_b = _kmeans_field_both(keep, lbl, field, ty0)
    return dets_a, dets_b, (mask, turf_hue, ty0)


# --------------------------------------------------------------------------- legacy blob detector
# The original colour-segmentation runner detector for the AVV render. Kept alongside the
# EDGE+FILL detector above as the legacy blob detector: vivid silk blobs = runners in the
# synthetic track ROI.
#
# The render draws each runner as a small (~12-26 px) saturated silk sprite on top of a darker
# horse body, against a low-saturation brown track. A COCO detector does not fire on these, but
# colour segmentation does: keep vivid, non-track pixels, connected-component them, and keep
# blob-sized components. Recall is kept modest-but-clean (a missed dark-silk runner is better
# than a track-texture false positive — re-id + the motion model bridge short gaps).
#
# Returns plain :class:`BlobDet` records (pixel space). cv2/numpy stay lazy via the caller.


@dataclass
class BlobDet:
    """One detected runner blob in one frame (pixel space)."""

    frame: int
    t: float
    cx: float            # blob centroid x (px)
    cy: float            # blob centroid y (px)
    area: int            # px
    lab: tuple           # (L, a, b) mean silk colour — the re-id key
    score: float = 1.0


def detect_blobs(
    img,
    layout,
    frame: int,
    t: float,
    *,
    sat_min: int = 90,
    val_min: int = 80,
    area_min: int = 12,
    area_max: int = 420,
    wh_max: int = 46,
    max_blobs: int | None = None,
) -> list[BlobDet]:
    """Detect runner silk blobs in the track ROI of one frame.

    Two render styles exist for the *Aerial Virtual View*: an older BROWN-track render
    (vivid silk sprites on a low-sat brown track) and a newer GREEN-turf render (small
    dark horse+jockey sprites with a tiny coloured silk patch, on a saturated green
    turf). We auto-detect the DOMINANT track hue (the modal hue of the mid-sat/mid-val
    pixels) and suppress it: a track-coloured pixel must be very saturated to survive,
    which kills the track texture while keeping the runner sprites. On the green render
    the horse BODY is dark, so we additionally keep low-value (dark) pixels — the body is
    a more reliable target than the ~6-px silk patch. Falls back to the brown heuristic
    if hue auto-detection is degenerate.
    """
    import cv2

    x0, x1 = layout.panel_x0, layout.panel_x1
    y0, y1 = layout.track_y0, layout.track_y1
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Dominant track hue from the mid-sat/mid-val (i.e. turf/dirt, not sprite) pixels.
    turf_pix = hue[(sat > 40) & (val > 40)]
    track_hue = int(np.median(turf_pix)) if turf_pix.size > 100 else 20
    is_green_render = 28 <= track_hue <= 70   # green turf vs brown (~12-22)

    if is_green_render:
        # Suppress the turf-hue band; keep dark horse bodies + vivid off-turf silk patches.
        track_band = (np.abs(hue.astype(int) - track_hue) <= 12) & (sat > 40)
        dark = val < 72
        off_turf = (~track_band) & (sat > 70) & (val > 70)
        mask = (dark | off_turf).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    else:
        brownish = (hue >= 8) & (hue <= 30)
        vivid = (sat > sat_min) & (val > val_min)
        mask = (vivid & (~brownish | (sat > 170))).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    lab_img = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)
    out: list[BlobDet] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (area_min <= area <= area_max and bw <= wh_max and bh <= wh_max):
            continue
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        patch = lab_img[by : by + bh, bx : bx + bw].reshape(-1, 3).astype(float)
        if is_green_render:
            # The re-id key is the SILK colour (the jockey patch), not the dark body. Take
            # the LAB of the most chromatic pixels in the blob (max |a|+|b| from neutral),
            # so look-alike dark bodies still separate by their silk colour.
            chroma = np.abs(patch[:, 1] - 128.0) + np.abs(patch[:, 2] - 128.0)
            k = max(3, int(0.30 * len(patch)))
            top = patch[np.argsort(chroma)[-k:]]
            mlab = top.mean(axis=0)
        else:
            mlab = patch.mean(axis=0)
        out.append(
            BlobDet(
                frame=frame,
                t=t,
                cx=float(cent[i, 0]) + x0,
                cy=float(cent[i, 1]) + y0,
                area=area,
                lab=(float(mlab[0]), float(mlab[1]), float(mlab[2])),
            )
        )
    if max_blobs is not None and len(out) > max_blobs:
        out.sort(key=lambda b: b.area, reverse=True)
        out = out[:max_blobs]
    return out

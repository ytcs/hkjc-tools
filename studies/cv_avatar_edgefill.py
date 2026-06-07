"""Refined EDGE+FILL avatar detector (the chosen direction) -> one solid blob per horse.

From the variant bake-off, edges->fill gives the cleanest whole-horse silhouettes (no fragmentation).
This refines it to ~field clean blobs:
  1. suppress turf AND bright rail/markers BEFORE Canny (kills the rail-bar artifact);
  2. edges -> dilate/close -> fill contours -> solid silhouettes;
  3. INTERSECT with the non-turf colour mask (edge noise on turf can't survive);
  4. close (merge a bright-silk gap that splits one horse) + open (denoise);
  5. shape/size filter: drop blobs that are not horse-sized or are rail-bar-elongated.

Shows original | edge+fill mask | final blobs (count) for review, to data_cache/diag/.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_edgefill.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
FIELD = 11
AREA_MIN = 40          # px^2: drop noise specks
AREA_MAX = 1500        # px^2: drop the rail bar / huge merges
RAIL_LEN = 130.0       # px: principal-axis length above this = rail/line (horse ~50, 2-merge ~100)
RAIL_ELONG = 6.0       # major/minor axis ratio above this = thin line (horse ~2-3)
DEBUG_FRAMES = [770, 1400, 2030, 2580]


def _field_roi(color, shape, margin=20):
    """Bbox (0/1 mask) of the horse cluster. OPENING removes the thin rail (~3-5px) but keeps horse
    bodies (~16px); the robust (1-99 pct) bbox of what remains is the field, rail excluded."""
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


def detect_edgefill(panel, ty0, ty1):
    """Field-ROI -> turf+rail suppress -> Canny -> fill -> (∩ colour) -> shape-filtered horse blobs."""
    import cv2

    band = panel[ty0:ty1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    turf = (s > 50) & (v > 60)
    turf_hue = int(np.bincount(h[turf].ravel(), minlength=180).argmax()) if turf.any() else 45
    d_hue = np.minimum(np.abs(h - turf_hue), 180 - np.abs(h - turf_hue))
    is_turf = (d_hue < 14) & (s > 35) & (v > 50)
    bright = (v > 175) & (s < 55)                 # rail + painted markers + light overlay (desat)
    # suppress turf + bright BEFORE edges
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
    # FIELD ROI (user): the horse CLUSTER is a DENSE region of non-turf; the rail/markers are thin/
    # sparse. Blur the non-turf colour mask -> the cluster lights up, the rail doesn't -> restrict the
    # detection mask to the densest region's bbox (+margin). Kills the distant-frame rail leak.
    roi = _field_roi(color, mask.shape)
    if roi is not None:
        mask = (mask * roi).astype(np.uint8)
    # denoise: keep non-trivial components, REJECT rail/line components by PCA length+elongation
    # (a diagonal rail has a square-ish bbox so bbox-aspect misses it; its principal axis is long+thin).
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
    dets_a, dets_b = _kmeans_field_both(keep, lbl, FIELD, ty0)
    return dets_a, dets_b, (mask, turf_hue, ty0)


def _kmeans_field_both(mask, lbl, field, ty0):
    """Partition foreground into EXACTLY `field` horses (k-means on pixel coords). Return TWO centroid
    variants per horse:
      A = PIXEL centroid of the cluster (pixel-weighted; leans to the bigger fragment).
      B = unweighted mean of the per-CONNECTED-COMPONENT-piece centroids in the cluster (= midpoint
          of the two blobs for a silk-split; identical to A for a single-blob horse).
    """
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
    dets_a, dets_b = [], []
    for j in range(k):
        m = labels == j
        sub = pts[m]
        cj = comp[m]
        if len(sub) < 3:
            continue
        c = sub - sub.mean(0)
        _u, _sg, vt = np.linalg.svd(c, full_matrices=False)
        ang = float(np.arctan2(vt[0, 1], vt[0, 0]))
        dets_a.append((float(sub[:, 0].mean()), float(sub[:, 1].mean() + ty0), ang, len(sub)))
        ccents = [[sub[cj == c2, 0].mean(), sub[cj == c2, 1].mean()] for c2 in np.unique(cj)]
        cc = np.array(ccents, float)
        dets_b.append((float(cc[:, 0].mean()), float(cc[:, 1].mean() + ty0), ang, len(sub)))
    return dets_a, dets_b


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
        dets_a, dets_b, (mask, turf_hue, _t) = detect_edgefill(panel, ty0, ty1)
        # crop to field
        if dets_a:
            xs = [d[0] for d in dets_a]
            ys = [d[1] for d in dets_a]
            x0, x1 = max(0, int(min(xs) - 40)), min(panel.shape[1], int(max(xs) + 40))
            y0, y1 = max(0, int(min(ys) - 30)), min(panel.shape[0], int(max(ys) + 30))
        else:
            x0, y0, x1, y1 = 0, ty0, panel.shape[1], ty1
        orig = panel[y0:y1, x0:x1].copy()
        maskfull = np.zeros(panel.shape[:2], np.uint8)
        maskfull[ty0:ty0 + mask.shape[0]] = mask
        maskc = cv2.cvtColor(maskfull[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        boxed = orig.copy()
        # A = pixel centroid (RED), B = avg-of-blob-centroids (CYAN); line where they differ
        offs = []
        for (a, b) in zip(dets_a, dets_b):
            ax, ay = int(a[0] - x0), int(a[1] - y0)
            bx, by = int(b[0] - x0), int(b[1] - y0)
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            offs.append(d)
            if d > 0.7:
                cv2.line(boxed, (ax, ay), (bx, by), (255, 255, 255), 1)
            cv2.circle(boxed, (ax, ay), 3, (0, 0, 255), -1)        # A pixel-centroid
            cv2.circle(boxed, (bx, by), 2, (255, 255, 0), -1)      # B blob-avg
        for im, txt in [(orig, "ORIGINAL"), (maskc, "edge+fill mask"),
                        (boxed, f"{len(dets_a)} horses  RED=A pixel-centroid  CYAN=B blob-avg")]:
            cv2.putText(im, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        combo = np.hstack([orig, maskc, boxed])
        z = 3
        combo = cv2.resize(combo, (combo.shape[1] * z, combo.shape[0] * z),
                           interpolation=cv2.INTER_NEAREST)
        out = f"{DIAG}/cv_edgefill_f{f:04d}.png"
        cv2.imwrite(out, combo)
        nd = sum(1 for d in offs if d > 0.7)
        mx = max(offs) if offs else 0.0
        print(f"f{f}: {len(dets_a)} horses, turf-hue {turf_hue}; A-vs-B differ on {nd} horses "
              f"(max offset {mx:.1f}px) -> {out}")


if __name__ == "__main__":
    main()

"""Rail-fitting primitives shared by the per-feed observation models.

A running rail is the one geometric reference every feed shares: a horse's lateral
trip is its perpendicular offset from the inner rail. Three feeds read the rail three
ways (a fixed mount sees a stable straight line; the synthetic aerial render draws a
crisp curve; the panning drone reads the turf boundary), so this module keeps the three
distinct FIT strategies but shares the white-pixel detection + robust polyfit core.

See ``docs/CV_RECON.md`` for how the rail feeds the lateral reconstruction.

Pure numpy at import; cv2 is imported lazily inside the functions that need it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- shared core
def robust_polyfit(x: np.ndarray, y: np.ndarray, deg: int, *,
                   n_iter: int = 3, k_sigma: float = 2.0, min_keep: int = 10) -> np.ndarray:
    """Least-squares ``y ≈ poly(x)`` with iterative ``k_sigma`` residual outlier culling.

    The shared robust-fit core: a first fit, then up to ``n_iter`` re-fits dropping points
    whose residual exceeds ``k_sigma`` standard deviations, while at least ``min_keep`` remain.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    coeffs = np.polyfit(x, y, deg)
    for _ in range(n_iter):
        resid = y - np.polyval(coeffs, x)
        s = float(np.std(resid)) + 1e-6
        keep = np.abs(resid) < k_sigma * s
        if keep.sum() < min_keep:
            break
        coeffs = np.polyfit(x[keep], y[keep], deg)
    return coeffs


def white_rail_pixels(bgr: np.ndarray, *, v_min: float = 170.0, s_max: float = 60.0):
    """Crisp near-white, low-saturation rail pixels in a BGR image — the shared rail detector.

    Returns ``(xs, ys)`` float arrays of pixel coordinates. Both the fixed-mount patrol rail
    and the aerial render rail are drawn as bright low-sat lines; this is their common detector
    (callers restrict the ROI / pick the field-side band themselves).
    """
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = (gray > v_min) & (hsv[:, :, 1] < s_max)
    ys, xs = np.where(white)
    return xs.astype(float), ys.astype(float)


# ============================================================ fixed-mount straight-rail fit
@dataclass(frozen=True)
class FixedRail:
    """A constant straight rail ``x = a*y + b`` from a fixed head-on mount (patrol feed)."""

    a: float
    b: float
    vp_x: float
    vp_y: float
    slope_cv: float       # std(a)/|a| across the window — the fixed-camera stability metric
    intercept_std: float
    n_frames: int
    per_frame: list       # [(t, a, b), ...]


def fit_white_rail_line(bgr: np.ndarray, *, legend_box: tuple[float, float] = (0.10, 0.30),
                        y_top: float = 0.24, x_search: float = 0.45) -> tuple[float, float] | None:
    """Fit the LEFT white running-rail line ``x = a*y + b`` from one fixed-mount frame.

    Per scanline below ``y_top`` (fraction of height) the leftmost white pixel within the left
    ``x_search`` fraction is the rail edge; a robust degree-1 fit gives the line. ``legend_box``
    (h-frac, w-frac) blanks the top-left replay-legend overlay. Returns ``(a, b)`` or None.
    """
    import cv2

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = ((v > 175) & (s < 55)).astype(np.uint8)
    white[: int(legend_box[0] * h), : int(legend_box[1] * w)] = 0
    pts = []
    for y in range(int(y_top * h), h, 2):
        row = np.where(white[y, : int(x_search * w)] > 0)[0]
        if len(row):
            pts.append((row.min(), y))
    if len(pts) < 30:
        return None
    pts = np.array(pts, float)
    coeffs = robust_polyfit(pts[:, 1], pts[:, 0], 1, min_keep=30)
    return float(coeffs[0]), float(coeffs[1])


def fit_fixed_rail(frames) -> FixedRail | None:
    """Fit the rail ONCE as the median line across a fixed head-on window.

    ``frames`` is an iterable of ``(t, bgr)``. The camera is fixed, so one homography serves
    the whole window: take the per-frame line fits and use their median. The slope CV
    (``std(a)/|a|``) reports stability; the vanishing point is taken at the image-top horizon
    (``y=0``) as the along-track foreshortening pole.
    """
    slopes, intercepts, per = [], [], []
    for t, fr in frames:
        r = fit_white_rail_line(fr)
        if r:
            slopes.append(r[0])
            intercepts.append(r[1])
            per.append((float(t), r[0], r[1]))
    if not slopes:
        return None
    slopes, intercepts = np.array(slopes), np.array(intercepts)
    a_med, b_med = float(np.median(slopes)), float(np.median(intercepts))
    vp_y = 0.0
    vp_x = a_med * vp_y + b_med
    cv = float(np.std(slopes) / abs(a_med)) if a_med else float("nan")
    return FixedRail(a_med, b_med, vp_x, vp_y, cv, float(np.std(intercepts)), len(slopes), per)


def fit_homography_from_rail(rail: FixedRail, h: int, w: int, mpp_near: float,
                             track_width_m: float) -> np.ndarray:
    """Ground-plane homography ``image(x,y) -> track(off_m, depth)`` from ONE rail + its VP.

    The rail is ``off=0``, a constant-lateral ground line through the VP. A parallel line at
    ``off = track_width_m`` shares the VP; four coplanar correspondences give an exact
    perspective homography. ``mpp_near`` = px per metre at the near scanline (a scale anchor;
    a within/between RATIO read is scale-free regardless).
    """
    import cv2

    y_near = 0.90 * h
    y_far = 0.45 * h
    xl_near = rail.a * y_near + rail.b
    xl_far = rail.a * y_far + rail.b
    width = track_width_m
    dx_near = width * mpp_near
    conv = (y_near - rail.vp_y) / (y_far - rail.vp_y)
    dx_far = dx_near / conv
    src = np.array(
        [[xl_near, y_near], [xl_near + dx_near, y_near],
         [xl_far, y_far], [xl_far + dx_far, y_far]], np.float32,
    )
    dst = np.array([[0.0, 0.0], [width, 0.0], [0.0, 1.0], [width, 1.0]], np.float32)
    hmat, _ = cv2.findHomography(src, dst)
    return hmat


def lateral_from_homography(hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a ground-plane homography to ``(N, 2)`` image points -> ``(N, 2)`` track coords."""
    if len(pts) == 0:
        return np.empty((0, 2))
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = (hmat @ p.T).T
    return q[:, :2] / q[:, 2:3]


# ============================================================ aerial-render spline rail fit
@dataclass(frozen=True)
class RailSpline:
    """A smooth rail curve for one render frame, in absolute panel-pixel coords.

    Sampled as a dense polyline ``(px, py)`` ordered along the rail with cumulative pixel
    arc-length ``s_px``. :meth:`project` returns ``(s_px, signed_perp_px)`` for an image point.
    """

    px: np.ndarray
    py: np.ndarray
    s_px: np.ndarray
    valid: bool
    n_px: int

    def project(self, x: float, y: float, track_ref) -> tuple[float, float]:
        """Nearest rail arc-length + signed perpendicular offset (positive = track side)."""
        d2 = (self.px - x) ** 2 + (self.py - y) ** 2
        k = int(np.argmin(d2))
        k0, k1 = max(0, k - 1), min(len(self.px) - 1, k + 1)
        tx, ty = self.px[k1] - self.px[k0], self.py[k1] - self.py[k0]
        tn = float(np.hypot(tx, ty)) + 1e-9
        tx, ty = tx / tn, ty / tn
        rx, ry = x - self.px[k], y - self.py[k]
        perp = rx * (-ty) + ry * tx
        ref_perp = (track_ref[0] - self.px[k]) * (-ty) + (track_ref[1] - self.py[k]) * tx
        s = 1.0 if ref_perp >= 0 else -1.0
        return float(self.s_px[k]), float(s * perp)


def _rail_white_pts(panel: np.ndarray, ty0: int, ty1: int):
    """White rail pixels in the render track band, in PANEL-relative coords."""
    import cv2

    band = panel[ty0:ty1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (gray > 170) & (hsv[:, :, 1] < 60) & (gray > 0)
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.where(white)
    return xs.astype(float), (ys + ty0).astype(float)


def fit_rail_spline(panel: np.ndarray, ty0: int, ty1: int, field_centroid) -> RailSpline:
    """Fit the INNER rail of a noise-free render as a smooth arc-length-parameterised polyline.

    The render draws the rail as a crisp white curve; a per-scanline nearest-band trace (the
    band nearest the field, since runners hug the inner rail) + a robust cubic is exact. Returns
    a :class:`RailSpline`; ``valid=False`` if too few rail pixels.
    """
    xs, ys = _rail_white_pts(panel, ty0, ty1)
    if len(xs) < 80:
        return RailSpline(np.array([]), np.array([]), np.array([]), False, len(xs))
    fcx, fcy = field_centroid
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    axis = "x" if span_x >= span_y else "y"

    def bands(vals):
        vals = np.sort(vals)
        out, start, prev = [], vals[0], vals[0]
        for v in vals[1:]:
            if v - prev > 6:
                out.append((start + prev) / 2.0)
                start = v
            prev = v
        out.append((start + prev) / 2.0)
        return out

    maj, mnr = [], []
    if axis == "x":
        for cx in range(int(xs.min()), int(xs.max()) + 1, 3):
            sel = np.abs(xs - cx) <= 1.5
            if sel.sum() == 0:
                continue
            bc = bands(ys[sel])
            maj.append(cx)
            mnr.append(min(bc, key=lambda c: abs(c - fcy)))
    else:
        for cy in range(int(ys.min()), int(ys.max()) + 1, 3):
            sel = np.abs(ys - cy) <= 1.5
            if sel.sum() == 0:
                continue
            bc = bands(xs[sel])
            maj.append(cy)
            mnr.append(min(bc, key=lambda c: abs(c - fcx)))
    if len(maj) < 12:
        return RailSpline(np.array([]), np.array([]), np.array([]), False, len(xs))
    maj = np.array(maj, float)
    mnr = np.array(mnr, float)
    coeffs = robust_polyfit(maj, mnr, 3, n_iter=1, k_sigma=2.5)
    gg = np.linspace(maj.min(), maj.max(), 200)
    hh = np.polyval(coeffs, gg)
    px, py = (gg, hh) if axis == "x" else (hh, gg)
    s_px = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    return RailSpline(px, py, s_px, True, len(xs))


# ===================================================== aerial-render rigid field-selected rail
def candidate_rail_curves(band: np.ndarray, ty0: int, white_xs: np.ndarray, white_ys: np.ndarray,
                          *, min_extent: int = 40, deg: int = 2) -> list[np.ndarray]:
    """ALL candidate rail lines in a band: each large white connected-component → its own curve.

    The disambiguation primitive for the OCCLUDED-inner-rail problem (the inner running rail is
    hidden by the field running on it, so a single "nearest line" fit latches the outer rail). This
    enumerates EVERY coherent white line (running rail, outer boundary, markers) as a robust polyline
    in PANEL coords, so a downstream selector (geometry/odometer prior, or temporal track) can pick
    the inner one. Returns a list of ``(N,2)`` polylines; components shorter than ``min_extent`` (rail
    dashes) are dropped.
    """
    import cv2

    if len(white_xs) < 30:
        return []
    h, w = band.shape[:2]
    m = np.zeros((h, w), np.uint8)
    yy, xx = white_ys.astype(int), white_xs.astype(int)
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    m[yy[ok], xx[ok]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        ext = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if ext < min_extent:
            continue
        ys_i, xs_i = np.where(lbl == i)
        pc = np.column_stack([xs_i, ys_i]).astype(float)
        horiz = np.ptp(pc[:, 0]) >= np.ptp(pc[:, 1])
        a, b = (pc[:, 0], pc[:, 1]) if horiz else (pc[:, 1], pc[:, 0])
        if len(a) < 12:
            continue
        coeffs = np.polyfit(a, b, min(deg, 2))
        ga = np.linspace(a.min(), a.max(), 120)
        gb = np.polyval(coeffs, ga)
        out.append(np.column_stack([ga, gb + ty0]) if horiz else np.column_stack([gb, ga + ty0]))
    return out


def fit_rail_curve(band: np.ndarray, ty0: int, white_xs: np.ndarray, white_ys: np.ndarray, *,
                   field_ctr=None, min_extent: int = 40, inlier: float = 12.0,
                   deg: int = 2) -> np.ndarray | None:
    """Rigid CURVE fit to the rail candidates, FIELD-SELECTED.

    The field centroid is used ONLY to SELECT which coherent white line is the running rail (the one
    nearest the field — there are several: running rail, outer/foreground boundary, markers); the
    curve POSITION then comes entirely from that line's own pixels. Components (extent≥``min_extent``
    drops dashes) → seed = the line nearest the field (else the largest) → robust quadratic → gather
    collinear inliers (within ``inlier`` px, so an occlusion-broken rail re-joins while the far
    boundary/markers are excluded) → refit. ``deg`` 1 = straight, 2 = arc (caller picks by geometry).
    Returns a panel-coord ``(N,2)`` polyline or ``None``. Pixel coords are band-relative on input;
    ``ty0`` shifts the output back to panel coords.
    """
    import cv2

    if len(white_xs) < 50:
        return None
    h, w = band.shape[:2]
    m = np.zeros((h, w), np.uint8)
    yy, xx = white_ys.astype(int), white_xs.astype(int)
    ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    m[yy[ok], xx[ok]] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        ext = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if ext >= min_extent:
            ys_i, xs_i = np.where(lbl == i)
            comps.append(np.column_stack([xs_i, ys_i]).astype(float))
    if not comps:
        return None
    if field_ctr is not None:
        fx, fy = field_ctr[0], field_ctr[1] - ty0
        seed = min(comps, key=lambda pc: float(np.min((pc[:, 0] - fx) ** 2 + (pc[:, 1] - fy) ** 2)))
    else:
        seed = max(comps, key=len)
    horiz = np.ptp(seed[:, 0]) >= np.ptp(seed[:, 1])

    def axes(pc):
        return (pc[:, 0], pc[:, 1]) if horiz else (pc[:, 1], pc[:, 0])
    a, b = axes(seed)
    if len(a) < 12:
        return None
    deg = min(deg, 2)
    coeffs = np.polyfit(a, b, deg)
    all_pts = np.vstack(comps)
    aa, bb = axes(all_pts)
    for _ in range(3):
        kp = np.abs(bb - np.polyval(coeffs, aa)) < inlier
        if kp.sum() < 20:
            break
        coeffs = np.polyfit(aa[kp], bb[kp], deg)
    kp = np.abs(bb - np.polyval(coeffs, aa)) < inlier
    if kp.sum() < 20:
        return None
    ga = np.linspace(aa[kp].min(), aa[kp].max(), 120)
    gb = np.polyval(coeffs, ga)
    return np.column_stack([ga, gb + ty0]) if horiz else np.column_stack([gb, ga + ty0])


def project_to_polyline(curve: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Nearest-vertex projection of ``(x,y)`` onto a rail polyline → ``(s_px along rail, signed perp)``.

    ``s`` is cumulative pixel arc-length to the nearest vertex; ``perp`` is the signed perpendicular
    (image-left normal positive). The longitudinal running order is ``argsort(s)``; ``|perp|`` bounds
    the lane (off-track blobs sit far from the rail). Mirrors :meth:`RailSpline.project` for the
    field-selected curve.
    """
    s_px = np.r_[0.0, np.cumsum(np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1])))]
    k = int(np.argmin((curve[:, 0] - x) ** 2 + (curve[:, 1] - y) ** 2))
    k0, k1 = max(0, k - 1), min(len(curve) - 1, k + 1)
    tx, ty = curve[k1, 0] - curve[k0, 0], curve[k1, 1] - curve[k0, 1]
    tn = float(np.hypot(tx, ty)) + 1e-9
    perp = (x - curve[k, 0]) * (-ty / tn) + (y - curve[k, 1]) * (tx / tn)
    return float(s_px[k]), float(perp)


# ============================================================ panning-drone pooled-window rail
def fit_stable_rail(rail_pts_list) -> dict:
    """Pool rail-boundary points across a near-static WINDOW of frames and fit ONE rail line.

    The panning-drone analog of the render's stable spline: rather than re-fit the turf boundary
    every frame (the per-frame instability), pool the boundary points over a roughly-static window
    and fit a single principal axis (PCA). Returns ``{mean, axis, perp, straight, across_std}``;
    ``straight`` in [0, 1] (1 = straight in-frame) classifies the window straight-vs-bend.
    """
    pts = np.vstack(rail_pts_list)
    c = pts - pts.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    perp_axis = vt[1]
    along = c @ axis
    across = c @ perp_axis
    straight = 1.0 - min(1.0, float(np.std(across)) / (float(np.std(along)) + 1e-6))
    return {"mean": pts.mean(0), "axis": axis, "perp": perp_axis,
            "straight": straight, "across_std": float(np.std(across))}


def rail_perp(rail: dict, x: float, y: float) -> float:
    """Signed perpendicular offset (px) of ``(x, y)`` from a pooled-window :func:`fit_stable_rail`."""
    v = np.array([x, y], float) - rail["mean"]
    return float(v @ rail["perp"])

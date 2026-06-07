"""Aerial-render rail detection + RUNNING-rail selection (the racing inner rail).

Promoted from ``studies/cv_rail_robust.py`` (the 2026-06 rail-backbone work; journal 0099). The AVV
right panel draws the rails as bright lines, but the per-frame fit must survive camera angle ROTATION
(the rail goes diagonal / near-vertical), BACKGROUND clutter entering as the camera pans (grandstand
apron / hedge -> big bright blobs), and the rail's SHADOW / a parallel marker line (a near-parallel
twin). There is NO occlusion — the field runs on the turf BESIDE the rail, not on it.

Pipeline:
  ``line_candidates``   -> every bright connected component in the track band, with shape stats
                           (elongation, mean width) classifying THIN rail pieces vs fat blobs/text.
  ``fit_curve_pca``     -> orientation-free robust curve: parametrise along the PCA major axis and fit
                           minor = poly(major). No x-vs-y switch, so the perpendicular SIGN is stable.
  ``enumerate_lines``   -> grow every distinct rail line by gathering collinear thin pixels (dashes
                           re-joined), each with descriptors (coverage/solidity, curvature, brightness).
  ``select_running_rail`` -> of the two main rails (top-2 coverage), pick the one the field hugs by
                           MEDIAN distance to the (rail-independent) detect_edgefill blobs, run
                           TEMPORALLY (carry the previous pick; let field distance override only when
                           confident) so equidistant-field bend frames don't flip. Returns a
                           tangent-oriented curve so ``geometry_cv.project_to_polyline``'s perp sign is
                           consistent across the race.

Pure numpy at import; cv2 is imported lazily inside the functions that need it. See ``docs/CV_RECON.md``.
"""
from __future__ import annotations

import numpy as np

# shape gates (data-driven, stated): a rail line is long, thin, straight-ish
MIN_EXT = 60          # px; shorter coherent whites are dashes/markers (re-joined later via inliers)
MIN_ELONG = 6.0       # PCA major/minor axis ratio; field + background blobs are fat (<~6)
MAX_WIDTH = 10.0      # area/extent = mean line width (px); rails ~3, blobs/text/aprons 20-70
LEGEND_Y = 0.70       # fraction of panel height below which the silk legend strip sits -> excluded


# --------------------------------------------------------------------------- candidate extraction
def _white_mask(band: np.ndarray):
    import cv2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = ((gray > 165) & (hsv[:, :, 1] < 70)).astype(np.uint8)
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), gray


def line_candidates(panel: np.ndarray, ty0: int, ty1: int) -> list[dict]:
    """All bright connected components in the track band ``[ty0, ty1)``, with shape stats.

    Each dict carries pixels in PANEL coords plus ``ext``/``elong``/``width``/``meanV`` and the
    classification flags ``is_thin`` (a rail piece incl. a short dash) and ``is_line`` (a long
    coherent rail segment). Fat blobs (the field cluster, grandstand apron, painted text) fail the
    thin gate and drop out.
    """
    import cv2

    band = panel[ty0:ty1]
    ph = panel.shape[0]
    mask, gray = _white_mask(band)
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        ext = max(w, h)
        if ext < 25:
            continue
        ys, xs = np.where(lbl == i)
        cy_panel = float(ys.mean()) + ty0
        cc = np.column_stack([xs, ys]).astype(float)
        cc -= cc.mean(0)
        sv = np.linalg.svd(cc, full_matrices=False, compute_uv=False)
        elong = float(sv[0] / (sv[1] + 1e-6)) if sv.size > 1 and sv[1] > 1e-6 else 999.0
        width = area / float(ext + 1e-6)        # mean line width: thin for rails, fat for blobs/text
        in_legend = cy_panel > LEGEND_Y * ph
        is_thin = (ext >= 18) and (elong >= 4.0) and (width <= MAX_WIDTH) and not in_legend
        is_line = (ext >= MIN_EXT) and (elong >= MIN_ELONG) and (width <= MAX_WIDTH) and not in_legend
        out.append(dict(xs=xs.astype(float), ys=ys.astype(float) + ty0, ext=ext, elong=elong,
                        width=width, meanV=float(gray[ys, xs].mean()), is_thin=is_thin,
                        ctr=(float(xs.mean()), cy_panel), is_line=is_line))
    return out


# --------------------------------------------------------------------------- orientation-free fit
def fit_curve_pca(xs, ys, deg: int = 2, n_iter: int = 3, k_sigma: float = 2.0):
    """Orientation-free robust curve: parametrise along the PCA major axis, fit minor = poly(major).

    Returns ``(polyline (N,2), frame)`` where ``frame=(mu,u,v,coeffs,t_lo,t_hi)`` is the fitted PCA
    frame (so the curve can be extrapolated and used to gather collinear pixels). Handles diagonal /
    near-vertical rail with no x-vs-y switch (the source of the perp sign-flips).
    """
    pp = np.column_stack([xs, ys]).astype(float)
    mu = pp.mean(0)
    qq = pp - mu
    _u, _s, vt = np.linalg.svd(qq, full_matrices=False)
    u, v = vt[0], vt[1]                       # major, minor axes
    t = qq @ u
    wv = qq @ v
    coeffs = np.polyfit(t, wv, deg)
    for _ in range(n_iter):
        resid = wv - np.polyval(coeffs, t)
        s = float(np.std(resid)) + 1e-6
        keep = np.abs(resid) < k_sigma * s
        if keep.sum() < max(20, deg + 2):
            break
        coeffs = np.polyfit(t[keep], wv[keep], deg)
    t_lo, t_hi = float(t.min()), float(t.max())
    tg = np.linspace(t_lo, t_hi, 160)
    pts = mu[None, :] + tg[:, None] * u[None, :] + np.polyval(coeffs, tg)[:, None] * v[None, :]
    return pts, (mu, u, v, coeffs, t_lo, t_hi)


def _gather_inliers(cands, frame, band_px: float = 12.0, t_margin: float = 600.0):
    """THIN pixels within ``band_px`` PERPENDICULAR of the (extrapolated) fitted curve.

    Works in the fit's PCA frame so collinear dashes BEYOND the seed's own span are captured (the
    extrapolation along the rail), while a fat background blob is excluded (thin-only).
    """
    mu, u, v, coeffs, t_lo, t_hi = frame
    keep_x, keep_y = [], []
    for cnd in cands:
        if not cnd.get("is_thin"):
            continue
        qq = np.column_stack([cnd["xs"], cnd["ys"]]) - mu
        t = qq @ u
        w = qq @ v
        m = (np.abs(w - np.polyval(coeffs, t)) < band_px) & (t > t_lo - t_margin) & (t < t_hi + t_margin)
        if m.any():
            keep_x.append(cnd["xs"][m])
            keep_y.append(cnd["ys"][m])
    if not keep_x:
        return np.array([]), np.array([])
    return np.concatenate(keep_x), np.concatenate(keep_y)


# --------------------------------------------------------------------------- line enumeration
def enumerate_lines(cands, deg: int, gray=None) -> list[dict]:
    """Grow every distinct rail line from the thin components (dashes merged into full lines).

    Seed from each not-yet-absorbed thin component (longest first), grow it by collinear gather, and
    record which components it absorbed — so the dashed and the solid rails both come out as separate
    full-length candidates. Each line gets descriptors: ``coverage`` (support px / length px; a
    solidity that separates real rails ~[1.5,5] from sparse fit-artifacts and the fat field blob),
    ``curv`` (leading quadratic coeff), ``meanV`` (brightness if ``gray`` given), ``mid``.
    """
    thin = sorted([c for c in cands if c["is_thin"]], key=lambda c: -c["ext"])
    used = [False] * len(thin)
    lines = []
    for i, seed in enumerate(thin):
        if used[i]:
            continue
        pts, frame = fit_curve_pca(seed["xs"], seed["ys"], deg=deg)
        for _ in range(3):
            gx, gy = _gather_inliers(thin, frame)
            if len(gx) < 30:
                break
            pts, frame = fit_curve_pca(gx, gy, deg=deg)
        mu, u, v, coeffs, t_lo, t_hi = frame
        npix = nabs = 0
        for j, c in enumerate(thin):           # absorb every comp whose centre lies on this curve
            q = np.array(c["ctr"]) - mu
            t = q @ u
            if abs((q @ v) - np.polyval(coeffs, t)) < 16 and t_lo - 600 < t < t_hi + 600:
                used[j] = True
                npix += len(c["xs"])
                nabs += 1
        curv = float(coeffs[0]) if len(coeffs) >= 3 else 0.0
        mean_v = float("nan")
        if gray is not None and len(pts):
            ix = np.clip(pts[:, 0].astype(int), 0, gray.shape[1] - 1)
            iy = np.clip(pts[:, 1].astype(int), 0, gray.shape[0] - 1)
            mean_v = float(gray[iy, ix].mean())
        coverage = npix / float(t_hi - t_lo + 1e-6)
        gx, gy = _gather_inliers(thin, frame)   # thinness: perp spread of the line's own pixels
        if len(gx) >= 10:
            qq = np.column_stack([gx, gy]) - mu
            spread = float(np.median(np.abs((qq @ v) - np.polyval(coeffs, qq @ u))))
        else:
            spread = 99.0
        lines.append(dict(pts=pts, frame=frame, npix=npix, ndash=nabs, ext=float(t_hi - t_lo),
                          curv=curv, meanV=mean_v, coverage=coverage, spread=spread,
                          mid=(float(pts[:, 0].mean()), float(pts[:, 1].mean()))))
    return [ln for ln in lines if ln["ext"] > 80]    # drop stubs that never grew into a line


# --------------------------------------------------------------------------- running-rail selection
def _orient(pts, direction):
    """Return the polyline ordered so its end-to-start tangent aligns with ``direction``."""
    return pts[::-1] if float((pts[-1] - pts[0]) @ direction) < 0 else pts


def _median_dist_to_field(curve, field_pts):
    """MEDIAN nearest-distance (px) from the field blobs to a rail polyline.

    Median (not mean) so a scattered minority of background blobs that detect_edgefill misclassifies
    as field cannot drag the score — the majority real pack dominates.
    """
    d = ((field_pts[:, None, 0] - curve[None, :, 0]) ** 2
         + (field_pts[:, None, 1] - curve[None, :, 1]) ** 2)
    return float(np.median(np.sqrt(d.min(1))))


def _curve_dist(a, b):
    """Median nearest-vertex distance between two polylines (px) — the temporal continuity score."""
    d = ((a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2)
    return float(np.median(np.sqrt(d.min(1))))


def select_running_rail(cands, deg, field_pts, prev_tang=None, prev_curve=None, conf=0.7):
    """Pick the RUNNING rail (racing 'inner') — temporally smooth, field-corrected.

    Of the two main rails (top-2 coverage), the running rail is the one the field hugs. Field
    proximity (median distance to the detect_edgefill blobs) is non-circular: those blobs come from
    the green-turf ``track_band``, not the rail; the median resists the scattered background
    false-positives the detector emits. But on some bend viewpoints the field sits ~equidistant from
    both rails and the nearest flips frame-to-frame (perp spikes), so we run TEMPORALLY: keep the rail
    nearest the previous frame's pick and let field distance OVERRIDE only when confident (one rail
    clearly closer: ``min_dist < conf * max_dist``). A frame whose nearest rail is implausibly far
    (>180 px; a bad fit) is skipped. Concavity is deliberately NOT used — it flips as the camera pans a
    bend. The returned curve is tangent-oriented (to ``prev_tang``) so the perp SIGN stays consistent.

    Returns ``(curve_pts, tang_out, info)`` or ``(None, prev_tang, None)``.
    """
    if field_pts is None or len(field_pts) == 0:
        return None, prev_tang, None
    alllines = enumerate_lines(cands, deg)
    lines = [ln for ln in alllines if 1.0 <= ln["coverage"] <= 5.0]
    if not lines:
        # solid-rail-only frame: the ceiling (which rejects the fat field-cluster line when a proper
        # rail coexists) left nothing -> admit high-coverage THIN lines (a real rail is thin).
        lines = [ln for ln in alllines if ln["coverage"] >= 1.0 and ln["spread"] <= 5.0]
    if not lines:
        return None, prev_tang, None
    main = sorted(lines, key=lambda ln: -ln["coverage"])[:2]
    dists = [_median_dist_to_field(ln["pts"], field_pts) for ln in main]
    i_near = int(np.argmin(dists))
    if dists[i_near] > 150.0:        # nearest rail implausibly far from the field -> bad fit, skip
        return None, prev_tang, None  # (applies to the single-rail case too: kills the f1280 stub)
    if len(main) == 1:
        pick, basis = 0, "single"
    elif prev_curve is None:
        pick, basis = i_near, "field-init"
    else:
        carry = int(np.argmin([_curve_dist(ln["pts"], prev_curve) for ln in main]))
        confident = dists[i_near] < conf * max(dists)   # one rail clearly closer -> trust field
        pick, basis = (i_near, "field-correct") if (confident and carry != i_near) else (carry, "carry")
    curve = main[pick]["pts"]
    if prev_tang is not None:
        curve = _orient(curve, prev_tang)
    tang = curve[-1] - curve[0]
    tang = tang / (np.hypot(*tang) + 1e-9)
    return curve, tang, {"basis": basis}

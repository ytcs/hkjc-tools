"""Aerial-render rail detection, RUNNING-rail selection, and full rail RECONSTRUCTION.

The AVV right panel draws the rails as bright lines. The per-frame fit must survive camera-angle
ROTATION (the rail goes diagonal / near-vertical), BACKGROUND clutter as the camera pans (grandstand
apron / hedge -> big bright blobs), and the rail's SHADOW / a parallel marker line (a near-parallel
twin). There is NO occlusion — the field runs on the turf BESIDE the rail, not on it.

This module has three layers (one rail home — no module sprawl):

PRIMITIVES (shape gates + orientation-free curve fitting, reused everywhere incl. ``aerial_detect``):
  ``line_candidates``   -> every bright connected component in the track band, with shape stats
                           (elongation, mean width) classifying THIN rail pieces vs fat blobs/text.
  ``fit_curve_pca``     -> orientation-free robust curve: parametrise along the PCA major axis and fit
                           minor = poly(major). No x-vs-y switch, so the perpendicular SIGN is stable.
  ``enumerate_lines``   -> grow every distinct rail line by gathering collinear thin pixels (dashes
                           re-joined), each with descriptors (coverage/solidity, curvature, brightness).

RECONSTRUCTION (the 0115/0116 redesign — the CURRENT promoted path, validated R08):
  ``clean_pixels``      -> LOOP 1: candidate-pixel precision by a LONG-SMOOTH-CURVE gate (turf-relative
                           colour + colour-context rejection + adaptive spread). Replaces the per-pixel
                           tagger AND the bottom-legend cut.
  ``track_sequence``    -> LOOP 2: per-frame two-rail fit by temporal fit+gather+refit (carry the prev
                           curve as the gather SEED, refit on the CURRENT frame's pixels -> bridges
                           occluded bends with no smearing). Rails fit INDEPENDENTLY; partner accepted
                           only if it does not CROSS the running rail (the one bend-robust discriminator).
  ``build_kappa`` + ``backproject_fixed_r`` -> GLOBAL geometry: reconstruct the rail's signed
                           image-curvature profile kappa(s) vs odometer arclength, pin R = 1/kappa(s)
                           per frame, fit only the pose, draw across the box -> full span even from a
                           sliver / single rail (config-A). All pixel constants are fractions of the
                           content width and colour gates are turf-MEDIAN-relative (robust on bends),
                           so the pipeline survives render drift. See journals 0115 / 0116.

LEGACY (``select_running_rail`` — the per-frame field-proximity selector, superseded by
  ``track_sequence``). Retained only because the not-yet-ablated full aerial pipeline
  (``studies/cv_aerial_track_id.py``, ``cv_pose_backbone.py``) still rides on it; slated for removal
  once the reconstruction layer is wired into that pipeline (CV_RECON §10 roadmap: integrate-then-ablate).

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


OCC_NBINS = 20            # along-length bins for the continuity (occupancy) metric


def _occupancy(t, t_lo, t_hi, nb: int = OCC_NBINS) -> float:
    """Fraction of ``nb`` equal bins along the line's MAJOR axis that hold >=1 support pixel.

    A rail is ONE continuous curve, so it fills ~all bins (occ ~0.9-1.0) even when dashed (gaps < a
    bin) or noisy; HUD/disclaimer text scatters into words/glyphs with large gaps (occ ~0.3-0.5). This
    is the owner's continuity discriminator. Independent of thinness, so it certifies a THICK but
    continuous arc (config-A bend rail) that the spread gate alone would reject."""
    if len(t) == 0 or t_hi <= t_lo:
        return 0.0
    b = np.clip(((np.asarray(t) - t_lo) / (t_hi - t_lo + 1e-9) * nb).astype(int), 0, nb - 1)
    return len(np.unique(b)) / float(nb)


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
            tt = qq @ u
            spread = float(np.median(np.abs((qq @ v) - np.polyval(coeffs, tt))))
            occ = _occupancy(tt, t_lo, t_hi)    # along-length continuity: rail ~1, gappy text ~0.3-0.5
        else:
            spread, occ = 99.0, 0.0
        lines.append(dict(pts=pts, frame=frame, npix=npix, ndash=nabs, ext=float(t_hi - t_lo),
                          curv=curv, meanV=mean_v, coverage=coverage, spread=spread, occ=occ,
                          mid=(float(pts[:, 0].mean()), float(pts[:, 1].mean()))))
    return [ln for ln in lines if ln["ext"] > 80]    # drop stubs that never grew into a line


# ===================== LEGACY: per-frame running-rail selection (superseded by track_sequence) ======
# Retained only for the not-yet-ablated full aerial pipeline (studies/cv_aerial_track_id.py,
# cv_pose_backbone.py). The current promoted path is clean_pixels + track_sequence + build_kappa below.
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
        return None, prev_tang, None  # (also rejects a lone far stub in the single-rail case)
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
    # expose the OTHER main rail (the outer): it sits a KNOWN 22.5 m from the inner, so the local
    # inner->outer pixel separation is the metric ruler for a perspective-correct y_lat.
    outer = main[1 - pick]["pts"] if len(main) == 2 else None
    return curve, tang, {"basis": basis, "outer": outer}


# =================================================================================================
# RECONSTRUCTION — the 0115/0116 redesign (current promoted path). Pixel cleaning -> temporal
# fit-gather two-rail tracker -> global kappa(s) back-projection. All pixel constants are fractions
# of content width W; colour gates are turf-MEDIAN-relative (robust on bends — tying to the std
# over-tightens where the band includes dark fence/shadows, journal 0116 addendum 5). Factors are
# calibrated to reproduce the prior absolute values at the current AVV scale (W~949, turf gray~150).
# =================================================================================================

# --- shared adaptive-spread + coverage gate (identical in LOOP 1 and LOOP 2) ---
SPREAD_BASE_FAC = 0.0042    # rail max thickness, fraction of W (~4 px @949) -- this thin = a rail outright
SPREAD_MARGIN_FAC = 0.0021  # ADAPTIVE tolerance above the frame's OWN thinnest spanning line (~2 px @949)
SPREAD_CEIL_FAC = 0.0095    # hard thinness ceiling, fraction of W (~9 px @949) -- beyond this it is a 2D block
COV_SOLID = 2.0             # min coverage (support px / length) in the loosened regime -- rejects sparse text rows
# --- LOOP 1 (pixel cleaning) ---
PIX_MIN_EXT_FRAC = 0.33     # a kept curve must span >= this fraction of W
S_THIN_FAC = 0.00126        # crisp-rail thinness FLOOR, fraction of W (~1.2 px @949): a spanning line THIS
                            # thin IS a rail regardless of continuity (rescues R08 bend rails that go gappy
                            # -- occ~0.6 -- but stay razor-thin spread~0.7). Set below R02A thin-text (~1.6).
OCC_HI = 0.80               # continuity FLOOR: a thick/noisy line is a rail only if it is one CONTINUOUS
                            # curve (config-A bend arc occ~1.0); HUD/disclaimer text scatters to occ<=0.7.
PIX_MINCOMP_FAC = 0.008     # min thin-component length, fraction of W (~8 px @949)
BRIGHT_FAC = 0.19           # brightness threshold = turf_gray_median * (1 + BRIGHT_FAC); turf-MEDIAN-relative
WHITE_S_FAC = 0.58          # rail whiteness: saturation < this * turf_saturation_median (~75 @ R08)
VIVID_S_FAC = 1.08          # vivid silk/legend: saturation > this * turf_saturation_median (~140 @ R08) + hue far
VIVID_V_FAC = 0.50          # vivid silk/legend: value > this * turf_value_median (~80 @ R08)
ADJ_FAC = 0.017             # silk-adjacency reject radius, fraction of W (~16 px @949)
# --- LOOP 2 (rail fit) + global geometry ---
FIT_MIN_EXT_FRAC = 0.30     # a line must span >= this fraction of W to be a rail
FIT_MINCOMP_FAC = 0.0126    # min connected-component length in the fit enumerate, ~12 px @949
MIN_RAIL_FAC = 0.026        # min rail pixel count, ~25 @949
GATHER_FAC = 0.0084         # gather band, ~8 px @949
SEED_FAC = 0.0126           # temporal seed band, ~12 px @949
NOCROSS_FAC = 0.006        # min rail-rail gap: reject coincident/crossing (gap~0); ALLOW a close parallel pair (~6px+@949). Config-A near+far rails sit ~15px apart -> must pass.
MAXSEP_FAC = 0.36           # max plausible rail separation, ~340 px @949
RLINE_FAC = 8.4             # straight-vs-arc radius threshold, ~8000 px @949 (x W)
PAIR_SEP_FAC = 0.011        # min near<->far rail separation to RESOLVE a close pair, ~10 px @949. The
                            # venue has a fixed FAR + movable NEAR rail; in config A they sit ~15-20 px apart
                            # on straights (joined by cross-braces, fit as ONE band) and must be split back
                            # into two. Below this they are treated as coincident (one observable rail).


# --------------------------------------------------------------------------- LOOP 1: pixel cleaning
def _thin_components(panel, y0, y1, content_w):
    """Bright THIN connected components, with pixels + shape stats. NO legend cut. All thresholds are
    DATA-RELATIVE (no absolute px/colour magic numbers) so the algo survives render drift:
      * brightness  = turf_median * (1 + BRIGHT_FAC)       (turf-relative; robust to bend shadows)
      * whiteness   = s < WHITE_S_FAC * turf_sat_median    (rail is far less saturated than turf)
      * vivid colour= s > VIVID_S_FAC * turf_sat_median AND v > VIVID_V_FAC * turf_val_median AND
                      hue far (>20 deg) from turf -> silks / legend swatches
    COLOUR-CONTEXT rejection: drop white within ADJ_FAC*W px of a vivid non-turf blob (legend sleeve /
    jockey silk sits beside vivid colour; a real rail sits on turf). ADJ and the min-component length
    are fractions of the content width W -> resolution-invariant."""
    import cv2
    band = panel[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    s, v, h = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    turf = s > 40                                         # coloured (turf) pixels for the per-frame stats
    turf_g = float(np.median(gray[turf])) if turf.any() else 120.0
    turf_s = float(np.median(s[turf])) if turf.any() else 130.0
    turf_v = float(np.median(v[turf])) if turf.any() else 150.0
    thr = turf_g * (1.0 + BRIGHT_FAC)                     # rail brighter than turf by a turf-relative margin
    turf_h = float(np.median(h[turf & (s < 120)])) if (turf & (s < 120)).any() else 60.0
    dh = np.minimum(np.abs(h.astype(int) - turf_h), 180 - np.abs(h.astype(int) - turf_h))
    vivid = ((s > VIVID_S_FAC * turf_s) & (v > VIVID_V_FAC * turf_v) & (dh > 20)).astype(np.uint8)
    adj = max(3, int(round(ADJ_FAC * content_w)))
    vivid_near = cv2.dilate(vivid, np.ones((2 * adj + 1, 2 * adj + 1), np.uint8))
    white = ((gray > thr) & (s < WHITE_S_FAC * turf_s) & (vivid_near == 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, st, _c = cv2.connectedComponentsWithStats(white, 8)
    min_ext = max(5, int(round(PIX_MINCOMP_FAC * content_w)))
    cands = []
    for i in range(1, n):
        ext = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        area = st[i, cv2.CC_STAT_AREA]
        if ext < min_ext or area / (ext + 1e-6) > 6.0:   # not thin (width ratio is dimensionless)
            continue
        ys, xs = np.where(lbl == i)
        cands.append(dict(xs=xs.astype(float), ys=ys.astype(float) + y0, is_thin=True,
                          ext=int(ext), ctr=(float(xs.mean()), float(ys.mean()) + y0)))
    return cands


def _gather_curve(cands, frame, band, iters=4):
    """Iterated fit->gather: gather thin pixels within ``band`` of the curve, refit a deg-2 on them, and
    re-gather -- so the parabola WALKS along a sharp arc that a single PCA fit clips. Grows monotonically;
    stops when the inlier set stabilises. Used only for the curving/striped (high-spread) config-A rails."""
    gx, gy = _gather_inliers(cands, frame, band_px=band)
    for _ in range(iters):
        if len(gx) < 20:
            break
        _, frame = fit_curve_pca(gx, gy, deg=2)
        ngx, ngy = _gather_inliers(cands, frame, band_px=band)
        if len(ngx) <= len(gx):
            break
        gx, gy = ngx, ngy
    return gx, gy


def clean_pixels(panel, y0, y1, content_w, deg=2):
    """Keep only pixels lying on a LONG, LOW-SPREAD smooth curve (a rail). Returns (kept_xy, dropped_xy)."""
    cands = _thin_components(panel, y0, y1, content_w)
    allpts = np.column_stack([np.concatenate([c["xs"] for c in cands]),
                              np.concatenate([c["ys"] for c in cands])]) if cands else np.zeros((0, 2))
    if not cands:
        return allpts, allpts
    s_base, s_ceil = SPREAD_BASE_FAC * content_w, SPREAD_CEIL_FAC * content_w
    s_thin = S_THIN_FAC * content_w
    lines = enumerate_lines(cands, deg)
    spanning = [ln for ln in lines if ln["ext"] >= PIX_MIN_EXT_FRAC * content_w]
    kept_xy = []
    if spanning:
        for ln in spanning:
            # A rail is THIN and/or CONTINUOUS. Two first-principles classes of genuine rail (validated
            # across both venues + all phases by the line-descriptor probe, journal next):
            #   * CRISP: spread <= s_thin -- razor-thin spanning lines ARE rails, even when the bend cuts
            #     them gappy (R08 f734/f2158 rails: spread ~0.7, occ ~0.6). Text never gets this thin.
            #   * CONTINUOUS: occ >= OCC_HI -- a thick/noisy line that is ONE near-complete curve is a rail
            #     (config-A bend arc: spread ~4.5, occ ~1.0). HUD/disclaimer text scatters to occ <= 0.7.
            # The 2D-block ceiling (s_ceil) drops a dense field/grandstand blob that is continuous but fat.
            # This REPLACES the old coverage+adaptive-thr gate, which let thin text leak (it passed the
            # thin test) AND let a thin TEXT line set the adaptive thr low, knife-edging the real rail out.
            if (ln["spread"] <= s_thin or ln["occ"] >= OCC_HI) and ln["spread"] <= s_ceil:
                if ln["spread"] <= s_base:                 # CRISP rail (R08): the single deg-2 gather suffices
                    gx, gy = _gather_inliers(cands, ln["frame"])
                else:                                      # CURVING/striped rail (config-A bends): a single
                    # deg-2 PCA fit can't follow a sharp arc, so the 12 px perp gather keeps only the straight
                    # middle and the curved ends are dropped (R02A f1700: 317 of 4500 rail px). Iterate
                    # fit->gather with a spread-scaled band so the parabola WALKS along the arc and recovers
                    # the ends. R08 never enters this branch (its rails are below s_base), so it is unchanged.
                    gx, gy = _gather_curve(cands, ln["frame"], band=max(25.0, 8.0 * ln["spread"]))
                if len(gx):
                    kept_xy.append(np.column_stack([gx, gy]))
    kept = np.unique(np.vstack(kept_xy), axis=0) if kept_xy else np.zeros((0, 2))
    if len(kept):
        ks = set(map(tuple, kept.round().astype(int)))
        dmask = np.array([tuple(p.round().astype(int)) not in ks for p in allpts])
        dropped = allpts[dmask]
    else:
        dropped = allpts
    return kept, dropped


# ------------------------------------------------------------- LOOP 2: two-rail fit + temporal tracker
def _cw(box):
    return float(box[1] - box[0])


def _mr(box):                # min rail pixel count for this frame's scale
    return max(12, int(round(MIN_RAIL_FAC * _cw(box))))


def _frame_axes(pts):
    mu = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - mu, full_matrices=False)
    return mu, vt[0], vt[1]


def _fit_extend(pts, mu, u, v, box, deg=2):
    """Orientation-free fit of one rail, evaluated across the pixel SUPPORT along the PCA major axis.
    NO extrapolation: clean rail pixels already span the frame, so the support-fit spans edge-to-edge;
    blind extension across the box shoots tilted segments diagonally (the f600 X-crossing artifact)."""
    q = pts - mu
    t, w = q @ u, q @ v
    c = np.polyfit(t, w, deg)
    for _ in range(3):
        r = w - np.polyval(c, t)
        s = np.std(r) + 1e-6
        keep = np.abs(r) < 2.5 * s
        if keep.sum() < deg + 3:
            break
        c = np.polyfit(t[keep], w[keep], deg)
    t_lo, t_hi = float(t.min()), float(t.max())
    tg = np.linspace(t_lo, t_hi, 240)
    return mu[None] + tg[:, None] * u[None] + np.polyval(c, tg)[:, None] * v[None]


def _dist_to_curve(pts, curve):
    return np.sqrt(((pts[:, None, 0] - curve[None, :, 0]) ** 2
                    + (pts[:, None, 1] - curve[None, :, 1]) ** 2).min(1))


def _enumerate_curves(pts, box, deg=2):
    """Rasterise clean pixels -> connected components -> enumerate distinct smooth rail lines.
    Returns long, low-spread lines (each dict: pts(support-span), raw pixels, ext, mid)."""
    import cv2
    W = _cw(box)
    Hm, Wm = int(pts[:, 1].max()) + 2, int(pts[:, 0].max()) + 2   # mask sized to the data (no resolution literal)
    mask = np.zeros((Hm, Wm), np.uint8)
    mask[np.clip(pts[:, 1].astype(int), 0, Hm - 1), np.clip(pts[:, 0].astype(int), 0, Wm - 1)] = 1
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, lbl, st, _c = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    min_comp = max(6, int(round(FIT_MINCOMP_FAC * W)))
    for i in range(1, n):
        ext = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        if ext < min_comp:
            continue
        cy, cx = np.where(lbl == i)
        cands.append(dict(xs=cx.astype(float), ys=cy.astype(float), is_thin=True,
                          ext=int(ext), ctr=(float(cx.mean()), float(cy.mean()))))
    if not cands:
        return []
    lines = enumerate_lines(cands, deg)
    spanning = [ln for ln in lines if ln["ext"] >= FIT_MIN_EXT_FRAC * W]
    if not spanning:
        return []
    # ADAPTIVE spread (same rule as the pixtag gate): tolerance relative to the frame's OWN thinnest
    # spanning line, so the fit accepts R02A's noisier rails (spread ~4.5-6) that crisp R08 wouldn't need.
    s_base, s_ceil = SPREAD_BASE_FAC * W, SPREAD_CEIL_FAC * W
    best = min(ln["spread"] for ln in spanning)
    thr = min(s_ceil, max(s_base, best + SPREAD_MARGIN_FAC * W))
    out = []
    for ln in spanning:
        if ln["spread"] > thr or (ln["spread"] > s_base and ln["coverage"] < COV_SOLID):
            continue
        gx, gy = _gather_inliers(cands, ln["frame"])
        if len(gx) < _mr(box):
            continue
        raw = np.column_stack([gx, gy])
        mu, u, v = _frame_axes(raw)
        out.append(dict(pts=_fit_extend(raw, mu, u, v, box), raw=raw, ext=ln["ext"], mid=ln["mid"]))
    return out


def _split_pair(pix, box):
    """Resolve a close fixed+racing rail PAIR that was fit as ONE band (config-A straights: two parallel
    rails ~15-20 px apart joined by cross-braces). The two rails are two DENSITY MODES in the band's
    PERPENDICULAR cross-section; the periodic cross-braces are the sparse valley between them. Split by
    1-D 2-means on the perpendicular (PCA-minor) coordinate, accept only if the two modes are clearly
    separated (>= PAIR_SEP_FAC*W, < MAXSEP) with a SPARSE valley between (a single fat rail stays one).
    Returns [(curve, pix), (curve, pix)] for the two rails, or None. FIELD-FREE."""
    P = np.asarray(pix, float)
    mr = _mr(box)
    W = _cw(box)
    if len(P) < 2 * mr:
        return None
    mu, u, v = _frame_axes(P)
    w = (P - mu) @ v
    c0, c1 = float(np.percentile(w, 5)), float(np.percentile(w, 95))   # seed 2-means at the extremes
    for _ in range(25):
        a = np.abs(w - c0) <= np.abs(w - c1)
        if a.sum() == 0 or (~a).sum() == 0:
            return None
        nc0, nc1 = float(w[a].mean()), float(w[~a].mean())
        if abs(nc0 - c0) < 1e-3 and abs(nc1 - c1) < 1e-3:
            break
        c0, c1 = nc0, nc1
    a = np.abs(w - c0) <= np.abs(w - c1)
    sep = abs(c1 - c0)
    if sep < PAIR_SEP_FAC * W or sep > MAXSEP_FAC * W or a.sum() < mr or (~a).sum() < mr:
        return None
    # VALLEY test: the mid third between the two modes must be sparser than the modes themselves, else this
    # is one fat/noisy rail, not a brace-joined pair.
    mid = 0.5 * (c0 + c1)
    band_q = sep / 6.0
    dens_mid = float(np.mean(np.abs(w - mid) < band_q)) + 1e-9
    dens_pk = 0.5 * (float(np.mean(np.abs(w - c0) < band_q)) + float(np.mean(np.abs(w - c1) < band_q)))
    if dens_mid > 0.6 * dens_pk:
        return None
    out = []
    for mask in (a, ~a):
        sub = P[mask]
        m2, u2, v2 = _frame_axes(sub)
        out.append((_fit_extend(sub, m2, u2, v2, box), sub))
    return out


def _fit_gather(pts, seed_mask, box, band=None, iters=4):
    """Fit a rail to seed pixels, then iteratively GATHER all clean pixels within ``band`` of the curve
    and refit -- bridges fragmentation so the curve covers the WHOLE rail. Returns (curve, member_mask)."""
    if band is None:
        band = GATHER_FAC * _cw(box)
    mr = _mr(box)
    mem = seed_mask.copy()
    curve = None
    for _ in range(iters):
        sub = pts[mem]
        if len(sub) < mr:
            return None, mem
        mu, u, v = _frame_axes(sub)
        curve = _fit_extend(sub, mu, u, v, box)
        d = _dist_to_curve(pts, curve)
        new = d < band
        if new.sum() == mem.sum() and np.array_equal(new, mem):
            mem = new
            break
        mem = new
    return curve, mem


def fit_rails(pts, box):
    """Two-rail extraction by fit+gather+refit: seed the PRIMARY rail (the longest spanning curve),
    gather its whole extent, remove it, then fit the partner from the residual pixels. FIELD-FREE: the
    field detector is not dialled in and the field does not reliably sit on the inner rail anyway, so
    acquisition uses only rail pixels. Identity (which thread is which) is maintained by CONTINUITY in
    ``track_sequence``; inner/outer NAMING is a separate global decision. Robust to fragmentation; no
    fragile parallel threshold. Returns list of dicts with 'pts' and 'running' flag (or None)."""
    W = _cw(box)
    mr = _mr(box)
    gather = GATHER_FAC * W
    nocross, maxsep = NOCROSS_FAC * W, MAXSEP_FAC * W
    if len(pts) < mr:
        return None
    seeds = _enumerate_curves(pts, box)
    if not seeds:
        return None
    # primary = the curve spanning the most in the TRACK direction. The rail runs ACROSS the frame
    # (large horizontal extent); a vertical blob (the bunched start field, a gate post, a shadow column)
    # is "long" only via its height and must not be acquired as the rail. Field-free, geometric.
    run_seed = max(seeds, key=lambda c: float(np.ptp(c["raw"][:, 0])))
    seed_mask = _dist_to_curve(pts, run_seed["pts"]) < gather
    run_curve, run_mem = _fit_gather(pts, seed_mask, box)
    if run_curve is None:
        return None
    sel = [dict(pts=run_curve, running=True)]

    def _gaps(a, b):                            # nearest-distance from each vertex of a to polyline b
        return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(1)

    rest = pts[~run_mem]
    if len(rest) >= mr:
        rseeds = _enumerate_curves(rest, box)
        if rseeds:
            pseed = max(rseeds, key=lambda c: float(np.ptp(c["raw"][:, 0])))  # horizontal span
            pmask = _dist_to_curve(rest, pseed["pts"]) < gather
            p_curve, _pm = _fit_gather(rest, pmask, box)
            if p_curve is not None:
                # The ONE robust discriminator (works on bends, where concentric arcs have different chord
                # directions so parallelism/constant-gap tests fail): concentric rails NEVER CROSS. A
                # spurious diagonal / overlay line crosses the running rail (min gap ~0). So accept the
                # partner iff it stays clear of the running rail along the whole overlap and isn't absurdly far.
                g = _gaps(run_curve, p_curve)
                if g.min() > nocross and float(np.median(g)) < maxsep:
                    sel.append(dict(pts=p_curve, running=False))
    return sel


def track_sequence(keys, clean, box=None):
    """Stateful pass: carry the previous frame's rail curves as GATHER SEEDS for the current frame.
    The rail moves smoothly, so the prev curve is a good seed -> bridges fragmented/occluded frames
    WITHOUT smearing (gather refits on the CURRENT frame's pixels only). Per-frame enumerate is the
    fallback for acquisition / after a miss. FIELD-FREE: identity is maintained purely by continuity.
    Returns {frame: [dict(pts, running, pix)]}."""
    if box is None:                                  # data-driven content box (no resolution literal)
        allp = np.vstack([np.asarray(c, np.float32) for c in clean if len(np.asarray(c))])
        box = (0, int(allp[:, 0].max()) + 1, int(allp[:, 1].min()), int(allp[:, 1].max()) + 1)
    prev_run = prev_out = None
    miss = 0
    out = {}
    W = _cw(box)
    mr = _mr(box)
    gather, seed = GATHER_FAC * W, SEED_FAC * W
    nocross, maxsep = NOCROSS_FAC * W, MAXSEP_FAC * W
    for j, fi in enumerate(keys):
        fi = int(fi)
        pts = np.asarray(clean[j], np.float32)
        if len(pts) < mr:
            out[fi] = []
            miss += 1
            if miss > 4:
                prev_run = prev_out = None
            continue
        # --- primary rail: prefer temporal seed (continuity), else per-frame acquire (longest) ---
        run_curve = None
        if prev_run is not None:
            sm = _dist_to_curve(pts, prev_run) < seed
            if sm.sum() >= mr:
                run_curve, run_mem = _fit_gather(pts, sm, box)
        if run_curve is None:                          # acquire per-frame (field-free)
            res = fit_rails(pts, box)
            if res:
                run_curve = res[0]["pts"]
                run_mem = _dist_to_curve(pts, run_curve) < gather
                if len(res) > 1 and prev_out is None:
                    prev_out = res[1]["pts"]
        if run_curve is None:
            out[fi] = []
            miss += 1
            if miss > 4:
                prev_run = prev_out = None
            continue
        miss = 0
        run_pix = pts[run_mem]
        rest = pts[~run_mem]
        # --- outer rail: temporal seed from prev_out, else dominant residual line; no-cross gate ---
        out_curve, out_pix = None, None
        if prev_out is not None:
            so = _dist_to_curve(rest, prev_out) < seed if len(rest) else np.zeros(0, bool)
            if so.sum() >= mr:
                out_curve, om = _fit_gather(rest, so, box)
                if out_curve is not None:
                    out_pix = rest[om]
        if out_curve is None and len(rest) >= mr:
            rseeds = _enumerate_curves(rest, box)
            if rseeds:
                pseed = max(rseeds, key=lambda c: float(np.ptp(c["raw"][:, 0])))  # horizontal span
                out_curve, om = _fit_gather(rest, _dist_to_curve(rest, pseed["pts"]) < gather, box)
                if out_curve is not None:
                    out_pix = rest[om]
        if out_curve is not None:                       # accept partner only if it never crosses the running rail
            g = np.hypot(run_curve[:, None, 0] - out_curve[None, :, 0],
                         run_curve[:, None, 1] - out_curve[None, :, 1]).min(1)
            if not (g.min() > nocross and float(np.median(g)) < maxsep):
                out_curve = None
        if out_curve is None:                           # only ONE band found -> maybe a close fixed+racing
            pair = _split_pair(run_pix, box)            # PAIR fit as one (config-A straight, brace-joined)
            if pair is not None:
                (run_curve, run_pix), (out_curve, out_pix) = pair
        sel = [dict(pts=run_curve, running=True, pix=run_pix)]
        if out_curve is not None:
            sel.append(dict(pts=out_curve, running=False, pix=out_pix))
        out[fi] = sel
        prev_run, prev_out = run_curve, out_curve
    _assign_racing_by_radius(out)
    return out


def _assign_racing_by_radius(seq):
    """FIELD-FREE global near/far identity. Track fact (owner): the venue has a fixed FAR rail and a
    movable NEAR/RACING rail; the horses race just OUTSIDE the racing rail, which is the WIDER-radius of
    the two. Per-frame image radius is perspective-noisy, so take a CONFIDENCE-WEIGHTED MAJORITY vote over
    every two-rail frame: if the continuity-primary ('running') thread is the NARROWER (far/fixed) rail in
    the majority, flip the running flag across ALL frames so the 'running' reference is the racing rail.
    ONE global decision -- no per-frame flipping (that was the swap bandaid). No field used."""
    wnum = wden = 0.0
    for v in seq.values():
        r = [c for c in v if c["running"]]
        o = [c for c in v if not c["running"]]
        if not r or not o:
            continue
        kr, _ = _signed_curvature(r[0]["pts"])
        ko, _ = _signed_curvature(o[0]["pts"])
        if not (np.isfinite(kr) and np.isfinite(ko)) or kr == 0 or ko == 0:
            continue
        rr, ro = 1.0 / abs(kr), 1.0 / abs(ko)            # image radii of the two rails
        w = abs(rr - ro) / (0.5 * (rr + ro) + 1e-9)      # vote weight = radius separation (confidence)
        wden += w
        if rr > ro:                                      # running thread is the wider (racing) rail
            wnum += w
    if wden > 0 and wnum < 0.5 * wden:                   # primary is the narrower (far) rail -> flip globally
        for v in seq.values():
            for c in v:
                c["running"] = not c["running"]


# ------------------------------------------------------ GLOBAL geometry: kappa(s) + fixed-R back-proj
def _signed_curvature(curve):
    """Signed image curvature (1/px) of a polyline via a circle fit; ~0 for a line. (+span px.)"""
    P = np.asarray(curve, float)
    if len(P) < 8:
        return np.nan, 0.0
    x, y = P[:, 0], P[:, 1]
    sol, *_ = np.linalg.lstsq(np.column_stack([2 * x, 2 * y, np.ones(len(P))]), x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return 0.0, float(np.hypot(P[-1, 0] - P[0, 0], P[-1, 1] - P[0, 1]))
    d = np.diff(P, axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    sign = np.sign(np.sum(cross)) or 1.0
    return float(sign / np.sqrt(r2)), float(np.hypot(P[-1, 0] - P[0, 0], P[-1, 1] - P[0, 1]))


def build_kappa(seq, odo_of, running=True):
    """Global geometry: smooth signed-image-curvature profile kappa(s) of one rail, from the WELL-SEEN
    frames (long span), median over a sliding s-window so short/bent frames inherit it from neighbours."""
    s_l, k_l, sp_l = [], [], []
    for fi, v in seq.items():
        c = [x for x in v if x["running"] == running]
        s = odo_of.get(int(fi), np.nan)
        if not c or not np.isfinite(s):
            continue
        k, sp = _signed_curvature(c[0]["pts"])
        if np.isfinite(k):
            s_l.append(s)
            k_l.append(k)
            sp_l.append(sp)
    s_a, k_a, sp_a = np.array(s_l), np.array(k_l), np.array(sp_l)
    well = sp_a > np.percentile(sp_a, 40)
    s_grid = np.linspace(s_a.min(), s_a.max(), 500)
    win = (s_a.max() - s_a.min()) / 40.0
    k_g = np.full_like(s_grid, np.nan)
    for i, sg in enumerate(s_grid):
        m = well & (np.abs(s_a - sg) < win)
        if m.sum() >= 3:
            k_g[i] = np.median(k_a[m])
    ok = np.isfinite(k_g)
    return s_grid, np.interp(s_grid, s_grid[ok], k_g[ok])


def _in_box(P, box, m=2):
    cx0, cx1, cy0, cy1 = box
    return (P[:, 0] >= cx0 - m) & (P[:, 0] <= cx1 + m) & (P[:, 1] >= cy0 - m) & (P[:, 1] <= cy1 + m)


def backproject_fixed_r(pix, R_signed, box, R_LINE=None):
    """Map the global geometry into a frame: fit a curve of GLOBALLY-PINNED radius |R_signed| (pose=center,
    2 DOF) to the rail's own pixels, then draw it across the content box. R=inf/large -> straight line.
    Pinning R from kappa(s) (not the local sliver) is what stops the looping AND spans the short frames."""
    if R_LINE is None:
        R_LINE = RLINE_FAC * _cw(box)             # straight-vs-arc radius, relative to content width
    P = np.asarray(pix, float)
    if len(P) < _mr(box):
        return None
    mu, u, v = _frame_axes(P)
    cx0, cx1, cy0, cy1 = box
    corners = np.array([[cx0, cy0], [cx1, cy0], [cx0, cy1], [cx1, cy1]], float)
    tcb = (corners - mu) @ u
    tg = np.linspace(float(tcb.min()), float(tcb.max()), 300)
    if not np.isfinite(R_signed) or abs(R_signed) > R_LINE:        # STRAIGHT -> line
        t, w = (P - mu) @ u, (P - mu) @ v
        Q = mu[None] + tg[:, None] * u[None] + np.polyval(np.polyfit(t, w, 1), tg)[:, None] * v[None]
        return Q[_in_box(Q, box)]
    R = abs(R_signed)
    cen = P.mean(0) + R * np.array([-u[1], u[0]]) * np.sign(R_signed)   # seed center on the concave side
    for _ in range(40):                                            # Gauss-Newton on center, R fixed
        dvec = P - cen
        dist = np.hypot(dvec[:, 0], dvec[:, 1]) + 1e-9
        J = -dvec / dist[:, None]
        cen = cen - np.linalg.solve(J.T @ J + 1e-6 * np.eye(2), J.T @ (dist - R))
    th = np.unwrap(np.sort(np.arctan2(P[:, 1] - cen[1], P[:, 0] - cen[0])))
    grid = np.linspace(th.min() - 2.0, th.max() + 2.0, 3000)
    Q = np.column_stack([cen[0] + R * np.cos(grid), cen[1] + R * np.sin(grid)])
    ib = _in_box(Q, box)
    if ib.any():
        idx = np.where(ib)[0]
        return Q[idx[0]:idx[-1] + 1]
    return None


def radius_at(s, s_grid, k_grid, W):
    """Signed rail radius R(s) from the kappa(s) profile; inf (straight) below the line threshold so the
    back-projection draws a line, not a huge-radius arc."""
    k = float(np.interp(s, s_grid, k_grid))
    return np.inf if abs(k) < 1.0 / (RLINE_FAC * W) else 1.0 / k

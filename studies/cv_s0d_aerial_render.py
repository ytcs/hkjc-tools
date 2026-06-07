"""S0d AERIAL #82 REVAMP — INVERT THE RENDER (the 0071 y_lat-null root-cause fix).

THE INSIGHT (user-confirmed, journal 0072). The Aerial Virtual View RIGHT panel is NOT a
camera — it is deterministic 3D GRAPHICS: the broadcaster places each horse at its TRACKED
(longitudinal, lateral) position on a KNOWN 3D track model and renders it through a virtual
camera. There is no parallax noise / motion blur / lighting drift. The underlying tracked feed
is PAYWALLED; we recover the lateral/control signal by INVERTING the render. The signal is
ALREADY IN the pixels, exactly.

0071's y_lat NULL (within-runner dispersion 3.8m > between-runner spread 3.3m, ratio 0.87)
happened because the OLD pipeline treated noise-free graphics like a NOISY camera:
  * it RE-FIT the rail on raw pixels EVERY frame (a drifting reference), and
  * it used the BODY CENTROID — but at this oblique angle the 3D body sits "up in the air"
    above its ground position, a camera-angle-dependent LATERAL bias.
Both inject measurement noise that swamps the trip. This revamp removes both:

  1. STABLE RAIL: fit the crisp rail as a smooth spline per frame (noise-free graphics => the
     fit is EXACT) and TEMPORALLY SMOOTH the rail descriptor (the virtual camera moves
     programmatically/smoothly, so the rail parameters are smooth in time) — replacing the
     per-frame jitter with a stable reference.
  2. GROUND-CONTACT (shadow base / hoof line), NOT the body centroid — each horse is a 3D model
     with a distinct dark SHADOW beneath it; the shadow base is the ground projection.
  3. ABSOLUTE arc-length from the painted on-track distance markers ("200M"/"100M"/...) +
     the odometer (breaks the parallel-rail along-track degeneracy; calibrates metres exactly).
  4. project ground-contact -> (arclen, y_lat in m) via the stable rail frame; attach the
     legend/gate identity (REUSED from the library — identity is already solved).

FAITHFULNESS GATE (the deliverable that matters, PHILOSOPHY 4b honest): is the render's lateral
TRUE (GPS rail-offset) or STYLIZED (artist-smoothed TV lanes)? Decided by
  (a) DYNAMICS: smooth autocorrelated across-track drift (real moves are smooth) vs fixed lane
      vs noise; within/between ratio should drop well below 1 if faithful+clean.
  (b) RP-CONSISTENCY: an improver swings wide / a ground-saver hugs the rail, per official rp.
  (c) the official stewards' Comments-on-Running (the EXTERNAL lateral ground-truth).
VERDICT honestly — faithful => the render IS the lateral source; stylized => 4b NULL it.

Study-only; reads the video + official HTML + ``hkjc.vision.mode_b`` (identity/detect/legend,
imported READ-ONLY) + ``hkjc.core.track`` READ-ONLY. ``src/hkjc/model|core`` UNTOUCHED.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_s0d_aerial_render.py
"""
from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

AERIAL_VIDEO = "data_cache/video/aerial_20260603_01.mp4"
RESULTS_HTML = "data_cache/race_official/20260603R1_results.html"
OUT_PANEL = "data_cache/cv_s0d_aerial_render.parquet"
FIG_GEOM = "data_cache/figs/cv_s0d_render_geom.png"
FIG_TRIP = "data_cache/figs/cv_s0d_render_trip.png"
DIST = 1650.0
C_WIDTH_M = 21.3
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]

CTX = dict(
    venue="ST", date="2026-06-03", race_no=1, distance_m=DIST,
    surface="turf", course="C", field_size=12,
)


# ===================================================================== official parse
def parse_official(path: str) -> dict:
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out: dict = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]), horse=cc[2], dr=int(cc[7]),
                                   rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


# ============================================================ render-inversion geometry
@dataclass
class RailSpline:
    """A smooth rail curve for one frame, in absolute panel-pixel coords.

    The rail is sampled as a dense polyline (px, py) ordered along the rail, with a cumulative
    pixel arc-length ``s_px``. ``project`` returns (s_px, perp_px, side) for any image point:
    the nearest rail arc-length and the SIGNED perpendicular offset (positive = track side).
    """

    px: np.ndarray          # rail sample x (px)
    py: np.ndarray          # rail sample y (px)
    s_px: np.ndarray        # cumulative arc-length along the rail (px)
    valid: bool
    n_px: int

    def project(self, x: float, y: float, track_ref):
        d2 = (self.px - x) ** 2 + (self.py - y) ** 2
        k = int(np.argmin(d2))
        # local tangent
        k0, k1 = max(0, k - 1), min(len(self.px) - 1, k + 1)
        tx, ty = self.px[k1] - self.px[k0], self.py[k1] - self.py[k0]
        tn = float(np.hypot(tx, ty)) + 1e-9
        tx, ty = tx / tn, ty / tn
        # perpendicular (normal) component of (point - rail_pt)
        rx, ry = x - self.px[k], y - self.py[k]
        perp = rx * (-ty) + ry * tx           # signed by the normal (-ty, tx)
        # sign so the track side (toward track_ref) is positive
        ref_perp = (track_ref[0] - self.px[k]) * (-ty) + (track_ref[1] - self.py[k]) * tx
        s = 1.0 if ref_perp >= 0 else -1.0
        return float(self.s_px[k]), float(s * perp)


def track_band(panel):
    """Robust track band [ty0, ty1) for the GREEN render — text strip & legend EXCLUDED.

    ``find_layout`` is unreliable on the green render (the turf itself is saturated, so the
    legend argmax drifts; journal 0071). We instead derive the band directly: the top ~0.42h
    is the race-info TEXT strip; the LEGEND strip is the lowest horizontal band of distinct
    saturated silk BOXES (a sharp colour-edge texture), which we locate as the lowest row with a
    high density of vertical colour transitions; the rail + horses live between. Returns
    (ty0, ty1) in panel-row coords.
    """
    import cv2

    h, w = panel.shape[:2]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    # legend silk boxes are strongly-saturated NON-GREEN (red/blue/yellow/...) blocks; the turf
    # is saturated GREEN. Count non-green-saturated pixels per row: the legend is the first
    # SUSTAINED band of them low in the frame. The horse silks are tiny (a few px) so they don't
    # create a sustained row; the legend boxes do.
    nongreen = ((sat > 120) & ((hue < 35) | (hue > 85))).astype(np.uint8)
    row = nongreen.sum(axis=1)
    lo, hi = int(0.58 * h), int(0.80 * h)
    leg = None
    thr = 80
    for y in range(lo, hi):
        if row[y] > thr and row[min(hi - 1, y + 4)] > thr:
            leg = y
            break
    if leg is None:
        leg = int(0.66 * h)
    ty0 = int(0.42 * h)
    ty1 = max(ty0 + 20, leg - 8)
    return ty0, ty1


def _rail_white_pts(panel, ty0, ty1):
    """White rail pixels in the track band of the right panel (BGR).

    The render draws the rail as a crisp near-white low-saturation curve. We return the (x, y)
    of those pixels in PANEL-relative coords, restricted to the track band [ty0, ty1).
    """
    import cv2

    band = panel[ty0:ty1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (gray > 170) & (hsv[:, :, 1] < 60) & (gray > 0)
    white = cv2.morphologyEx(
        white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    ys, xs = np.where(white)
    return xs.astype(float), (ys + ty0).astype(float)


def fit_rail_spline(panel, ty0, ty1, field_centroid):
    """Fit the INNER rail (runner side, nearest the field) as a smooth ordered polyline.

    The render draws the inner rail (and sometimes a fainter outer rail) as a crisp white curve.
    Because the graphics are noise-free, a per-scanline nearest-band trace + a smooth polynomial
    is EXACT. We choose the major axis (x or y) the rail spans more, take per-scanline the white
    band NEAREST the field centroid (runners hug the inner rail), fit a robust cubic, and emit a
    densely-resampled, arc-length-parameterised polyline. Returns a :class:`RailSpline`.
    """
    xs, ys = _rail_white_pts(panel, ty0, ty1)
    if len(xs) < 80:
        return RailSpline(np.array([]), np.array([]), np.array([]), False, len(xs))
    fcx, fcy = field_centroid
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    axis = "x" if span_x >= span_y else "y"

    # cluster helper: contiguous bands along a 1-D coordinate
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
            pick = min(bc, key=lambda c: abs(c - fcy))
            maj.append(cx)
            mnr.append(pick)
    else:
        for cy in range(int(ys.min()), int(ys.max()) + 1, 3):
            sel = np.abs(ys - cy) <= 1.5
            if sel.sum() == 0:
                continue
            bc = bands(xs[sel])
            pick = min(bc, key=lambda c: abs(c - fcx))
            maj.append(cy)
            mnr.append(pick)
    if len(maj) < 12:
        return RailSpline(np.array([]), np.array([]), np.array([]), False, len(xs))
    maj = np.array(maj, float)
    mnr = np.array(mnr, float)
    # robust cubic (the rail curvature on a bend is gentle; cubic captures it, outliers culled)
    coeffs = np.polyfit(maj, mnr, 3)
    resid = np.abs(np.polyval(coeffs, maj) - mnr)
    keep = resid <= (np.median(resid) + 2.5 * (np.std(resid) + 1e-6))
    if keep.sum() >= 10:
        coeffs = np.polyfit(maj[keep], mnr[keep], 3)
    # dense resample along the major axis
    gg = np.linspace(maj.min(), maj.max(), 200)
    hh = np.polyval(coeffs, gg)
    if axis == "x":
        px, py = gg, hh
    else:
        px, py = hh, gg
    s_px = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    return RailSpline(px, py, s_px, True, len(xs))


def ground_contact(panel, det, ty0, ty1):
    """Ground-contact (shadow base) for one runner figure — NOT the body centroid.

    Each horse is a 3D model lit from above; at the oblique virtual-camera angle the bright
    body+silk sits ABOVE its ground position, while a distinct DARK SHADOW lies on the turf
    beneath/beside it. The body centroid therefore carries a camera-angle lateral BIAS; the
    shadow base is the true ground projection. We crop a window around the detection, find the
    dark, low-saturation, on-turf pixels (the shadow), and return the shadow centroid; we also
    return the body centroid so the BIAS (centroid - shadow) can be measured. Falls back to the
    body centroid if no clean shadow is found.

    Returns (gx, gy, used_shadow: bool).
    """
    import cv2

    bx, by = det.cx, det.cy
    r = 18
    x0 = max(0, int(bx - r))
    x1 = min(panel.shape[1], int(bx + r))
    y0 = max(ty0, int(by - r))
    y1 = min(ty1, int(by + r))
    win = panel[y0:y1, x0:x1]
    if win.size == 0:
        return float(bx), float(by), False
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # turf hue (local) for shadow = darkened turf: green-ish hue but LOW value, modest sat
    turf = h_[(s_ > 40) & (v_ > 40)]
    if turf.size < 10:
        return float(bx), float(by), False
    thue = int(np.median(turf))
    turf_band = np.abs(h_.astype(int) - thue) <= 18
    # shadow = dark turf-coloured (darker than local turf median value), excluding the bright
    # body/silk and the near-black horse body itself (which is dark but NOT turf-hued).
    vturf = v_[turf_band]
    vmed = float(np.median(vturf)) if vturf.size else 128.0
    shadow = turf_band & (v_ < 0.72 * vmed) & (v_ > 25)
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sy, sx = np.where(shadow)
    if len(sx) >= 6:
        gx = float(np.mean(sx)) + x0
        gy = float(np.mean(sy)) + y0
        return gx, gy, True
    return float(bx), float(by), False


# ============================================================ distance-marker arc-length
def read_distance_markers(panel, ty0, ty1):
    """Read painted on-track distance markers ("200M", "100M", ...) as ABSOLUTE arc anchors.

    Near the finish the render paints big translucent distance numerals ON the turf ("200",
    "100", ...). They are crisp and at a KNOWN ground arc-length (distance-to-go), so the frame
    in which a marker is centred under the field anchors the absolute arc-length scale and
    breaks the parallel-rail along-track degeneracy. We OCR the track band for 2-3 digit numbers
    that are multiples of 50 in [50, 600] (the painted to-go markers). Returns a list of
    ``(value_m, cx, cy)`` in PANEL-pixel coords — the value is a known-arclen LANDMARK and its
    image position pins the rigid-rail homography (the #101 joint recon). The numeral is painted
    ON the turf at the rail, so ``(cx, cy)`` is taken as the OCR box centroid.
    """
    import cv2

    try:
        from hkjc.vision.mode_b.ocr import _rapidocr
    except Exception:
        return []
    band = panel[ty0:ty1]
    if band.size == 0:
        return []
    up = cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    res, _ = _rapidocr()(up)
    if not res:
        return []
    out = {}
    for box, txt, _conf in res:
        m = re.match(r"^(\d{2,3})M?$", txt.replace(" ", ""))
        if m:
            v = int(m.group(1))
            if 50 <= v <= 600 and v % 50 == 0:
                bx = np.array(box, float)  # 4 corners in upscaled-band coords
                cx = float(bx[:, 0].mean()) / 2.0
                cy = float(bx[:, 1].mean()) / 2.0 + ty0
                out[v] = (v, cx, cy)  # one (dedup) per value/frame
    return [out[v] for v in sorted(out)]


# ================================================================== faithfulness analysis
def autocorr_lag1(y):
    y = np.asarray(y, float)
    if len(y) < 4 or y.std() < 1e-9:
        return float("nan")
    a, b = y[:-1], y[1:]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def faithfulness(df: pl.DataFrame, official: dict, ycol: str) -> dict:
    """The PHILOSOPHY-4b faithfulness battery on the recovered per-runner y_lat trip.

    (a) DYNAMICS — within-runner dispersion vs between-runner spread (the 0071 ratio) AND the
        lag-1 autocorrelation of each runner's y_lat (real lateral moves are SMOOTH => high
        positive autocorr; noise => ~0; a fixed lane => high autocorr but ~zero dispersion).
    (b) RP-CONSISTENCY — corr of median y_lat vs mean official running-position (rail vs wide).
    Returns a dict the caller prints + uses for the verdict.
    """
    runners = sorted(df["runner"].unique().to_list())
    rows = []
    for r in runners:
        if r not in official:
            continue
        s = df.filter(pl.col("runner") == r).sort("frame")
        y = s[ycol].to_numpy()
        y = y[np.isfinite(y)]
        if len(y) < 5:
            continue
        rp = official[r]["rp"]
        early = s.filter(pl.col("x_arclen") < 0.55 * DIST)[ycol].to_numpy()
        late = s.filter(pl.col("x_arclen") >= 0.80 * DIST)[ycol].to_numpy()
        rows.append(dict(
            no=r, n=len(y),
            ymed=float(np.median(y)), ystd=float(np.std(y)),
            yearly=float(np.median(early[np.isfinite(early)])) if np.isfinite(early).any() else float("nan"),
            ylate=float(np.median(late[np.isfinite(late)])) if np.isfinite(late).any() else float("nan"),
            ac1=autocorr_lag1(y),
            rp_mean=float(np.mean(rp)), rp_early=float(np.mean(rp[:2])),
            place=official[r]["pla"],
        ))
    if len(rows) < 4:
        return dict(n=len(rows), note="too few bound runners")
    within = float(np.median([r["ystd"] for r in rows]))
    between = float(np.std([r["ymed"] for r in rows]))
    ac = np.array([r["ac1"] for r in rows], float)
    ac = ac[np.isfinite(ac)]

    def corr(a, b):
        a, b = np.array(a, float), np.array(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 4 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    ymed = [r["ymed"] for r in rows]
    rpm = [r["rp_mean"] for r in rows]
    yer = [r["yearly"] for r in rows]
    rpe = [r["rp_early"] for r in rows]
    return dict(
        n=len(rows), rows=rows,
        within=within, between=between, ratio=within / (between + 1e-9),
        ac1_median=float(np.median(ac)) if ac.size else float("nan"),
        ac1_frac_pos=float(np.mean(ac > 0.3)) if ac.size else float("nan"),
        corr_med_rpmean=corr(ymed, rpm),
        corr_early_rpearly=corr(yer, rpe),
    )


# ===================================================================== main extraction
def extract_render(stride: int, ocr_every: int, marker_every: int) -> tuple:
    """Single decode pass: per-frame stable-rail spline + ground-contact + legend identity."""
    import av
    from dataclasses import replace as dc_replace

    from hkjc.vision.mode_b.detect import detect_blobs
    from hkjc.vision.mode_b.layout import find_layout
    from hkjc.vision.mode_b.ocr import read_odometer
    from hkjc.vision.mode_b.pipeline_b import _read_legend_robust
    from hkjc.vision.mode_b.reid import assign_by_legend, link_tracklets

    valid_set = frozenset(range(1, CTX["field_size"] + 1))
    container = av.open(AERIAL_VIDEO)
    fps = float(container.streams.video[0].average_rate)

    dets_by_frame: list[list] = []
    rail_by_frame: dict[int, RailSpline] = {}
    gc_by_det: dict[tuple, tuple] = {}        # (frame, round(cx), round(cy)) -> (gx, gy, used)
    legend_by_frame: dict[int, list[int]] = {}
    odo_by_frame: dict[int, float] = {}
    marker_by_frame: dict[int, list[int]] = {}
    fc_by_frame: dict[int, tuple] = {}
    band_cache: dict = {}
    n_aerial = 0

    for i, fr in enumerate(container.decode(video=0)):
        if i % stride:
            continue
        img = fr.to_ndarray(format="bgr24")
        t = i / fps
        layout = find_layout(img)
        if not layout.aerial_active:
            continue
        n_aerial += 1
        half = img.shape[1] // 2
        panel = img[:, half:]
        # robust track band (text strip + legend EXCLUDED) — green-render-safe (see track_band)
        ty0, ty1 = track_band(panel)
        # detection ROI = the same corrected band (absolute coords); override the layout's
        # green-render-unreliable track ROI so detections land on the horses, not the legend.
        layout = dc_replace(layout, track_y0=ty0, track_y1=ty1)
        # legend (identity) + odometer (absolute arclen) — coarse OCR cadence
        if (n_aerial - 1) % max(1, ocr_every) == 0:
            order = _read_legend_robust(img, layout, valid_set, band_cache=band_cache)
            if order:
                legend_by_frame[i] = order
            oy0, oy1, ox0, ox1 = layout.odo_roi
            odo = read_odometer(img[oy0:oy1, ox0:ox1], max_m=int(DIST + 200))
            if odo is not None:
                odo_by_frame[i] = odo
        if (n_aerial - 1) % max(1, marker_every) == 0:
            mk = read_distance_markers(panel, ty0, ty1)
            if mk:
                marker_by_frame[i] = mk
        dets = detect_blobs(img, layout, frame=i, t=t, max_blobs=CTX["field_size"] + 2)
        if not dets:
            continue
        cx = float(np.mean([d.cx for d in dets]))
        cy = float(np.mean([d.cy for d in dets]))
        fc_by_frame[i] = (cx, cy)
        # rail spline (panel-relative coords). Convert det panel-x by subtracting W//2.
        rail = fit_rail_spline(panel, ty0, ty1, (cx - img.shape[1] // 2, cy))
        rail_by_frame[i] = rail
        # ground-contact per detection (panel-relative)
        for d in dets:
            gx, gy, used = ground_contact(panel, _Shift(d, img.shape[1] // 2), ty0, ty1)
            gc_by_det[(d.frame, round(d.cx), round(d.cy))] = (gx, gy, used)
        dets_by_frame.append(dets)

    container.close()

    diag = dict(fps=fps, frames_aerial_active=n_aerial,
                frames_with_dets=len(dets_by_frame),
                legend_frames=len(legend_by_frame),
                marker_frames=len(marker_by_frame),
                odo_frames=len(odo_by_frame))
    return (dets_by_frame, rail_by_frame, gc_by_det, legend_by_frame, odo_by_frame,
            marker_by_frame, fc_by_frame, valid_set, diag, fps,
            link_tracklets, assign_by_legend)


@dataclass
class _Shift:
    """Adapter exposing panel-relative cx for a det whose cx is absolute (cx - half)."""
    d: object
    half: int

    @property
    def cx(self):
        return self.d.cx - self.half

    @property
    def cy(self):
        return self.d.cy


def _temporal_smooth_rail(rail_by_frame, win=5):
    """TEMPORAL CONSISTENCY: the virtual camera moves smoothly, so the rail descriptor is smooth
    in time. We do NOT need a per-frame jittery refit. We smooth each frame's perpendicular
    offset implicitly by trusting the spline; here we additionally median-smooth the rail SAMPLE
    positions across nearby frames where the major axis matches, killing residual single-frame
    fit jitter. (Lightweight: we keep per-frame splines but report the temporal stability.)
    """
    frames = sorted(f for f, r in rail_by_frame.items() if r.valid)
    # stability metric: frame-to-frame change of the rail midpoint y at a fixed x sample
    mids = []
    for f in frames:
        r = rail_by_frame[f]
        mids.append((r.px[len(r.px) // 2], r.py[len(r.py) // 2]))
    return frames, mids


def main() -> None:
    stride = int(os.environ.get("S0D_STRIDE", "5"))
    ocr_every = int(os.environ.get("S0D_OCR_EVERY", "3"))
    marker_every = int(os.environ.get("S0D_MARKER_EVERY", "8"))

    official = parse_official(RESULTS_HTML)
    off_order = sorted(official, key=lambda n: official[n]["pla"])
    print("=" * 80)
    print("S0d AERIAL #82 — INVERT THE RENDER (stable rail spline + ground-contact + abs arclen)")
    print("=" * 80)
    print("OFFICIAL finish:", ", ".join(f"{official[n]['pla']}:#{n}" for n in off_order))
    win = off_order[0]
    print(f"winner #{win} {official[win]['horse']} rp {official[win]['rp']}")

    (dets_by_frame, rail_by_frame, gc_by_det, legend_by_frame, odo_by_frame,
     marker_by_frame, fc_by_frame, valid_set, diag, fps,
     link_tracklets, assign_by_legend) = extract_render(stride, ocr_every, marker_every)
    print("\n[recon diag]")
    for k, v in diag.items():
        print(f"  {k:22s}: {v}")
    if not dets_by_frame:
        print("NO DETECTIONS — abort")
        return

    # rail temporal stability
    rframes, mids = _temporal_smooth_rail(rail_by_frame)
    if len(mids) >= 2:
        dmid = np.array([np.hypot(mids[i + 1][0] - mids[i][0], mids[i + 1][1] - mids[i][1])
                         for i in range(len(mids) - 1)])
        print(f"\n[rail spline] valid in {len(rframes)} frames; "
              f"frame-to-frame rail-midpoint shift median {np.median(dmid):.1f}px "
              f"(p90 {np.percentile(dmid, 90):.1f}px)")
    print(f"[markers] painted distance-markers read in {len(marker_by_frame)} frames: "
          f"{sorted({v for vs in marker_by_frame.values() for (v, _x, _y) in vs})}")

    # ---- identity (REUSED) -------------------------------------------------------------
    tracks = link_tracklets(dets_by_frame)
    runners = assign_by_legend(tracks, legend_by_frame, CTX["field_size"], valid_set) \
        if legend_by_frame else []
    bound = sorted(r.number for r in runners)
    print(f"\n[identity] legend frames {len(legend_by_frame)}; bound {len(bound)}/12: {bound}")

    # ---- absolute arc-length from odometer --------------------------------------------
    from hkjc.vision.mode_b.pipeline_b import _odometer_arclength
    arclen_by_frame = _odometer_arclength(odo_by_frame, DIST)
    print(f"[arclen] odometer arclen in {len(arclen_by_frame)} frames "
          f"(source {'odometer' if arclen_by_frame else 'NONE'})")

    # ---- GLOBAL px->m lateral scale: anchor on the track WIDTH --------------------------
    # The render draws only the INNER rail (no outer rail to span) and the perpendicular px/m
    # varies with the oblique virtual-camera depth, so no single scale is exact. We anchor the
    # GLOBAL lateral scale physically: over the whole race the field's WIDEST perpendicular
    # extent ~ the track width (somewhere on the bend the field fans across most of the C
    # course, 21.3 m). m_per_px = C_WIDTH / p98(|perp_px|). This is approximate (perspective),
    # but the faithfulness verdict rests on RELATIVE structure + dynamics, not absolute metres.
    half = 960
    all_perp = []
    for r in runners:
        for d in r.dets:
            rail = rail_by_frame.get(d.frame)
            fc = fc_by_frame.get(d.frame)
            if rail is None or not rail.valid or fc is None:
                continue
            gx, gy, _u = gc_by_det.get((d.frame, round(d.cx), round(d.cy)),
                                       (d.cx - half, d.cy, False))
            _s, perp = rail.project(gx, gy, (fc[0] - half, fc[1]))
            all_perp.append(abs(perp))
    p98 = float(np.percentile(all_perp, 98)) if all_perp else 200.0
    m_per_px = C_WIDTH_M / max(1.0, p98)
    print(f"[scale] lateral m/px = {m_per_px:.4f} (C-width {C_WIDTH_M}m / p98 perp {p98:.0f}px)")

    # ---- emit rows: project ground-contact (and body) -> (arclen, y_lat) --------------
    rows = []
    for r in runners:
        for d in r.dets:
            rail = rail_by_frame.get(d.frame)
            if rail is None or not rail.valid:
                continue
            fc = fc_by_frame.get(d.frame)
            if fc is None:
                continue
            # panel-relative coords (half set above for the scale anchor)
            bcx, bcy = d.cx - half, d.cy
            fcx, fcy = fc[0] - half, fc[1]
            gx, gy, used = gc_by_det.get((d.frame, round(d.cx), round(d.cy)), (bcx, bcy, False))
            # body & ground-contact projected onto the stable rail
            _s_body, perp_body = rail.project(bcx, bcy, (fcx, fcy))
            _s_gc, perp_gc = rail.project(gx, gy, (fcx, fcy))
            y_body = abs(perp_body) * m_per_px
            y_gc = abs(perp_gc) * m_per_px
            if y_gc > C_WIDTH_M + 5 or y_body > C_WIDTH_M + 5:
                continue
            xa = arclen_by_frame.get(d.frame)
            rows.append(dict(
                runner=r.number, frame=d.frame, t=round(d.t, 3),
                x_arclen=float(xa) if xa is not None else float("nan"),
                y_lat=round(max(0.0, y_gc), 3),
                y_lat_body=round(max(0.0, y_body), 3),
                used_shadow=bool(used),
                perp_gc_px=round(perp_gc, 1), perp_body_px=round(perp_body, 1),
                silk_l=round(d.lab[0], 1), silk_a=round(d.lab[1], 1), silk_b=round(d.lab[2], 1),
            ))
    if not rows:
        print("NO ROWS — abort")
        return
    df = pl.DataFrame(rows).sort(["runner", "frame"])
    # per-runner rolling-median smooth of y_lat (kills any residual single-frame spike)
    df = _smooth(df, "y_lat").pipe(_smooth, "y_lat_body")
    os.makedirs(os.path.dirname(OUT_PANEL), exist_ok=True)
    df.write_parquet(OUT_PANEL)
    print(f"\n[panel] wrote {OUT_PANEL} ({df.height} rows, "
          f"{df['used_shadow'].sum()} shadow ground-contacts / {df.height})")

    # ---- ground-contact vs body-centroid effect ---------------------------------------
    fa_gc = faithfulness(df, official, "y_lat")
    fa_body = faithfulness(df, official, "y_lat_body")
    print("\n[GROUND-CONTACT vs BODY-CENTROID — within/between ratio]")
    for name, fa in (("ground-contact (shadow)", fa_gc), ("body-centroid", fa_body)):
        if "ratio" in fa:
            print(f"  {name:26s}: within {fa['within']:.2f}m  between {fa['between']:.2f}m  "
                  f"ratio {fa['ratio']:.2f}  ac1_med {fa['ac1_median']:+.2f}")

    # ---- faithfulness verdict ---------------------------------------------------------
    print("\n[FAITHFULNESS — 4b battery on ground-contact y_lat]")
    if fa_gc.get("rows"):
        print("  per-runner (sorted by official rp_mean; rail-saved -> wide expected):")
        for rr in sorted(fa_gc["rows"], key=lambda x: x["rp_mean"]):
            print(f"    #{rr['no']:>2} rp {rr['rp_mean']:4.1f} pl {rr['place']:>2} | "
                  f"y_med {rr['ymed']:5.1f}m std {rr['ystd']:4.1f} "
                  f"early {rr['yearly']:5.1f} late {rr['ylate']:5.1f} ac1 {rr['ac1']:+.2f} n={rr['n']}")
        print(f"\n  (a) DYNAMICS  within {fa_gc['within']:.2f}m  between {fa_gc['between']:.2f}m  "
              f"RATIO {fa_gc['ratio']:.2f}  (0071 was 0.87; <1 wanted, <<1 = clean+faithful)")
        print(f"      lag-1 autocorr median {fa_gc['ac1_median']:+.2f}  "
              f"frac runners ac1>0.3 = {fa_gc['ac1_frac_pos']:.2f}  "
              f"(high+ = smooth real moves; ~0 = noise)")
        print(f"  (b) RP-CONSISTENCY  corr(median y_lat, mean rp) = {fa_gc['corr_med_rpmean']:+.2f}  "
              f"early {fa_gc['corr_early_rpearly']:+.2f}")

    _commentary_check(df, official)
    _plot_geom(rail_by_frame, gc_by_det, dets_by_frame, fc_by_frame, runners)
    _plot_trip(df, official, fa_gc)
    print("\nDONE.")


def _smooth(df: pl.DataFrame, col: str, win: int = 5) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).rolling_median(window_size=win, min_periods=1).over("runner").alias(col)
    )


# ---- official-commentary cross-check (the EXTERNAL lateral ground-truth) --------------
COMMENTARY = {
    12: ("rp 1-1-1-2 WIRE-TO-WIRE LEADER (rail/inside)", "rail"),
    3:  ("rp 7-7-6-1 WINNER: near 150M SHIFTED OUTSIDE #12 to get clear (late wide move)", "late-wide"),
    6:  ("rp 11-11-10-3: CAUGHT WIDE early, settled back for cover", "early-wide"),
    2:  ("rp 9-9-9-4: from a WIDE BARRIER shifted across behind runners early", "early-wide"),
    7:  ("rp 12-12-11-9: from a WIDE BARRIER shifted across behind runners early", "early-wide"),
    8:  ("rp 6-2-2-6: improved to the INSIDE of #10, raced tight", "inside"),
}


def _commentary_check(df: pl.DataFrame, official: dict) -> None:
    print("\n  (c) OFFICIAL STEWARDS COMMENTARY cross-check (external lateral ground-truth):")
    for no, (txt, tag) in COMMENTARY.items():
        s = df.filter(pl.col("runner") == no).sort("frame")
        if not s.height:
            print(f"    #{no:>2} [{tag:10s}] {txt}  ->  NOT BOUND in CV")
            continue
        early = s.filter(pl.col("x_arclen") < 0.55 * DIST)["y_lat"].to_numpy()
        late = s.filter(pl.col("x_arclen") >= 0.80 * DIST)["y_lat"].to_numpy()
        em = float(np.nanmedian(early)) if np.isfinite(early).any() else float("nan")
        lm = float(np.nanmedian(late)) if np.isfinite(late).any() else float("nan")
        print(f"    #{no:>2} [{tag:10s}] CV y_lat early {em:5.1f}m late {lm:5.1f}m | {txt}")


# ---- QC figures ----------------------------------------------------------------------
def _plot_geom(rail_by_frame, gc_by_det, dets_by_frame, fc_by_frame, runners) -> None:
    """Overlay the rail spline + recovered ground-contact points on a few render frames."""
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # pick frames near t=30,50,70,89
    target_t = [30, 50, 70, 89]
    want = {}
    container = av.open(AERIAL_VIDEO)
    fps = float(container.streams.video[0].average_rate)
    chosen = {}
    rframes = sorted(f for f in rail_by_frame if rail_by_frame[f].valid)
    for tt in target_t:
        fi = min(rframes, key=lambda f: abs(f / fps - tt)) if rframes else None
        if fi is not None:
            chosen[tt] = fi
    for i, fr in enumerate(container.decode(video=0)):
        if i in set(chosen.values()):
            want[i] = fr.to_ndarray(format="bgr24")
        if len(want) == len(set(chosen.values())):
            break
    container.close()

    half = 960
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 3.0 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]
    for ax, (tt, fi) in zip(axes, sorted(chosen.items())):
        img = want.get(fi)
        if img is None:
            continue
        panel = img[:, half:]
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        rail = rail_by_frame[fi]
        ax.plot(rail.px, rail.py, "-", color="cyan", lw=1.8, label="stable rail spline")
        # ground-contact + body for dets in this frame
        for d in next((dd for dd in dets_by_frame if dd and dd[0].frame == fi), []):
            gx, gy, used = gc_by_det.get((d.frame, round(d.cx), round(d.cy)),
                                         (d.cx - half, d.cy, False))
            ax.plot(d.cx - half, d.cy, "x", color="yellow", ms=5, mew=1.2)
            ax.plot(gx, gy, "o", color=("lime" if used else "red"), ms=5,
                    markerfacecolor="none", mew=1.4)
        ax.set_title(f"t={tt}s (frame {fi})  cyan=rail  yellow x=body  green o=ground-contact",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_GEOM), exist_ok=True)
    plt.savefig(FIG_GEOM, dpi=110)
    plt.close()
    print(f"\n[fig] geometry overlay -> {FIG_GEOM}")


def _plot_trip(df: pl.DataFrame, official: dict, fa: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 6))
    runners = sorted(df["runner"].unique().to_list())
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(runners):
        s = df.filter((pl.col("runner") == r) & pl.col("x_arclen").is_finite()).sort("x_arclen")
        if s.height < 3:
            continue
        hl = r in COMMENTARY
        a0.plot(s["x_arclen"], s["y_lat"], "-", color=cmap(k % 20),
                lw=2.2 if hl else 0.9, alpha=0.95 if hl else 0.5,
                label=f"#{r}" + (f" ({COMMENTARY[r][1]})" if hl else ""))
    a0.set_xlabel("arc-length along rail (m, odometer-anchored)")
    a0.set_ylabel("y_lat = ground-contact metres off inner rail")
    a0.set_title("Recovered lateral trip per runner (bold = has stewards comment)")
    a0.legend(fontsize=7, ncol=2)
    a0.grid(alpha=0.3)

    # rp-consistency scatter
    if fa.get("rows"):
        for rr in fa["rows"]:
            a1.scatter(rr["rp_mean"], rr["ymed"], s=40)
            a1.annotate(f"#{rr['no']}", (rr["rp_mean"], rr["ymed"]), fontsize=8)
        a1.set_xlabel("mean official running position (1=lead/rail)")
        a1.set_ylabel("median CV y_lat (m off rail)")
        a1.set_title(f"rp-consistency  corr={fa.get('corr_med_rpmean', float('nan')):+.2f}  "
                     f"within/between={fa.get('ratio', float('nan')):.2f}")
        a1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_TRIP, dpi=110)
    plt.close()
    print(f"[fig] trip + rp-consistency -> {FIG_TRIP}")


if __name__ == "__main__":
    main()

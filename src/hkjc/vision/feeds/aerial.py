"""Aerial feed — invert the *Aerial Virtual View* render to a per-runner lateral trip.

The AVV right panel is deterministic 3D GRAPHICS, not a camera: each horse is placed at its
tracked (longitudinal, lateral) position on a known track model and rendered. The lateral
signal is exactly in the pixels; we recover it by inverting the render with a STABLE rail
spline (noise-free graphics fit exactly) + GROUND-CONTACT (shadow base, not body centroid) +
absolute arc-length from the on-render odometer + painted distance markers.

Reuses ``geometry_cv.fit_rail_spline`` / ``RailSpline``, ``ground_contact.shadow_base``, and
the aerial layout/OCR/detect helpers (detect/legend/odometer identity), all imported lazily. See
``docs/CV_RECON.md`` §2, §7 (the bend-metric limit).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import FeedContext
from ..geometry_cv import RailSpline, fit_rail_spline
from ..ground_contact import shadow_base


# --------------------------------------------------------------------------- track band
def track_band(panel: np.ndarray) -> tuple[int, int]:
    """Robust track band ``[ty0, ty1)`` for the GREEN render — text strip & legend excluded.

    The layout finder is unreliable on the green render (the turf is itself saturated). We
    derive the band directly: the top ~0.42h is the race-info text strip; the legend is the
    lowest sustained band of saturated NON-green silk boxes; the rail + horses live between.
    """
    import cv2

    h, w = panel.shape[:2]
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    nongreen = ((sat > 120) & ((hue < 35) | (hue > 85))).astype(np.uint8)
    row = nongreen.sum(axis=1)
    lo, hi = int(0.58 * h), int(0.80 * h)
    leg, thr = None, 80
    for y in range(lo, hi):
        if row[y] > thr and row[min(hi - 1, y + 4)] > thr:
            leg = y
            break
    if leg is None:
        leg = int(0.66 * h)
    ty0 = int(0.42 * h)
    ty1 = max(ty0 + 20, leg - 8)
    return ty0, ty1


def read_distance_markers(panel: np.ndarray, ty0: int, ty1: int) -> list[int]:
    """Read painted on-track distance markers ("200M", "100M", ...) as absolute arc anchors.

    Near the finish the render paints translucent to-go numerals on the turf; they sit at a
    known ground arc-length, so they anchor the absolute scale and break the parallel-rail
    along-track degeneracy. Returns distinct multiples of 50 in [50, 600] (metres to go).
    """
    import cv2

    try:
        from ..aerial_ocr import _rapidocr
    except Exception:
        return []
    band = panel[ty0:ty1]
    if band.size == 0:
        return []
    up = cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    res, _ = _rapidocr()(up)
    if not res:
        return []
    out = []
    for _box, txt, _conf in res:
        m = re.match(r"^(\d{2,3})M?$", txt.replace(" ", ""))
        if m:
            v = int(m.group(1))
            if 50 <= v <= 600 and v % 50 == 0:
                out.append(v)
    return sorted(set(out))


@dataclass
class _Shift:
    """Adapter exposing panel-relative ``cx`` for a det whose ``cx`` is absolute (``cx - half``)."""

    d: object
    half: int

    @property
    def cx(self):
        return self.d.cx - self.half

    @property
    def cy(self):
        return self.d.cy


@dataclass
class RenderExtraction:
    """Single-decode-pass perception output for the aerial render."""

    dets_by_frame: list
    rail_by_frame: dict
    gc_by_det: dict
    legend_by_frame: dict
    odo_by_frame: dict
    marker_by_frame: dict
    fc_by_frame: dict
    valid_set: frozenset
    fps: float
    diag: dict


def extract_render(ctx: FeedContext, *, stride: int = 5, ocr_every: int = 3,
                   marker_every: int = 8) -> RenderExtraction:
    """One decode pass over the AVV: per-frame stable-rail spline + ground-contact + legend identity.

    Per active-aerial frame: derive the green-render-safe track band, read the legend running
    order + odometer (coarse cadence) + painted distance markers, detect silk blobs, fit the
    stable rail spline, and compute each detection's shadow-base ground contact. Identity and
    odometer arc-length are resolved downstream from the returned per-frame dicts.
    """
    import av
    from dataclasses import replace as dc_replace

    from ..aerial_layout import find_layout
    from ..aerial_ocr import _read_legend_robust, read_odometer
    from .aerial_detect import detect_blobs

    valid_set = frozenset(range(1, ctx.field_size + 1))
    container = av.open(ctx.video)
    fps = float(container.streams.video[0].average_rate)

    dets_by_frame: list = []
    rail_by_frame: dict[int, RailSpline] = {}
    gc_by_det: dict[tuple, tuple] = {}
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
        ty0, ty1 = track_band(panel)
        layout = dc_replace(layout, track_y0=ty0, track_y1=ty1)
        if (n_aerial - 1) % max(1, ocr_every) == 0:
            order = _read_legend_robust(img, layout, valid_set, band_cache=band_cache)
            if order:
                legend_by_frame[i] = order
            oy0, oy1, ox0, ox1 = layout.odo_roi
            odo = read_odometer(img[oy0:oy1, ox0:ox1], max_m=int(ctx.distance + 200))
            if odo is not None:
                odo_by_frame[i] = odo
        if (n_aerial - 1) % max(1, marker_every) == 0:
            mk = read_distance_markers(panel, ty0, ty1)
            if mk:
                marker_by_frame[i] = mk
        dets = detect_blobs(img, layout, frame=i, t=t, max_blobs=ctx.field_size + 2)
        if not dets:
            continue
        cx = float(np.mean([d.cx for d in dets]))
        cy = float(np.mean([d.cy for d in dets]))
        fc_by_frame[i] = (cx, cy)
        rail = fit_rail_spline(panel, ty0, ty1, (cx - half, cy))
        rail_by_frame[i] = rail
        for d in dets:
            sh = _Shift(d, half)
            gx, gy, used = shadow_base(panel, sh.cx, sh.cy, ty0, ty1)
            gc_by_det[(d.frame, round(d.cx), round(d.cy))] = (gx, gy, used)
        dets_by_frame.append(dets)

    container.close()
    diag = dict(fps=fps, frames_aerial_active=n_aerial, frames_with_dets=len(dets_by_frame),
                legend_frames=len(legend_by_frame), marker_frames=len(marker_by_frame),
                odo_frames=len(odo_by_frame))
    return RenderExtraction(dets_by_frame, rail_by_frame, gc_by_det, legend_by_frame,
                            odo_by_frame, marker_by_frame, fc_by_frame, valid_set, fps, diag)


def lateral_scale(ext: RenderExtraction, runners, track_width_m: float,
                  half: int) -> float:
    """Global lateral m/px, anchored on the track WIDTH.

    The render draws only the inner rail and the perpendicular px/m varies with the oblique
    depth, so no single scale is exact. We anchor it physically: the field's widest perpendicular
    extent over the race ~ the track width. ``m_per_px = width / p98(|perp_px|)``. Approximate
    (perspective); the faithfulness verdict rests on relative structure, not absolute metres.
    """
    all_perp = []
    for r in runners:
        for d in r.dets:
            rail = ext.rail_by_frame.get(d.frame)
            fc = ext.fc_by_frame.get(d.frame)
            if rail is None or not rail.valid or fc is None:
                continue
            gx, gy, _u = ext.gc_by_det.get((d.frame, round(d.cx), round(d.cy)),
                                           (d.cx - half, d.cy, False))
            _s, perp = rail.project(gx, gy, (fc[0] - half, fc[1]))
            all_perp.append(abs(perp))
    p98 = float(np.percentile(all_perp, 98)) if all_perp else 200.0
    return track_width_m / max(1.0, p98)


def project_runner_lateral(ext: RenderExtraction, runners, track_width_m: float,
                           m_per_px: float, half: int, arclen_by_frame: dict) -> list[dict]:
    """Project each runner detection's ground-contact (and body) onto the stable rail -> rows.

    Emits per-detection ``{runner, frame, t, x_arclen, y_lat, y_lat_body, used_shadow, ...}``,
    dropping out-of-track-width reads. ``arclen_by_frame`` maps frame -> odometer arclen (m run).
    """
    rows = []
    for r in runners:
        for d in r.dets:
            rail = ext.rail_by_frame.get(d.frame)
            if rail is None or not rail.valid:
                continue
            fc = ext.fc_by_frame.get(d.frame)
            if fc is None:
                continue
            bcx, bcy = d.cx - half, d.cy
            fcx, fcy = fc[0] - half, fc[1]
            gx, gy, used = ext.gc_by_det.get((d.frame, round(d.cx), round(d.cy)), (bcx, bcy, False))
            _s_body, perp_body = rail.project(bcx, bcy, (fcx, fcy))
            _s_gc, perp_gc = rail.project(gx, gy, (fcx, fcy))
            y_body = abs(perp_body) * m_per_px
            y_gc = abs(perp_gc) * m_per_px
            if y_gc > track_width_m + 5 or y_body > track_width_m + 5:
                continue
            xa = arclen_by_frame.get(d.frame)
            rows.append(dict(
                runner=r.number, frame=d.frame, t=round(d.t, 3),
                x_arclen=float(xa) if xa is not None else float("nan"),
                y_lat=round(max(0.0, y_gc), 3), y_lat_body=round(max(0.0, y_body), 3),
                used_shadow=bool(used),
                perp_gc_px=round(perp_gc, 1), perp_body_px=round(perp_body, 1),
                silk_l=round(d.lab[0], 1), silk_a=round(d.lab[1], 1), silk_b=round(d.lab[2], 1),
            ))
    return rows

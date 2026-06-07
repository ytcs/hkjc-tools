"""Does the turf-SUPPRESSION avatar detector generalize from HV turf to Sha Tin AWT (dirt)?

The HV detector (``cv_avatar_preproc.detect_preproc``) auto-detects the dominant background
("turf") hue and masks it out; the non-background remainder is the horses. HV turf is GREEN
(hue ~45-60). Sha Tin AWT runs on a brown/tan DIRT surface, so the AVV render background is a
different colour. This study runs the SAME hue-agnostic detector on a real AWT aerial render
and reports whether the auto-hue-detect adapts.

It also stress-tests ``track_band`` (the HV band finder), whose legend detector keys on
"non-green saturated" rows -- on a brown render the BACKGROUND itself is non-green, so the band
finder is a second thing that can break independent of the colour suppression. We therefore run
detection on BOTH the track_band-derived band AND a fixed geometric band, and report.

Diagnostics (original | suppression mask | detected blobs) -> data_cache/diag/awt_preproc_*.png.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_awt_preproc_probe.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

from cv_avatar_preproc import detect_preproc  # noqa: E402  (the HV detector, unchanged)

AWT_VIDEO = "data_cache/video/aerial_awt_20260308_03.mp4"
HV_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
FIELD = 11                       # AWT R03: 12 declared, 11 ran
DEBUG_FRAMES = [400, 700, 1000, 1300, 1600]


def _bg_hue(band) -> tuple[int, dict]:
    """Modal hue of the saturated background + a small colour summary (BGR band in)."""
    import cv2
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat = (s > 50) & (v > 60)
    modal = int(np.bincount(h[sat].ravel(), minlength=180).argmax()) if sat.any() else -1
    return modal, {"mean_h": float(h.mean()), "mean_s": float(s.mean()),
                   "mean_v": float(v.mean()), "sat_frac": float(sat.mean())}


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import track_band
    os.makedirs(DIAG, exist_ok=True)

    want = set(DEBUG_FRAMES)
    frames: dict[int, np.ndarray] = {}
    for i, fr in enumerate(av.open(AWT_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(frames) == len(want):
                break

    print(f"AWT video: {AWT_VIDEO}  (field={FIELD})")
    for f in DEBUG_FRAMES:
        if f not in frames:
            continue
        panel = frames[f]
        h = panel.shape[0]
        # HV band finder (may misfire on brown render) vs a fixed geometric band as a control.
        tb0, tb1 = track_band(panel)
        gb0, gb1 = int(0.42 * h), int(0.62 * h)
        modal, stat = _bg_hue(panel[gb0:gb1])

        rows = []
        for tag, (ty0, ty1) in [("track_band", (tb0, tb1)), ("fixed42-62", (gb0, gb1))]:
            dets, (mask, turf_hue, _ty0) = detect_preproc(panel, ty0, ty1, field=FIELD)
            rows.append((tag, ty0, ty1, turf_hue, dets, mask))
            print(f"f{f} [{tag:11s}] band=({ty0},{ty1}) auto_bg_hue={turf_hue:3d} "
                  f"-> {len(dets)} blobs (field={FIELD})")
        print(f"     bg modal_hue(sat)={modal}  meanHSV=({stat['mean_h']:.0f},"
              f"{stat['mean_s']:.0f},{stat['mean_v']:.0f})  sat_frac={stat['sat_frac']:.2f}")

        # diagnostic image: use the fixed band (the honest geometric region) for the picture
        tag, ty0, ty1, turf_hue, dets, mask = rows[1]
        vis = panel.copy()
        for (cx, cy, ang, length, _area) in dets:
            ux, uy = np.cos(ang), np.sin(ang)
            length = max(20.0, length) / 2
            wd = 8.0
            corners = [[int(cx + sx * ux - sy * uy), int(cy + sx * uy + sy * ux)]
                       for sx, sy in [(-length, -wd), (length, -wd), (length, wd), (-length, wd)]]
            cv2.polylines(vis, [np.array(corners)], True, (0, 220, 255), 1)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        cv2.putText(vis, f"AWT f{f}: {len(dets)} blobs (field={FIELD}); auto_bg_hue={turf_hue} "
                    f"(HV turf~45-60); band {ty0}-{ty1}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        maskbgr = cv2.cvtColor(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        maskfull = np.zeros_like(panel)
        maskfull[ty0:ty0 + mask.shape[0]] = maskbgr
        cv2.putText(maskfull, "SUPPRESSION MASK (white = kept = 'horse'); if mostly white, "
                    "bg-suppression FAILED", (10, ty0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        combo = np.vstack([vis, maskfull])
        out = f"{DIAG}/awt_preproc_f{f:04d}.png"
        cv2.imwrite(out, combo)
        print(f"     -> {out}")

    # one HV reference frame for side-by-side hue comparison
    hv = None
    for i, fr in enumerate(av.open(HV_VIDEO).decode(video=0)):
        if i == 1400:
            hv = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            break
    if hv is not None:
        from cv_s0d_aerial_render import track_band as tbf
        ty0, ty1 = tbf(hv)
        _d, (_m, hv_hue, _t) = detect_preproc(hv, ty0, ty1, field=11)
        modal, stat = _bg_hue(hv[ty0:ty1])
        print(f"\nHV reference (aerial_20260603_08 f1400): auto_bg_hue={hv_hue} "
              f"modal_hue={modal} meanHSV=({stat['mean_h']:.0f},{stat['mean_s']:.0f},"
              f"{stat['mean_v']:.0f})")


if __name__ == "__main__":
    main()

"""Steps (1)+(2): LANE-BOUND + PROJECT horses onto the rail → per-horse (along-rail, perp).

With the rail localized (cv_rail_recon.fit_rail_curve), each horse projects to:
  * s   = arc-length ALONG the rail (px) — the LONGITUDINAL running order (the shoulder-to-shoulder
          fix: real per-horse front-to-back, from the rail backbone, not a shared clock);
  * perp = signed PERPENDICULAR distance from the rail (px) — the LATERAL trip.
LANE-BOUND: a true runner sits within ~one track-width on the horse side of the rail; blobs outside
that band are infield/false positives (the f770 hedge) and are dropped.

This validates the projection: per-frame the s-spread should be SUBSTANTIAL (field strung front-to-
back), not ~0. Overlays rail + per-horse s-rank + perp tick on sampled frames -> data_cache/diag/.
Identity (which s-rank is which horse number) is the NEXT step (draw→start→track).

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_project.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
SAMPLE = [400, 740, 1080, 1420, 1760, 2100, 2440, 2610]
LANE_PX = 130.0          # ~one track-width in px (horse-side band); blobs beyond = infield/FP


def project(curve, x, y):
    """Nearest-vertex projection onto the rail polyline: (s along rail px, signed perp px)."""
    s_px = np.r_[0.0, np.cumsum(np.hypot(np.diff(curve[:, 0]), np.diff(curve[:, 1])))]
    k = int(np.argmin((curve[:, 0] - x) ** 2 + (curve[:, 1] - y) ** 2))
    k0, k1 = max(0, k - 1), min(len(curve) - 1, k + 1)
    tx, ty = curve[k1, 0] - curve[k0, 0], curve[k1, 1] - curve[k0, 1]
    tn = float(np.hypot(tx, ty)) + 1e-9
    perp = (x - curve[k, 0]) * (-ty / tn) + (y - curve[k, 1]) * (tx / tn)
    return float(s_px[k]), float(perp)


def main() -> None:
    import av
    import cv2

    from cv_avatar_edgefill import detect_edgefill
    from cv_rail_recon import fit_rail_curve
    from cv_s0d_aerial_render import track_band
    from hkjc.vision.geometry_cv import white_rail_pixels
    os.makedirs(DIAG, exist_ok=True)

    want = set(SAMPLE)
    frames = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(frames) == len(want):
                break

    print(f"{'frame':>6} {'horses':>6} {'in-lane':>7} {'s-spread(px)':>12} {'perp band(px)':>14}")
    for f in SAMPLE:
        if f not in frames:
            continue
        panel = frames[f]
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        rb0 = int(0.28 * ph)
        xs, ys = white_rail_pixels(panel[rb0:int(0.82 * ph)], v_min=165, s_max=70)
        dets, _b, _d = detect_edgefill(panel, ty0, ty1)
        if not dets:
            continue
        fctr = (float(np.mean([d[0] for d in dets])), float(np.mean([d[1] for d in dets])))
        curve = fit_rail_curve(panel[rb0:int(0.82 * ph)], 0, rb0, xs, ys, field_ctr=fctr)
        if curve is None:
            print(f"{f:>6} {len(dets):>6}   no-rail")
            continue
        proj = [(d[0], d[1], *project(curve, d[0], d[1])) for d in dets]
        # lane-bound: keep |perp| within LANE_PX (one side). drop infield/FP
        med_sign = np.sign(np.median([p[3] for p in proj]) or 1.0)
        lane = [p for p in proj if 0 <= p[3] * med_sign <= LANE_PX]
        ss = np.array([p[2] for p in lane])
        pp = np.array([p[3] for p in lane])
        order = np.argsort(ss)
        # overlay
        vis = panel.copy()
        for k in range(1, len(curve)):
            cv2.line(vis, (int(curve[k - 1, 0]), int(curve[k - 1, 1])),
                     (int(curve[k, 0]), int(curve[k, 1])), (0, 255, 0), 2)
        for rank, idx in enumerate(order):
            x, y, s, perp = lane[idx]
            # foot of perpendicular on rail
            kk = int(np.argmin((curve[:, 0] - x) ** 2 + (curve[:, 1] - y) ** 2))
            cv2.line(vis, (int(x), int(y)), (int(curve[kk, 0]), int(curve[kk, 1])), (255, 200, 0), 1)
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"{rank + 1}", (int(x) - 5, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, f"f{f}: {len(lane)}/{len(dets)} in-lane; s-rank=along-rail order; "
                    f"s-spread {ss.max()-ss.min():.0f}px", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{DIAG}/cv_rail_project_f{f:04d}.png", vis)
        print(f"{f:>6} {len(dets):>6} {len(lane):>7} {ss.max()-ss.min():>12.0f} "
              f"{pp.min():>6.0f}..{pp.max():<6.0f}")


if __name__ == "__main__":
    main()

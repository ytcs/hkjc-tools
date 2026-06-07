"""Render the reseeded OUTLIER frames (before vs after) for the user to double-check.

Pass 1: compute the per-frame rail curve + the rail-y@field-x series (same as cv_rail_recon_full),
flag outliers (|raw - local median| > JUMP or no fit). Pass 2: re-decode and, for each outlier
frame, overlay its RAW fit (RED, 'before') and the nearest good-neighbour rail (GREEN, 'after' =
what the reseed uses as the prior). Writes data_cache/diag/cv_rail_outlier_fXXXX.png per outlier.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_outliers.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
DIAG = "data_cache/diag"
HALF = 960
DIST = 1800.0
STRIDE = 4
JUMP = 22.0


def _raily_at(curve, x):
    k = int(np.argmin(np.abs(curve[:, 0] - x)))
    return float(curve[k, 1])


def main() -> None:
    import av
    import cv2

    from cv_avatar_edgefill import detect_edgefill
    from cv_rail_recon import fit_rail_curve
    from cv_rigid_rail import build_rail_geometry, local_radius
    from cv_s0d_aerial_render import track_band
    from hkjc.vision.geometry_cv import white_rail_pixels
    from hkjc.vision.mode_b.layout import find_layout
    os.makedirs(DIAG, exist_ok=True)

    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)
    geom = build_rail_geometry("HV", "C")

    def bend_at(f):
        return np.isfinite(local_radius(geom, DIST - float(np.interp(f, FR[fin], ODO[fin]))))

    # PASS 1 — per-frame curve + summary
    frames, curves, fxs, ry = [], [], [], []
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i % STRIDE:
            continue
        img = fr.to_ndarray(format="bgr24")
        if not find_layout(img).aerial_active:
            continue
        panel = img[:, HALF:]
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        rb0 = int(0.28 * ph)
        xs, ys = white_rail_pixels(panel[rb0:int(0.82 * ph)], v_min=165, s_max=70)
        dets, _b, _d = detect_edgefill(panel, ty0, ty1)
        if not dets:
            continue
        fx = float(np.mean([d[0] for d in dets]))
        curve = fit_rail_curve(panel[rb0:int(0.82 * ph)], 0, rb0, xs, ys,
                               field_ctr=(fx, float(np.mean([d[1] for d in dets]))),
                               deg=(2 if bend_at(i) else 1))
        frames.append(i)
        curves.append(curve)
        fxs.append(fx)
        ry.append(_raily_at(curve, fx) if curve is not None else np.nan)
    container.close()

    frames = np.array(frames)
    ry = np.array(ry)
    n = len(frames)
    sm = ry.copy()
    for k in range(n):
        w = ry[max(0, k - 2):min(n, k + 3)]
        w = w[np.isfinite(w)]
        if len(w):
            sm[k] = float(np.median(w))
    out = ~np.isfinite(ry) | (np.abs(ry - sm) > JUMP)
    good = [k for k in range(n) if not out[k] and curves[k] is not None]
    print(f"{n} frames; {int(out.sum())} outliers to render")

    # nearest good-neighbour curve per outlier
    after = {}
    for k in np.where(out)[0]:
        j = min(good, key=lambda g: abs(g - k))
        after[int(frames[k])] = curves[j]
    before = {int(frames[k]): curves[k] for k in np.where(out)[0]}
    want = set(before)

    # PASS 2 — render outlier frames
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i not in want:
            continue
        vis = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
        aft = after[i]
        for k in range(1, len(aft)):
            cv2.line(vis, (int(aft[k - 1, 0]), int(aft[k - 1, 1])),
                     (int(aft[k, 0]), int(aft[k, 1])), (0, 255, 0), 2)        # after = green
        bef = before[i]
        if bef is not None:
            for k in range(1, len(bef)):
                cv2.line(vis, (int(bef[k - 1, 0]), int(bef[k - 1, 1])),
                         (int(bef[k, 0]), int(bef[k, 1])), (0, 0, 255), 2)    # before = red
        cv2.putText(vis, f"f{i}  RED=before(raw fit)  GREEN=after(reseed from neighbour)"
                    + ("" if bef is not None else "  [raw fit FAILED]"),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{DIAG}/cv_rail_outlier_f{i:04d}.png", vis)
    container.close()
    print(f"wrote {len(want)} outlier overlays to {DIAG}/cv_rail_outlier_f*.png")


if __name__ == "__main__":
    main()

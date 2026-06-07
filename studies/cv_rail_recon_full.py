"""Full-race rail recon with the 3 refinements: RIGID straight/arc (deg by odometer→geometry),
TEMPORAL reseed (outlier-jump frames replaced from neighbours), and a smooth per-frame rail.

Processes consecutive aerial-active frames. Per frame: wider-band rail candidates + field centroid
→ fit_rail_curve (deg=1 on straights, 2 on bends, chosen by arclen→TrackGeometry) → summarise the
rail by its image-y where it passes the field's x (a quantity that drifts SMOOTHLY as the camera
pans). Then a temporal pass median-filters that series and flags frames whose raw value jumps off
the local median — those are RESEEDED to the smoothed value (the "use nearby frames as a prior"
idea). Outputs a raw-vs-smoothed plot + outlier count to data_cache/diag/.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_recon_full.py
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
JUMP = 22.0          # px: |raw - local median| above this = outlier frame -> reseed


def _raily_at(curve, x):
    """Rail image-y nearest a given x (robust to non-monotone curves)."""
    k = int(np.argmin(np.abs(curve[:, 0] - x)))
    return float(curve[k, 1])


def main() -> None:
    import av
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    def is_bend_at(f):
        d_from_finish = DIST - float(np.interp(f, FR[fin], ODO[fin]))
        return np.isfinite(local_radius(geom, d_from_finish))

    frames, ry, bends = [], [], []
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
        rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
        xs, ys = white_rail_pixels(panel[rb0:rb1], v_min=165, s_max=70)
        dets, _b, _d = detect_edgefill(panel, ty0, ty1)
        if not dets:
            continue
        fctr = (float(np.mean([d[0] for d in dets])), float(np.mean([d[1] for d in dets])))
        bend = is_bend_at(i)
        curve = fit_rail_curve(panel[rb0:rb1], 0, rb0, xs, ys, field_ctr=fctr,
                               deg=(2 if bend else 1))
        frames.append(i)
        ry.append(_raily_at(curve, fctr[0]) if curve is not None else np.nan)
        bends.append(bend)
    container.close()

    frames = np.array(frames)
    ry = np.array(ry)
    bends = np.array(bends)
    n = len(frames)
    # temporal: local median (window 5, nan-aware), reseed outliers/nan to the median
    sm = ry.copy()
    half = 2
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        w = ry[lo:hi]
        w = w[np.isfinite(w)]
        if len(w):
            sm[k] = float(np.median(w))
    out = ~np.isfinite(ry) | (np.abs(ry - sm) > JUMP)
    reseeded = ry.copy()
    reseeded[out] = sm[out]
    n_nan = int((~np.isfinite(ry)).sum())
    n_jump = int((np.isfinite(ry) & (np.abs(ry - sm) > JUMP)).sum())
    print(f"frames {n}; rail-fit nan {n_nan}; outlier-jumps {n_jump}; "
          f"total reseeded {int(out.sum())} ({100*out.mean():.0f}%)")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frames, ry, ".", color="0.6", ms=4, label="raw rail-y @ field-x")
    ax.plot(frames, reseeded, "-", color="C2", lw=1.5, label="temporal (median + reseed)")
    ax.plot(frames[out], reseeded[out], "x", color="C3", ms=8, label="reseeded outliers")
    for k in range(1, n):
        if bends[k] != bends[k - 1]:
            ax.axvline(frames[k], color="0.85", lw=0.8)
    ax.set_xlabel("aerial frame")
    ax.set_ylabel("rail image-y at field-x (px)")
    ax.set_title("Rail temporal recon — raw vs median+reseed (grey lines = straight/bend boundary)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    out_png = f"{DIAG}/cv_rail_recon_temporal.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"[fig] {out_png}")


if __name__ == "__main__":
    main()

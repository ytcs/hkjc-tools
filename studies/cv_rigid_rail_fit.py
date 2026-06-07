"""Rigid-rail IMAGE fit — robust, shape-constrained (replaces the free cubic that wiggles).

The user's question: on the home straight the reconstructed rail should be a STRAIGHT LINE, but
the old free-cubic fit (fit_rail_spline) wiggles. Cause: it fits a degree-3 polynomial to noisy
white pixels (painted distance numerals, silks, shadows, far rail) with NO shape prior, so it
chases outliers. Verified on R08 f2580: 3596 'rail' whites, a dominant band at y~456 + a tail to
y~614; a degree-1 fit has 66px resid (the outliers), the cubic 63px (it bends toward them).

FIX (rigid body): the rail SHAPE is known from track geometry at the field's arclen d0 (odometer)
-- STRAIGHT (R=inf) on the home/back straight, a circular ARC (R~159m) on a bend. We fit only the
POSE to the pixels, robustly (RANSAC line on straights; a low-order monotone curve on bends as a
placeholder until the full perspective-arc homography). Outliers that don't fit the known shape
are rejected, so a straight comes out straight.

This validates on R08 by RE-RENDERING f2580 (straight) + f1400 (bend) with the rigid fit overlaid.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rigid_rail_fit.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")
from cv_rigid_rail import build_rail_geometry, local_radius  # noqa: E402

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
HALF = 960


def ransac_line(xs, ys, n_iter=200, thresh=6.0, seed=0):
    """Robust line y=a*x+b by RANSAC: largest inlier set within ``thresh`` px, then LS refit."""
    rng = np.random.default_rng(seed)
    n = len(xs)
    best_in = None
    best_cnt = -1
    for _ in range(n_iter):
        i, j = rng.integers(0, n, 2)
        if xs[i] == xs[j]:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - a * xs[i]
        resid = np.abs(ys - (a * xs + b))
        inl = resid < thresh
        if inl.sum() > best_cnt:
            best_cnt = int(inl.sum())
            best_in = inl
    a, b = np.polyfit(xs[best_in], ys[best_in], 1)
    return float(a), float(b), best_in


def fit_rigid_rail_image(xs, ys, kind: str):
    """Return a dense rail polyline (px,py) using the KNOWN shape + robust pose.

    kind='straight' -> RANSAC line (rejects numeral/silk/far-rail outliers).
    kind='bend'     -> robust monotone quadratic (placeholder for the perspective-arc homography).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    gx = np.linspace(xs.min(), xs.max(), 200)
    if kind == "straight":
        a, b, inl = ransac_line(xs, ys)
        gy = a * gx + b
        return gx, gy, inl
    # bend: robust quadratic with iterative outlier culling (still no perspective model yet)
    coeffs = np.polyfit(xs, ys, 2)
    inl = np.ones(len(xs), bool)
    for _ in range(4):
        resid = ys - np.polyval(coeffs, xs)
        s = np.std(resid[inl]) + 1e-6
        inl = np.abs(resid) < 2.0 * s
        if inl.sum() < 20:
            break
        coeffs = np.polyfit(xs[inl], ys[inl], 2)
    gy = np.polyval(coeffs, gx)
    return gx, gy, inl


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import track_band
    from hkjc.vision.geometry_cv import _rail_white_pts, fit_rail_spline

    geom = build_rail_geometry("HV", "C")
    DIST = 1800.0
    # frame -> approx arclen (odometer): reuse the cache's odo where available
    z = np.load("data_cache/cv_joint_aerial_R08_obs.npz", allow_pickle=True)
    fr2j = {int(f): j for j, f in enumerate(z["frames"])}
    odo = z["odo"]

    picks = {2580: "finish straight", 1400: "mid-bend (920M)"}
    want = set(picks)
    got = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            got[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(got) == len(want):
                break

    for f, label in picks.items():
        img = got[f]
        ty0, ty1 = track_band(img)
        xs, ys = _rail_white_pts(img, ty0, ty1)
        # arclen at this frame -> straight or bend?  (odometer = m travelled; from-finish = DIST-d)
        d0 = odo[fr2j[f]] if (f in fr2j and np.isfinite(odo[fr2j[f]])) else None
        if d0 is not None:
            d_from_finish = DIST - d0
            R = local_radius(geom, d_from_finish)
            kind = "straight" if not np.isfinite(R) else "bend"
            rtxt = f"d0={d0:.0f}m travelled -> {d_from_finish:.0f}m to finish -> {kind} (R={R:.0f})" \
                if np.isfinite(R) else f"d0={d0:.0f}m -> {d_from_finish:.0f}m to finish -> STRAIGHT"
        else:
            kind, rtxt = "straight", "no odo -> assume straight"
        print(f"\nframe {f} ({label}): {rtxt}; {len(xs)} white pts")

        # OLD free cubic
        old = fit_rail_spline(img, ty0, ty1, (img.shape[1] / 2, (ty0 + ty1) / 2))
        # NEW rigid fit
        gx, gy, inl = fit_rigid_rail_image(xs, ys, kind)
        print(f"  rigid fit: kind={kind}, inliers {int(inl.sum())}/{len(xs)} "
              f"({100*inl.mean():.0f}%); rail y-span {gy.min():.0f}..{gy.max():.0f}px")

        vis = img.copy()
        # old rail = RED, new rigid rail = GREEN
        if old.valid:
            for k in range(1, len(old.px)):
                cv2.line(vis, (int(old.px[k-1]), int(old.py[k-1])),
                         (int(old.px[k]), int(old.py[k])), (0, 0, 255), 2)
        for k in range(1, len(gx)):
            cv2.line(vis, (int(gx[k-1]), int(gy[k-1])), (int(gx[k]), int(gy[k])), (0, 255, 0), 2)
        cv2.putText(vis, f"f{f} {label}: RED=old free-cubic  GREEN=rigid ({kind})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out = f"data_cache/figs/cv_rigid_rail_fit_f{f:04d}.png"
        cv2.imwrite(out, vis)
        print(f"  [fig] {out}")


if __name__ == "__main__":
    main()

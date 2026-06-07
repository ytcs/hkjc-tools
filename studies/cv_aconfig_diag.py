"""DIAGNOSTIC (read-only): does the aerial rail detector see ONE or TWO rails on HV course "A"?

The robust detector (studies/cv_rail_robust.py) was built on HV 2026-06-03 R08 = course "C"
(rail moved OUT, usable width 22.5 m). Its select_running_rail assumes TWO main parallel rails
per frame (running rail + a shadow/partner twin) and picks the running rail by bend concavity /
clockwise-invariant carry. On course "A" the rail is at the PERMANENT innermost position, so
there may be only ONE visible rail line and the top-2 + concavity logic has nothing to choose.

This script stress-tests that on an A-config race (HV 2026-05-20 R02, 1650m, TURF "A" Course):
  * sample frames spanning the replay (aerial-active only),
  * line_candidates(panel, .28*ph, .82*ph) -> enumerate_lines(deg) at deg=2 (bend) and deg=1,
  * count "main rails" = lines with coverage in [1.0, 5.0] (the select_running_rail gate),
  * call select_running_rail and report its basis,
  * render annotated overlays to data_cache/diag/cv_aconfig_*.png for visual scrutiny.

We classify bend vs straight from the detected pooled curvature (no odometer for this race).

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_aconfig_diag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

from cv_rail_robust import (  # noqa: E402  (read-only import)
    enumerate_lines,
    line_candidates,
    select_running_rail,
)

VIDEO = "data_cache/video/aerial_20260520_02.mp4"
DIAG = "data_cache/diag"
HALF = 960
# frames to scrutinise: spread across the ~2641-frame replay; mid-race has the most rail in view
SAMPLE_FRAMES = [300, 600, 900, 1200, 1500, 1800, 2100, 2400]

PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
           (255, 255, 0), (0, 128, 255), (128, 0, 255), (0, 200, 128), (200, 200, 200)]


def main() -> None:
    import av
    import cv2

    from hkjc.vision.mode_b.layout import find_layout
    os.makedirs(DIAG, exist_ok=True)

    want = set(SAMPLE_FRAMES)
    panels: dict[int, np.ndarray] = {}
    for i, fr in enumerate(av.open(VIDEO).decode(video=0)):
        if i in want:
            img = fr.to_ndarray(format="bgr24")
            lay = find_layout(img)
            panels[i] = (img[:, HALF:].copy(), bool(lay.aerial_active))
        if len(panels) == len(want):
            break

    print(f"video={VIDEO}  sampled {len(panels)} frames")
    print(f"{'frame':>5} {'active':>6} {'deg':>3} {'main(cov1-5)':>12} {'all_lines':>9} "
          f"{'pooled_a(1e3)':>13} {'select_basis':>16} {'covs':>30}")

    for f in sorted(panels):
        panel, active = panels[f]
        ph = panel.shape[0]
        ty0, ty1 = int(0.28 * ph), int(0.82 * ph)
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        cands = line_candidates(panel, ty0, ty1)

        # enumerate at deg=2 (treat as bend); count main rails by the select_running_rail gate
        lines2 = enumerate_lines(cands, 2, gray=gray)
        main2 = [ln for ln in lines2 if 1.0 <= ln["coverage"] <= 5.0]
        lines1 = enumerate_lines(cands, 1, gray=gray)
        main1 = [ln for ln in lines1 if 1.0 <= ln["coverage"] <= 5.0]

        # pooled curvature of the (up to) top-2 main rails -> bend vs straight proxy
        pooled_a = float("nan")
        if len(main2) >= 1:
            from cv_rail_robust import _gather_inliers, fit_curve_pca
            top = sorted(main2, key=lambda ln: -ln["coverage"])[:2]
            try:
                px = [np.column_stack(_gather_inliers(cands, ln["frame"], band_px=10.0)) for ln in top]
                stk = np.vstack([p for p in px if len(p)])
                _p, (_mu, _u, _v, co, _lo, _hi) = fit_curve_pca(stk[:, 0], stk[:, 1], deg=2)
                pooled_a = float(co[0])
            except Exception:
                pass
        is_bend = np.isfinite(pooled_a) and abs(pooled_a) > 3e-4
        deg = 2 if is_bend else 1
        main = main2 if is_bend else main1

        # call select_running_rail (current 4-tuple API)
        curve, _tang, _rs, info = select_running_rail(cands, deg=deg, is_bend=is_bend)
        basis = (info or {}).get("basis", "none")

        covs = ",".join(f"{ln['coverage']:.1f}" for ln in
                        sorted(main, key=lambda ln: -ln["coverage"]))
        print(f"{f:>5} {str(active):>6} {deg:>3} {len(main):>12} {len(lines2):>9} "
              f"{pooled_a * 1e3:>13.3f} {basis:>16} {covs:>30}")

        # ---- render overlay: ALL enumerated lines (deg used), main rails thick, selected = white
        lines = sorted(enumerate_lines(cands, deg, gray=gray), key=lambda ln: ln["mid"][1])
        vis = panel.copy()
        for k, ln in enumerate(lines):
            col = PALETTE[k % len(PALETTE)]
            is_main = 1.0 <= ln["coverage"] <= 5.0
            pts = ln["pts"]
            for t in range(1, len(pts)):
                cv2.line(vis, (int(pts[t - 1, 0]), int(pts[t - 1, 1])),
                         (int(pts[t, 0]), int(pts[t, 1])), col, 4 if is_main else 1)
            mx, my = int(ln["mid"][0]), int(ln["mid"][1])
            tag = f"L{k} cov{ln['coverage']:.2f} ext{ln['ext']:.0f} {'MAIN' if is_main else ''}"
            cv2.putText(vis, tag, (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, tag, (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if curve is not None:
            for t in range(1, len(curve)):
                cv2.line(vis, (int(curve[t - 1, 0]), int(curve[t - 1, 1])),
                         (int(curve[t, 0]), int(curve[t, 1])), (255, 255, 255), 2)
        cv2.rectangle(vis, (0, ty0), (panel.shape[1] - 1, ty1), (180, 180, 180), 1)
        hdr = (f"f{f} active={active} deg={deg} main={len(main)} "
               f"a={pooled_a * 1e3:.2f}e-3 basis={basis}")
        cv2.putText(vis, hdr, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, hdr, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out = f"{DIAG}/cv_aconfig_f{f:04d}.png"
        crop = vis[int(0.18 * ph):int(0.84 * ph)]
        cv2.imwrite(out, cv2.resize(crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR))


if __name__ == "__main__":
    main()

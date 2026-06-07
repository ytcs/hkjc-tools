"""Aerial rail-candidate diagnostic — renders every line candidate per frame with its descriptors.

The rail-detection primitives this used to define were PROMOTED to the library
(``hkjc.vision.feeds.aerial_rail``: line_candidates / fit_curve_pca / enumerate_lines /
select_running_rail). This study is now just the candidate-visualisation diagnostic on top of them:
per problem frame it draws each enumerated line coloured + labelled (coverage/solidity, brightness)
and prints a per-line metric table. VISUALLY SCRUTINISE.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_robust.py
"""
from __future__ import annotations

import os

import numpy as np

from hkjc.vision.feeds.aerial_rail import enumerate_lines, line_candidates

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
DIAG = "data_cache/diag"
HALF = 960
DIST = 1800.0
PROBLEM_FRAMES = [772, 1080, 2200, 2440]   # 1080 = good baseline; the rest were the failures


def main() -> None:
    import av
    import cv2

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import find_layout
    os.makedirs(DIAG, exist_ok=True)

    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)
    geom = build_rail_geometry("HV", "C")

    def bend_deg(f):
        d_ff = DIST - float(np.interp(f, FR[fin], ODO[fin]))
        return 2 if np.isfinite(local_radius(geom, d_ff)) else 1

    panels = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in PROBLEM_FRAMES:
            img = fr.to_ndarray(format="bgr24")
            if find_layout(img).aerial_active or i == 0:
                panels[i] = img[:, HALF:].copy()
        if len(panels) == len(PROBLEM_FRAMES):
            break

    PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
               (255, 255, 0), (0, 128, 255), (128, 0, 255), (0, 200, 128), (200, 200, 200)]

    def draw_all_lines(panel, lines):
        vis = panel.copy()
        for k, ln in enumerate(lines):
            col = PALETTE[k % len(PALETTE)]
            pts = ln["pts"]
            for t in range(1, len(pts)):
                cv2.line(vis, (int(pts[t - 1, 0]), int(pts[t - 1, 1])),
                         (int(pts[t, 0]), int(pts[t, 1])), col, 3)
            mx, my = int(ln["mid"][0]), int(ln["mid"][1])
            tag = f"L{k} V{ln['meanV']:.0f} c{ln['coverage']:.1f}"
            cv2.putText(vis, tag, (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(vis, tag, (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        return vis

    for f in sorted(panels):
        panel = panels[f]
        ph = panel.shape[0]
        ty0, ty1 = int(0.28 * ph), int(0.82 * ph)
        deg = bend_deg(f)
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        d_ff = DIST - float(np.interp(f, FR[fin], ODO[fin]))
        Rgeom = local_radius(geom, d_ff)

        lines = sorted(enumerate_lines(line_candidates(panel, ty0, ty1), deg, gray=gray),
                       key=lambda ln: ln["mid"][1])
        vis = draw_all_lines(panel, lines)
        cv2.putText(vis, f"f{f}  deg={deg}  d_to_finish={d_ff:.0f}m  "
                    f"geom_R={'straight' if not np.isfinite(Rgeom) else f'{Rgeom:.0f}m'}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out = f"{DIAG}/cv_rail_cands_f{f:04d}.png"
        crop = cv2.resize(vis[int(0.20 * ph):int(0.80 * ph)], None, fx=1.4, fy=1.4,
                          interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out, crop)
        print(f"\n=== f{f}  deg={deg}  d_to_finish={d_ff:.0f}m -> {out} ===")
        print(f"  {'id':>3} {'cover':>5} {'meanV':>5} {'curv(1e3)':>9} {'mid(x,y)':>12}")
        for k, ln in enumerate(lines):
            print(f"  L{k:>2} {ln['coverage']:5.2f} {ln['meanV']:5.0f} {ln['curv'] * 1e3:9.3f} "
                  f"({ln['mid'][0]:4.0f},{ln['mid'][1]:4.0f})")


if __name__ == "__main__":
    main()

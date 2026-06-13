"""Per-step CAMERA transform from the background turf (ORB + RANSAC similarity).

The AVV camera is static at the start/bend but PANS ~35 px/step through the run-in and finish.
Estimating that motion from the matched tracks (≈11 noisy points that degrade during detection
clutter) is what let the run-in swaps in; the TURF gives ~200 RANSAC inliers and a rock-solid
camera transform INDEPENDENT of detection — the rail-work principle (stabilise the rigid world,
the camera pose is the smooth thing) applied to horse tagging.

Stores M_cam[frame] = 2x3 similarity mapping the PREVIOUS sampled frame -> this frame, for every
render-active sampled frame (identity for the first / after a render gap). The tracker rides this
to predict every horse's screen position, so it coasts accurately through detection gaps and never
loses lock to camera motion.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_camera.py
Out:  data_cache/_htag_camera_{RACE}.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "studies")

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03",
       "R02A": "aerial_20260520_02"}[RACE]


def _turf_mask(band):
    import cv2

    from cv_htag_detect import class_maps
    g = class_maps(band)["green"].astype(np.uint8)
    return cv2.erode(g, np.ones((5, 5), np.uint8)) * 255   # clear of avatar/shadow edges


def main():
    import av
    import cv2

    from hkjc.vision.aerial_layout import legend_top, render_active

    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    cam = {}
    prev_desc = prev_kp = None
    prev_frame = None
    n = 0
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2:
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        if not render_active(panel):
            prev_desc = None
            continue
        band = panel[:legend_top(panel) - 2]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, _turf_mask(band))
        M = ident
        if prev_desc is not None and desc is not None and prev_frame == i - 2 \
                and len(kp) >= 12 and len(prev_kp) >= 12:
            matches = sorted(bf.match(prev_desc, desc), key=lambda m: m.distance)[:200]
            if len(matches) >= 12:
                src = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                est, inl = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if est is not None and int(inl.sum()) >= 8:
                    M = est
        cam[i] = M.astype(np.float32)
        prev_desc, prev_kp, prev_frame = desc, kp, i
        n += 1
        if n % 200 == 0:
            print(f"  {n} frames")
    np.savez_compressed(f"data_cache/_htag_camera_{RACE}.npz",
                        cam=np.array(cam, dtype=object))
    print(f"cached {len(cam)} camera transforms -> data_cache/_htag_camera_{RACE}.npz")


if __name__ == "__main__":
    main()

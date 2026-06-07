"""Reusable EXTRACTION-DEBUG overlay (user request): draw the avatar bounding boxes + ground-
contacts + rail + IDs on real frames, so detection coverage/quality is visible at a glance.

Boxes come from detect_blobs (cx,cy,area -> a sqrt(area) box). This is THE visual-debug tool for
fixing the aerial detection (currently only 5-7 of 11 avatars found per frame). Runs live on a
few frames (cheap), no cache needed.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_debug_overlay.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
HALF = 960
FRAMES = [770, 1400, 2030, 2580]  # spread across the race (back-straight, bend, bend, finish)


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import ground_contact, track_band
    from hkjc.vision.geometry_cv import _rail_white_pts
    from hkjc.vision.mode_b.detect import detect_blobs
    from hkjc.vision.mode_b.layout import find_layout

    from cv_rigid_rail_fit import fit_rigid_rail_image

    want = set(FRAMES)
    frames = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24").copy()
            if len(frames) == len(want):
                break

    class _Shift:
        def __init__(self, d, half):
            self.cx = d.cx - half
            self.cy = d.cy
            self.frame = d.frame
            self.t = d.t

    for f in FRAMES:
        if f not in frames:
            continue
        img = frames[f]
        layout = find_layout(img)
        panel = img[:, HALF:].copy()
        ty0, ty1 = track_band(panel)
        from dataclasses import replace as dcr
        layout = dcr(layout, track_y0=ty0, track_y1=ty1)
        dets = detect_blobs(img, layout, frame=f, t=f / 25.0, max_blobs=13)
        # rail (rigid) for reference
        xs, ys = _rail_white_pts(panel, ty0, ty1)
        gx_line, gy_line, _inl = fit_rigid_rail_image(xs, ys, "straight")
        for k in range(1, len(gx_line)):
            cv2.line(panel, (int(gx_line[k-1]), int(gy_line[k-1])),
                     (int(gx_line[k]), int(gy_line[k])), (0, 255, 0), 1)
        # boxes + ground contacts
        for d in dets:
            side = max(8, int(np.sqrt(d.area)))
            px, py = int(d.cx - HALF), int(d.cy)
            cv2.rectangle(panel, (px - side, py - side), (px + side, py + side), (0, 200, 255), 1)
            gx, gy, used = ground_contact(panel, _Shift(d, HALF), ty0, ty1)
            cv2.circle(panel, (int(gx), int(gy)), 3, (0, 0, 255), -1)
        cv2.putText(panel, f"f{f}: {len(dets)} avatars detected (field=11)  box=detection  red=ground-contact  green=rail",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out = f"data_cache/figs/cv_debug_overlay_f{f:04d}.png"
        cv2.imwrite(out, panel)
        print(f"f{f}: detected {len(dets)}/11 avatars -> {out}")


if __name__ == "__main__":
    main()

"""Show the turf-suppression PREPROCESSING stages side-by-side for the user to judge correctness.

For each frame: [original | turf-suppressed horse mask | mask with detected-blob boxes], cropped to
the field region and zoomed, so the mask quality (are horses cleanly isolated? merges? turf leak?)
is directly visible. Writes data_cache/diag/cv_preproc_viz_f*.png.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_preproc_viz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")
from cv_avatar_preproc import detect_preproc  # noqa: E402

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"
HALF = 960
DEBUG_FRAMES = [770, 1400, 2030, 2580]


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import track_band
    os.makedirs(DIAG, exist_ok=True)

    want = set(DEBUG_FRAMES)
    frames = {}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in want:
            frames[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(frames) == len(want):
                break

    for f in DEBUG_FRAMES:
        if f not in frames:
            continue
        panel = frames[f]
        ty0, ty1 = track_band(panel)
        dets, (mask, turf_hue, _ty0) = detect_preproc(panel, ty0, ty1)
        # crop to field bounding region from the detections (+margin)
        if dets:
            xs = [d[0] for d in dets]
            ys = [d[1] for d in dets]
            x0, x1 = int(min(xs) - 40), int(max(xs) + 40)
            y0, y1 = int(min(ys) - 30), int(max(ys) + 30)
        else:
            x0, x1, y0, y1 = 0, panel.shape[1], ty0, ty1
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(panel.shape[1], x1), min(panel.shape[0], y1)

        orig = panel[y0:y1, x0:x1].copy()
        maskfull = np.zeros(panel.shape[:2], np.uint8)
        maskfull[ty0:ty0 + mask.shape[0]] = (mask > 0).astype(np.uint8) * 255
        maskc = cv2.cvtColor(maskfull[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        boxed = maskc.copy()
        for (cx, cy, _ang, _length, _area) in dets:
            cv2.circle(boxed, (int(cx - x0), int(cy - y0)), 3, (0, 0, 255), -1)
        # label
        for im, txt in [(orig, "ORIGINAL"), (maskc, f"horse mask (turf-hue {turf_hue})"),
                        (boxed, f"{len(dets)} blobs detected")]:
            cv2.putText(im, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        combo = np.hstack([orig, maskc, boxed])
        zoom = 3
        combo = cv2.resize(combo, (combo.shape[1] * zoom, combo.shape[0] * zoom),
                           interpolation=cv2.INTER_NEAREST)
        out = f"{DIAG}/cv_preproc_viz_f{f:04d}.png"
        cv2.imwrite(out, combo)
        print(f"f{f}: {len(dets)} blobs, turf-hue {turf_hue}, crop ({x0},{y0})-({x1},{y1}) -> {out}")


if __name__ == "__main__":
    main()

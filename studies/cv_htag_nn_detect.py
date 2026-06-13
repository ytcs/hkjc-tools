"""Run the trained YOLO avatar detector over the whole race -> detection cache for the tracker.

Emits the SAME format as cv_htag_dets_cache.py (frames_dets + ty0) so cv_htag_track.py consumes
it unchanged. Each YOLO box -> a detection dict with the avatar centre, a silk signature read
from the box's vivid/white pixels (the tracker's appearance veto), and the core-pixel area.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_nn_detect.py
Out:  data_cache/_htag_dets_nn_{RACE}.npz   (point cv_htag_track at it, or copy over the cache)
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
WEIGHTS = f"data_cache/htag_yolo_{RACE}/runs/avatar/weights/best.pt"
CONF = 0.25
IOU = 0.5


def main():
    import av
    import cv2
    from ultralytics import YOLO

    from cv_htag_detect import _silk_sig, class_maps
    from hkjc.vision.aerial_layout import legend_top, render_active

    model = YOLO(WEIGHTS)
    frames_dets, ty0s = [], {}
    n = 0
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2:
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        if not render_active(panel):
            continue
        band = panel[:legend_top(panel) - 2]
        res = model.predict(band, conf=CONF, iou=IOU, imgsz=960, verbose=False)[0]
        m = class_maps(band)
        vivid, white = m["vivid"], m["white"]
        lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB)
        dets = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ix0, iy0 = max(0, int(x0)), max(0, int(y0))
            ix1, iy1 = min(band.shape[1], int(x1) + 1), min(band.shape[0], int(y1) + 1)
            sub_v = vivid[iy0:iy1, ix0:ix1]
            sub_w = white[iy0:iy1, ix0:ix1]
            silk = (sub_v | sub_w)
            area = int(silk.sum()) or int((ix1 - ix0) * (iy1 - iy0) * 0.3)
            sig_px = lab[iy0:iy1, ix0:ix1][silk] if silk.any() else \
                lab[iy0:iy1, ix0:ix1].reshape(-1, 3)
            dets.append(dict(
                cx=float(cx), cy=float(cy), area=area, marker=True, split=False,
                n_vivid=int(sub_v.sum()), n_white=int(sub_w.sum()),
                n_body=0, conf=float(b.conf[0]), sig=_silk_sig(sig_px),
            ))
        frames_dets.append((i, dets))
        ty0s[i] = 0
        n += 1
        if n % 200 == 0:
            print(f"  {n} frames, last {len(dets)} dets")
    np.savez_compressed(f"data_cache/_htag_dets_nn_{RACE}.npz",
                        frames_dets=np.array(frames_dets, dtype=object),
                        ty0=np.array(ty0s, dtype=object))
    cnts = [len(d) for _f, d in frames_dets]
    print(f"cached {len(frames_dets)} frames; mean dets {np.mean(cnts):.2f} "
          f"-> data_cache/_htag_dets_nn_{RACE}.npz")


if __name__ == "__main__":
    main()

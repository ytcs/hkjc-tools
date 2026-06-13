"""Generate a YOLO dataset for the AVV avatar detector from the pipeline's OWN tags.

The classical detector + global tracker (cv_htag_track.py) produce verified per-horse tags in
the clean zones. Those reliable-frame positions are high-quality PSEUDO-LABELS for a learned
detector: the avatars are consistent synthetic sprites, so a net trained on the clean frames
should generalise to the hard zones (the disclaimer-text-occluded run-in, the tiny zoomed-out
bend) where the silk-blob heuristic plateaus.

Only FULLY-LABELLED frames are exported (exactly FIELD tags present), so YOLO never sees a real
avatar as unlabelled background. Box side = k * sqrt(area) (avatar footprint), clamped.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_nn_labels.py
Out:  data_cache/htag_yolo_{RACE}/ (images/{train,val}, labels/{train,val}, data.yaml)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03",
       "R02A": "aerial_20260520_02"}[RACE]
FIELD = {"R08": 11, "R03": 12, "R02A": 12}[RACE]
OUT = f"data_cache/htag_yolo_{RACE}"
BOX_K = 2.4           # box side = BOX_K * sqrt(area); avatar footprint + margin
BOX_MIN, BOX_MAX = 14, 64
VAL_EVERY = 6         # every 6th labelled frame -> validation split


def main():
    import av
    import cv2

    from hkjc.vision.aerial_layout import legend_top, render_active

    tr = np.load(f"data_cache/_htag_tracks_{RACE}.npz")["rows"]
    byframe = defaultdict(list)
    for f, tg, x, y, a in tr:
        byframe[int(f)].append((int(tg), float(x), float(y), float(a)))
    # COMPLETE-per-frame criterion: keep a frame iff every DETECTION got a tag (no unlabelled
    # avatar -> YOLO never learns a real avatar as background), even when the full field is not
    # visible (R02A shows ~10-11 of 12). Generalises to partial-visibility races; requiring the
    # full field would keep only the rare all-visible frames (biased to spread-out moments).
    zd = np.load(f"data_cache/_htag_dets_nn_{RACE}.npz", allow_pickle=True)
    ndet = {int(f): len(d) for f, d in zd["frames_dets"]}
    full = {f: v for f, v in byframe.items()
            if len(v) == ndet.get(f, -1) and len(v) >= FIELD - 3}
    print(f"{len(byframe)} emitted frames; {len(full)} complete-labelled "
          f"(all dets tagged, >= {FIELD - 3})")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(OUT, sub)
        os.makedirs(d, exist_ok=True)
        for p in os.listdir(d):
            os.remove(os.path.join(d, p))

    want = set(full)
    mx = max(want)
    n_tr = n_val = 0
    order = sorted(want)
    val_set = set(order[::VAL_EVERY])
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i not in want:
            if i > mx:
                break
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        if not render_active(panel):
            continue
        band = panel[:legend_top(panel) - 2]
        h, w = band.shape[:2]
        split = "val" if i in val_set else "train"
        cv2.imwrite(f"{OUT}/images/{split}/f{i:05d}.jpg", band,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for _tg, x, y, a in full[i]:
            side = float(np.clip(BOX_K * np.sqrt(max(a, 1.0)), BOX_MIN, BOX_MAX))
            lines.append(f"0 {x / w:.6f} {y / h:.6f} {side / w:.6f} {side / h:.6f}")
        with open(f"{OUT}/labels/{split}/f{i:05d}.txt", "w") as fh:
            fh.write("\n".join(lines))
        if split == "val":
            n_val += 1
        else:
            n_tr += 1

    with open(f"{OUT}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\n"
                 "names:\n  0: avatar\n")
    print(f"train {n_tr}  val {n_val}  -> {OUT}")


if __name__ == "__main__":
    main()

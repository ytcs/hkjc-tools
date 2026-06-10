"""LOOP 1 DRIVER (journal 0115/0116): candidate-pixel cleaning by the LONG-SMOOTH-CURVE gate.

The algorithm now lives in the library (``hkjc.vision.feeds.aerial_rail.clean_pixels``) — this script
is just the I/O driver + the VISUAL diagnostic overlay (the gate per the CV discipline: metrics lie).
GREEN = kept (on a long curve = a rail), RED = dropped (silks / text / minimap / FINISH / legend nums).
The rails must be fully GREEN edge-to-edge (incl. through the legend band).

    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_exp_pixtag.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03", "R02A": "aerial_20260520_02",
       "R01": "aerial_20260603_01", "R06": "aerial_20260603_06", "AWT": "aerial_awt_20260308_03"}[RACE]
OUTDIR = "data_cache/diag/exp_pixtag" + ("" if RACE == "R08" else f"_{RACE}")
CACHE = f"data_cache/_railclean_{RACE}.npz"


def main():
    import av
    import cv2

    from hkjc.vision.aerial_layout import content_box as _content_box
    from hkjc.vision.aerial_layout import legend_top as _legend_top
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial_rail import clean_pixels
    os.makedirs(OUTDIR, exist_ok=True)
    for p in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, p))

    keys, clean_list = [], []
    n = 0
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2:
            continue
        full = fr.to_ndarray(format="bgr24")
        panel = full[:, full.shape[1] // 2:]           # right (render) half of the split screen
        if not render_active(panel):
            continue
        ph = panel.shape[0]
        cx0, cx1, cy0, cy1 = _content_box(panel)
        y0, y1 = max(cy0, int(0.15 * ph)), cy1                 # NO legend cut (full content band)
        kept, dropped = clean_pixels(panel, y0, y1, cx1 - cx0)
        keys.append(i)
        clean_list.append(kept.astype(np.float32))
        vis = panel.copy()
        for p in dropped:
            cv2.circle(vis, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)      # RED = dropped
        for p in kept:
            cv2.circle(vis, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)      # GREEN = kept (rail)
        ylc = _legend_top(panel) - 8                            # OLD legend cut (outer rail now survives below)
        cv2.line(vis, (0, ylc), (panel.shape[1] - 1, ylc), (0, 255, 255), 1)
        cv2.putText(vis, f"f{i} GREEN=kept(rail) RED=dropped kept {len(kept)}/{len(kept)+len(dropped)} "
                    f"(yel=old legend cut)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{OUTDIR}/f{i:05d}.png", vis)
        n += 1
    np.savez(CACHE, keys=np.array(keys), clean=np.array(clean_list, dtype=object))
    print(f"{n} frames -> full-res overlays {OUTDIR}/  + cleaned-pixel cache {CACHE}")


if __name__ == "__main__":
    main()

"""LOOP 2 DRIVER (journal 0115/0116/0119): rail FIT (direct support) on the cleaned pixels.

The algorithm lives in the library (``hkjc.vision.feeds.aerial_rail.track_sequence`` — the temporal
fit-gather tracker) — this script is just the I/O driver + the VISUAL diagnostic overlay. It draws each
rail's DIRECT support fit (the curve fit to that frame's own clean rail pixels). The gate: every drawn
curve must lie ON the clean pixels (no floating segments). A rail is allowed to be SHORT where its
support is short -- honest; the old circle-based ``backproject_fixed_r`` fill that extrapolated short
slivers across the box was removed (it shot floating lines off curving rails; CV_RECON Phase 2 will do
principled non-circular spanning). Full-res overlays -> data_cache/diag/exp_railfit{,_<RACE>}/.

    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_exp_fit.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03", "R02A": "aerial_20260520_02"}[RACE]
CLEAN = f"data_cache/_railclean_{RACE}.npz"
OUTDIR = "data_cache/diag/exp_railfit" + ("" if RACE == "R08" else f"_{RACE}")


def main():
    import av
    import cv2

    from hkjc.vision.feeds.aerial_rail import track_sequence
    os.makedirs(OUTDIR, exist_ok=True)
    for p in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, p))
    zc = np.load(CLEAN, allow_pickle=True)
    keys, clean = list(zc["keys"]), zc["clean"]

    want = set(int(k) for k in keys)
    seq = track_sequence(keys, clean)                  # PASS 1: field-free continuity tracking

    n = 0
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i not in want:
            continue
        j = keys.index(i)
        pts = np.asarray(clean[j], np.float32)
        full = fr.to_ndarray(format="bgr24")
        panel = full[:, full.shape[1] // 2:]           # right (render) half of the split screen
        res = seq.get(i, [])
        for p in pts:                                  # clean pixels (dim cyan)
            cv2.circle(panel, (int(p[0]), int(p[1])), 1, (160, 160, 0), -1)
        nrails = 0
        for c in res:
            run = c.get("running")
            # Draw the rail's DIRECT support fit -- fit straight to this frame's own rail pixels. NO
            # back-projection fill: the old circle-based backproject_fixed_r EXTRAPOLATED a short support
            # across the box, which on a CURVING rail shoots a straight line off into the turf (the f694-710
            # "floating outer" / apparent weave was this overlay fill, not the library output -- the support
            # fit there is a correct short segment, med_dist ~0.6 px). A short rail stays short and honest;
            # principled spanning/fill is the job of the non-circular global second pass (CV_RECON Phase 2).
            Q = np.asarray(c["pts"], float)
            nrails += 1
            col = (0, 255, 0) if run else (255, 140, 0)
            pl = Q.astype(int)
            for tt in range(1, len(pl)):
                if 0 <= pl[tt, 1] < panel.shape[0] and 0 <= pl[tt, 0] < panel.shape[1]:
                    cv2.line(panel, tuple(pl[tt - 1]), tuple(pl[tt]), col, 2)
        cv2.putText(panel, f"f{i} GREEN=running/BLUE=other  CYAN=clean px  rails={nrails} (direct support fit)",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{OUTDIR}/f{i:05d}.png", panel)
        n += 1
    print(f"{n} frames -> full-res railfit overlays {OUTDIR}/  (direct support fit, no back-proj)")


if __name__ == "__main__":
    main()

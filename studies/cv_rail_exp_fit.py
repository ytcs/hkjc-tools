"""LOOP 2 DRIVER (journal 0115/0116): rail FIT + global-kappa back-projection on the cleaned pixels.

The algorithm now lives in the library (``hkjc.vision.feeds.aerial_rail``: ``track_sequence`` temporal
fit-gather tracker + ``build_kappa`` / ``backproject_fixed_r`` / ``radius_at`` global geometry) — this
script is just the I/O driver + the VISUAL diagnostic overlay. The two fitted curves must lie ON the
clean pixels and reach both content-box edges on EVERY frame (the gate; metrics are triage only).
Full-res overlays -> data_cache/diag/exp_railfit{,_<RACE>}/.

    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_exp_fit.py
"""
from __future__ import annotations

import glob as _glob
import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03", "R02A": "aerial_20260520_02"}[RACE]
CLEAN = f"data_cache/_railclean_{RACE}.npz"
_pp = _glob.glob(f"data_cache/_railprep_{RACE}_*.npz")              # for field centroids + odometer
PREP = max(_pp, key=lambda p: int(p.split('_')[-1].split('.')[0]) - int(p.split('_')[-2])) if _pp else None
OUTDIR = "data_cache/diag/exp_railfit" + ("" if RACE == "R08" else f"_{RACE}")


def main():
    import av
    import cv2
    from cv_rail_prep import _content_box

    from hkjc.vision.feeds.aerial_rail import _cw, backproject_fixed_r, build_kappa, radius_at, track_sequence
    os.makedirs(OUTDIR, exist_ok=True)
    for p in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, p))
    zc = np.load(CLEAN, allow_pickle=True)
    keys, clean = list(zc["keys"]), zc["clean"]
    field_of, odo_of = {}, {}
    if PREP:
        zp = np.load(PREP, allow_pickle=True)
        field_of = {int(k): pf for k, pf in zip(zp["keys"], zp["field"])}
        odo_of = {int(k): float(o) for k, o in zip(zp["keys"], zp["odo"])}

    want = set(int(k) for k in keys)
    seq = track_sequence(keys, clean, field_of)        # PASS 1: support-only fits (carries temporal seeds)
    s_run, k_run = build_kappa(seq, odo_of, running=True)   # GLOBAL geometry: kappa(s) per rail
    s_out, k_out = build_kappa(seq, odo_of, running=False)

    n = 0
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i not in want:
            continue
        j = keys.index(i)
        pts = np.asarray(clean[j], np.float32)
        full = fr.to_ndarray(format="bgr24")
        panel = full[:, full.shape[1] // 2:]           # right (render) half of the split screen
        box = _content_box(panel)
        Wb = _cw(box)
        s = odo_of.get(i, np.nan)
        res = seq.get(i, [])
        for p in pts:                                  # clean pixels (dim cyan)
            cv2.circle(panel, (int(p[0]), int(p[1])), 1, (160, 160, 0), -1)
        nrails = 0
        for c in res:
            run = c.get("running")
            R = radius_at(s, s_run, k_run, Wb) if run else radius_at(s, s_out, k_out, Wb)
            Q = backproject_fixed_r(c["pix"], R, box) if np.isfinite(s) else None   # PASS 2: map global geom back
            if Q is None or len(Q) < 2:
                Q = c["pts"]                           # fall back to the support-only fit
            nrails += 1
            col = (0, 255, 0) if run else (255, 140, 0)
            pl = Q.astype(int)
            for tt in range(1, len(pl)):
                if 0 <= pl[tt, 1] < panel.shape[0] and 0 <= pl[tt, 0] < panel.shape[1]:
                    cv2.line(panel, tuple(pl[tt - 1]), tuple(pl[tt]), col, 2)
        cv2.putText(panel, f"f{i} GREEN=running/BLUE=other  CYAN=clean px  rails={nrails} (global kappa back-proj)",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(f"{OUTDIR}/f{i:05d}.png", panel)
        n += 1
    print(f"{n} frames -> full-res railfit overlays {OUTDIR}/  (global kappa(s) back-projection)")


if __name__ == "__main__":
    main()

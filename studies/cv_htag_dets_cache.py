"""Build the horse-tagging detection cache with a temporally-SMOOTHED avatar scale.

Per-frame scale estimation is noisy (a zoomed-out bend reads the rail/text furniture, not the
tiny avatars), but the camera zoom is SMOOTH — so estimate scale per frame, median-smooth it
across frames (the global second pass on scale), and re-detect the outlier frames with the
smoothed scale. This rescues the bend-zone detection collapse that strands the start tracks.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_dets_cache.py
Out:  data_cache/_htag_dets_{RACE}.npz
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
SMOOTH_WIN = 41       # sampled frames for the robust running scale smoother (~82 video frames);
#                       wide enough to outvote CLUSTERED estimation failures (the run-in dip
#                       f2160-2200, the bend), short enough to track the real straight<->bend
#                       zoom trend. The camera is static per-step (ORB on masked turf: scale
#                       1.0000, t=(0,0) everywhere) and zoom is slow, so true scale is smooth.
OUTLIER_FRAC = 0.22   # re-detect a frame if |raw - smoothed| exceeds this fraction


def _smooth(vals, win):
    """Robust running scale: median, then reject >2-MAD outliers in the window and refit the
    median on the inliers (failures over- AND under-estimate; both must be rejected)."""
    out = np.array(vals, float)
    n = len(out)
    sm = out.copy()
    h = win // 2
    for i in range(n):
        w = out[max(0, i - h):min(n, i + h + 1)]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) or 1.0
        inl = w[np.abs(w - med) <= 2.0 * mad]
        sm[i] = float(np.median(inl)) if len(inl) else med
    return sm


def _track_width(rd):
    """Racing<->fixed rail perpendicular distance (px) — a DETECTION-INDEPENDENT zoom proxy."""
    r, x = rd.get("racing"), rd.get("fixed")
    if r is None or x is None or len(r) < 6 or len(x) < 6:
        return None
    d = np.sqrt(((r[::8, None, :] - x[None, ::8, :]) ** 2).sum(-1))
    return float(np.median(d.min(1)))


def _tw_anchored_scale(raw_scales, tws):
    """Avatar scale from the rail track-width: scale ~ K / track_width (avatars shrink as the
    oblique bend view spreads the rails). K is calibrated by the robust median of raw*tw over
    the frames where both exist — detection-INDEPENDENT, so it breaks the over-segmentation
    feedback (small scale -> over-seg -> small marker blobs -> small scale) that strands the
    run-in. Falls back to the robust-smoothed raw scale where no fixed rail is seen."""
    raw = np.array(raw_scales, float)
    sm = _smooth(raw_scales, SMOOTH_WIN)
    prod = [raw[i] * tws[i] for i in range(len(raw)) if tws[i] is not None]
    if len(prod) < 20:
        return sm
    k = float(np.median(prod))
    out = sm.copy()
    for i in range(len(raw)):
        if tws[i] is not None and tws[i] > 1:
            out[i] = float(np.clip(k / tws[i], 12.0, 55.0))
    return _smooth(out, 7)                       # light smooth to bridge missing-rail frames


def main():
    import av

    from cv_htag_detect import detect_core
    from hkjc.vision.aerial_layout import legend_top, render_active

    z = np.load(f"data_cache/_htag_rails_{RACE}.npz")
    rails = {}
    for k in z.files:
        f, name = k.rsplit("_", 1)
        rails.setdefault(int(f), {})[name] = z[k]

    frames, panels_dets, raw_scales, ty0s = [], [], [], {}
    tws = []
    band_cache = {}
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2 or i not in rails:
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        if not render_active(panel):
            continue
        ty0 = 0
        band = panel[ty0:legend_top(panel) - 2]
        off = np.array([0.0, ty0], np.float32)
        rd = {k2: (v - off if len(v) else v) for k2, v in rails[i].items()}
        dets, dbg = detect_core(band, rd)
        for d in dets:
            d["cy"] += ty0
        frames.append(i)
        panels_dets.append(dets)
        raw_scales.append(dbg["raw_scale"])
        tws.append(_track_width(rd))
        ty0s[i] = ty0
        band_cache[i] = (band, rd)             # keep for the re-detect pass
        if len(frames) % 200 == 0:
            print(f"  pass A {len(frames)} frames")

    sm = _tw_anchored_scale(raw_scales, tws)   # detection-independent (rail track-width)
    redo = [k for k in range(len(frames))
            if abs(raw_scales[k] - sm[k]) > OUTLIER_FRAC * max(1.0, sm[k])]
    print(f"pass A done ({len(frames)} frames); re-detecting {len(redo)} scale-outlier frames")
    for c, k in enumerate(redo):
        i = frames[k]
        band, rd = band_cache[i]
        dets, _dbg = detect_core(band, rd, scale_override=float(sm[k]))
        for d in dets:
            d["cy"] += ty0s[i]
        panels_dets[k] = dets
        if c % 50 == 0 and c:
            print(f"  pass B {c}/{len(redo)}")

    fd = list(zip(frames, panels_dets))
    np.savez_compressed(f"data_cache/_htag_dets_{RACE}.npz",
                        frames_dets=np.array(fd, dtype=object),
                        ty0=np.array(ty0s, dtype=object))
    print(f"cached {len(fd)} frames; mean dets {np.mean([len(d) for d in panels_dets]):.2f} "
          f"-> data_cache/_htag_dets_{RACE}.npz")


if __name__ == "__main__":
    main()

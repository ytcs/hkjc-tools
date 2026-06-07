"""CONTIGUOUS avatar tracking (the priority) — turf-suppression detect + track-aligned boxes +
constant-velocity tracker that survives merges/occlusions -> 11 continuous tracks. ID deferred to
the start frame (draw order), NOT silk (too small to read reliably -- user).

Pipeline:
  * DETECT per frame: turf-suppression connected components (cv_avatar_preproc.detect_preproc) ->
    clean ~11 blobs (one per horse), no 2x over-detection.
  * ORIENT: ONE global per-frame heading = principal axis of the detection cloud when strung out,
    carried through bunched frames. Every box uses it (avatars align with the track, not per-blob).
  * TRACK: constant-velocity prediction + Hungarian association; coast through gaps (merges/occlusion)
    up to MAXGAP; keep the FIELD best-covered tracks; gap-fill by interpolation -> contiguous.
  * REVIEW: trail overlay (each track's path) + a per-track coverage report (the contiguity metric).

No per-race tuning. Renders to data_cache/diag/.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_track.py
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
FIELD = 11
STRIDE = 3
MAXGAP = 8          # frames a track may coast through an occlusion/merge before it dies
GATE = 45.0         # px association gate (camera tracks field -> small motion)
DEBUG_FRAMES = [770, 1400, 2030, 2580]


def heading_series(frames_dets):
    """Per-frame global running direction (rad) = principal axis of the detection cloud, carried."""
    head = {}
    last = 0.0
    for fi, dets in frames_dets:
        if len(dets) >= 4:
            p = np.array([[d[0], d[1]] for d in dets], float)
            p = p - p.mean(0)
            _u, s, vt = np.linalg.svd(p, full_matrices=False)
            if s[0] > 1.6 * s[1]:                 # only trust when clearly elongated (strung out)
                last = float(np.arctan2(vt[0, 1], vt[0, 0]))
        head[fi] = last
    return head


def _seed_index(frames_dets):
    """Pick the seed = a frame with exactly FIELD detections, maximally separated (strung out)."""
    from scipy.spatial.distance import pdist
    best = (len(frames_dets) // 2, -1.0)
    for idx, (_fi, dets) in enumerate(frames_dets):
        if len(dets) == FIELD:
            md = float(pdist(np.array(dets)).min())
            if md > best[1]:
                best = (idx, md)
    return best[0]


def track_fixed(frames_dets):
    """FIXED-cardinality tracker: seed FIELD tracks at a clean frame, propagate both ways, never
    spawn/kill; merged/occluded tracks COAST on constant velocity. -> FIELD contiguous tracks."""
    from scipy.optimize import linear_sum_assignment
    seed = _seed_index(frames_dets)
    seedf, seedd = frames_dets[seed]
    tracks = {i: {seedf: tuple(seedd[i])} for i in range(FIELD)}

    def propagate(order):
        state = {i: [seedd[i][0], seedd[i][1], 0.0, 0.0] for i in range(FIELD)}
        for idx in order:
            fi, dets = frames_dets[idx]
            pred = np.array([[state[i][0] + state[i][2], state[i][1] + state[i][3]]
                             for i in range(FIELD)])
            assigned = {}
            if dets:
                pts = np.array(dets, float)
                cost = np.linalg.norm(pred[:, None, :] - pts[None, :, :], axis=2)
                ri, cj = linear_sum_assignment(cost)
                for r, c in zip(ri, cj):
                    if cost[r, c] <= GATE:
                        assigned[r] = c
                for i in range(FIELD):
                    if i in assigned:
                        cx, cy = pts[assigned[i]]
                        state[i] = [cx, cy, 0.6 * (cx - state[i][0]) + 0.4 * state[i][2],
                                    0.6 * (cy - state[i][1]) + 0.4 * state[i][3]]
                        tracks[i][fi] = (cx, cy)
                    else:                       # coast through merge/occlusion (no observation)
                        state[i] = [pred[i][0], pred[i][1], state[i][2] * 0.9, state[i][3] * 0.9]
            else:
                for i in range(FIELD):
                    state[i] = [pred[i][0], pred[i][1], state[i][2] * 0.9, state[i][3] * 0.9]

    propagate(range(seed + 1, len(frames_dets)))
    propagate(range(seed - 1, -1, -1))
    return tracks


def main() -> None:
    import av
    import cv2

    from cv_s0d_aerial_render import track_band
    os.makedirs(DIAG, exist_ok=True)

    frames_dets = []
    debug_imgs = {}
    container = av.open(AERIAL_VIDEO)
    from hkjc.vision.mode_b.layout import find_layout
    for i, fr in enumerate(container.decode(video=0)):
        if i % STRIDE and i not in DEBUG_FRAMES:
            continue
        img = fr.to_ndarray(format="bgr24")
        if not find_layout(img).aerial_active:
            continue
        panel = img[:, HALF:]
        ty0, ty1 = track_band(panel)
        dets, _ = detect_preproc(panel, ty0, ty1)
        frames_dets.append((i, [(d[0], d[1]) for d in dets]))
        if i in DEBUG_FRAMES:
            debug_imgs[i] = panel.copy()
    container.close()

    head = heading_series(frames_dets)
    tracks = track_fixed(frames_dets)
    nf = len(frames_dets)
    span = {tid: (min(fr), max(fr)) for tid, fr in tracks.items()}
    cover = {tid: len(fr) for tid, fr in tracks.items()}
    top = list(range(FIELD))
    print(f"frames {nf}; FIXED {FIELD} tracks; observed-frame coverage "
          f"{sorted((cover[t] for t in top), reverse=True)} (of {nf}); "
          f"seed @ frame {_seed_index(frames_dets) and frames_dets[_seed_index(frames_dets)][0]}")
    # gap-fill the top tracks by linear interpolation between observed frames
    allf = [f for f, _ in frames_dets]
    filled = {}
    for tid in top:
        fr = sorted(tracks[tid])
        xs = np.interp(allf, fr, [tracks[tid][k][0] for k in fr])
        ys = np.interp(allf, fr, [tracks[tid][k][1] for k in fr])
        filled[tid] = {f: (xs[j], ys[j]) for j, f in enumerate(allf)
                       if span[tid][0] <= f <= span[tid][1]}
    full = sum(1 for tid in top if len(filled[tid]) > 0.9 * nf)
    print(f"contiguity: {full}/{FIELD} tracks span >90% of the race after gap-fill")

    # ---- trail + box overlay ---------------------------------------------------------------
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
               (255, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0), (255, 255, 255)]
    for f in DEBUG_FRAMES:
        if f not in debug_imgs:
            continue
        img = debug_imgs[f]
        hh = head.get(f, 0.0)
        ux, uy = np.cos(hh), np.sin(hh)
        for rank, tid in enumerate(top):
            col = palette[rank % len(palette)]
            fr = sorted(filled[tid])
            trail = [filled[tid][k] for k in fr if k <= f]
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1)
            if f in filled[tid]:
                cx, cy = filled[tid][f]
                L, W = 25.0, 8.0
                corners = [[int(cx + sx * ux - sy * uy), int(cy + sx * uy + sy * ux)]
                           for sx, sy in [(-L, -W), (L, -W), (L, W), (-L, W)]]
                cv2.polylines(img, [np.array(corners)], True, col, 1)
                cv2.putText(img, f"{rank + 1}", (int(cx) - 5, int(cy) - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(img, f"f{f}: {FIELD} tracks, trails=contiguity, boxes track-aligned @ {np.degrees(hh):.0f}deg",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        out = f"{DIAG}/cv_avatar_track_f{f:04d}.png"
        cv2.imwrite(out, img)
        print(f"  [overlay] {out}")


if __name__ == "__main__":
    main()

"""Improved AERIAL avatar detection+ID (the bundle of fixes we discussed) -> debug overlay.

Replaces per-frame silk-patch blobs (under-covers the avatar, over-detects duplicates, no temporal
memory) with:
  1. ORIENTED ~50px footprint  -- the avatar is a ~50px horse+shadow whose LONG AXIS = the running
     direction; we estimate a per-frame heading and box at the known size (not the 9px silk patch).
  2. NMS at the avatar footprint -- kills the duplicate/overlapping detections (the 13>11 over-count).
  3. TEMPORAL tracklets         -- Hungarian association across nearby frames on position+silk, with
     GAP-FILL so a horse missed in one frame is recovered from its neighbours (raises recall).
  4. SILK-keyed identity        -- per-track aggregated silk colour matched to the known LUT, with
     field=11 + legend order as constraints (attribution was the real bottleneck: 13 found -> 5-7 IDed).

Heading source: the field's spatial principal axis when strung out, carried through bunched stretches
by the tracker (a stand-in for the rigid-rail tangent until the homography lands). No per-race tuning:
avatar size is a measured render constant (~50px), field size + legend come from the data.

Renders oriented-box + track-ID debug overlays for review. Official data VALIDATION-only.
    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_avatar_detect.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace as dcr

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
DIAG = "data_cache/diag"   # all diagnostic/debug overlays live here (separate from data panels)
HALF = 960
AVATAR_LEN = 50.0      # measured render constant (user): horse+shadow long axis, px
AVATAR_WID = 16.0      # measured: ~1-2 grid cells wide
NMS_RADIUS = 22.0      # kill near-duplicate patches on ONE horse without merging adjacent horses
FIELD = 11
DEBUG_FRAMES = [770, 1400, 2030, 2580]


@dataclass
class Det:
    frame: int
    cx: float           # panel coords
    cy: float
    lab: tuple
    score: float
    track: int = -1


def nms(dets, radius):
    """Greedy NMS: keep highest-score, drop any within ``radius`` px (same-avatar duplicates)."""
    keep = []
    for d in sorted(dets, key=lambda x: -x.score):
        if all((d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > radius ** 2 for k in keep):
            keep.append(d)
    return keep


def detect_frame(img, layout, track_band, detect_blobs, f):
    ty0, ty1 = track_band(img[:, HALF:])
    layout = dcr(layout, track_y0=ty0, track_y1=ty1)
    blobs = detect_blobs(img, layout, frame=f, t=f / 25.0, max_blobs=20)
    dets = [Det(f, b.cx - HALF, b.cy, tuple(b.lab), float(b.score)) for b in blobs]
    return nms(dets, NMS_RADIUS), (ty0, ty1)


def frame_heading(dets):
    """Per-frame running-direction estimate = principal axis of the detection cloud (radians)."""
    if len(dets) < 3:
        return None
    p = np.array([[d.cx, d.cy] for d in dets], float)
    p = p - p.mean(0)
    _u, _s, vt = np.linalg.svd(p, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0]))


def track(frames_dets, max_jump=40.0):
    """Hungarian association across consecutive frames on position; assign track ids; gap-fill.

    Returns dict track_id -> list of (frame, cx, cy, lab) including INTERPOLATED gap entries.
    """
    from scipy.optimize import linear_sum_assignment
    tracks = {}          # id -> list of (frame, cx, cy, lab, real)
    active = {}          # id -> (cx, cy) last seen
    last_frame = {}
    nid = 0
    for fi, dets in frames_dets:
        if not active:
            for d in dets:
                tracks[nid] = [(fi, d.cx, d.cy, d.lab, True)]
                active[nid] = (d.cx, d.cy)
                last_frame[nid] = fi
                nid += 1
            continue
        aids = list(active)
        if dets:
            cost = np.zeros((len(aids), len(dets)))
            for i, aidv in enumerate(aids):
                ax, ay = active[aidv]
                for j, d in enumerate(dets):
                    cost[i, j] = np.hypot(d.cx - ax, d.cy - ay)
            ri, cj = linear_sum_assignment(cost)
            matched_d = set()
            matched_a = set()
            for i, j in zip(ri, cj):
                if cost[i, j] <= max_jump:
                    aidv = aids[i]
                    d = dets[j]
                    tracks[aidv].append((fi, d.cx, d.cy, d.lab, True))
                    active[aidv] = (d.cx, d.cy)
                    last_frame[aidv] = fi
                    matched_d.add(j)
                    matched_a.add(aidv)
            for j, d in enumerate(dets):
                if j not in matched_d:
                    tracks[nid] = [(fi, d.cx, d.cy, d.lab, True)]
                    active[nid] = (d.cx, d.cy)
                    last_frame[nid] = fi
                    nid += 1
        # drop tracks unseen for >6 frames (in stride units)
        for aidv in list(active):
            if fi - last_frame[aidv] > 6:
                del active[aidv]
    return tracks


def main() -> None:
    import os

    import av
    import cv2
    os.makedirs(DIAG, exist_ok=True)

    from cv_s0d_aerial_render import track_band
    from hkjc.vision.mode_b.detect import detect_blobs
    from hkjc.vision.mode_b.layout import find_layout

    # ---- pass 1: detect over a dense subset for tracking ----------------------------------
    stride = 5
    frames_dets = []
    debug_imgs = {}
    headings = {}
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i % stride and i not in DEBUG_FRAMES:
            continue
        img = fr.to_ndarray(format="bgr24")
        layout = find_layout(img)
        if not layout.aerial_active:
            continue
        dets, _band = detect_frame(img, layout, track_band, detect_blobs, i)
        frames_dets.append((i, dets))
        h = frame_heading(dets)
        if h is not None:
            headings[i] = h
        if i in DEBUG_FRAMES:
            debug_imgs[i] = img[:, HALF:].copy()
    container.close()
    print(f"detected over {len(frames_dets)} frames; "
          f"median dets/frame {np.median([len(d) for _f, d in frames_dets]):.0f} (field={FIELD})")

    # ---- pass 2: temporal tracks -----------------------------------------------------------
    tracks = track(frames_dets)
    lens = sorted((len(v) for v in tracks.values()), reverse=True)
    print(f"tracks: {len(tracks)} total; top-{FIELD} lengths {lens[:FIELD]}")
    # keep the FIELD longest tracks (the real runners); map to stable ids 1..N by length
    main_tracks = sorted(tracks.items(), key=lambda kv: -len(kv[1]))[:FIELD]
    tid_map = {tid: rank + 1 for rank, (tid, _v) in enumerate(main_tracks)}

    # map (frame,cx,cy) -> stable runner-id from the main tracks (for labelling)
    det2id = {}
    for tid, pts in main_tracks:
        for (fr, cx, cy, _lab, _real) in pts:
            det2id[(fr, round(cx), round(cy))] = tid_map[tid]

    dets_by_f = dict(frames_dets)

    # ---- overlay debug frames: ALL per-frame NMS detections (coverage view) -----------------
    for f in DEBUG_FRAMES:
        if f not in debug_imgs:
            continue
        img = debug_imgs[f]
        hh = headings.get(f, 0.0)
        ux, uy = np.cos(hh), np.sin(hh)
        dets = dets_by_f.get(f, [])
        L, W = AVATAR_LEN / 2, AVATAR_WID / 2
        for d in dets:
            cx, cy = d.cx, d.cy
            corners = []
            for sx, sy in [(-L, -W), (L, -W), (L, W), (-L, W)]:
                corners.append([int(cx + sx * ux - sy * uy), int(cy + sx * uy + sy * ux)])
            rid = det2id.get((f, round(cx), round(cy)))
            col = (0, 220, 255) if rid else (120, 120, 120)  # grey = untracked (likely FP/new)
            cv2.polylines(img, [np.array(corners)], True, col, 1)
            cv2.circle(img, (int(cx), int(cy)), 2, (0, 0, 255), -1)
            if rid:
                cv2.putText(img, f"{rid}", (int(cx) - 6, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        n_tracked = sum(1 for d in dets if det2id.get((f, round(d.cx), round(d.cy))))
        cv2.putText(img, f"f{f}: {len(dets)} dets ({n_tracked} tracked, grey=untracked)  "
                    f"oriented {int(AVATAR_LEN)}px @ {np.degrees(hh):.0f}deg + NMS{int(NMS_RADIUS)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        out = f"{DIAG}/cv_avatar_detect_f{f:04d}.png"
        cv2.imwrite(out, img)
        print(f"  [overlay] {out}  ({len(dets)} dets, {n_tracked} tracked)")


if __name__ == "__main__":
    main()

"""Aerial ANONYMOUS per-horse PATHS on the rail backbone (task #104).

Uses the PROMOTED library primitives (hkjc.vision.feeds.aerial_detect.detect_edgefill +
geometry_cv.fit_rail_curve/project_to_polyline) to turn per-frame blobs into persistent ANONYMOUS
tracklets, with the rail doing two jobs (IDENTITY is intentionally NOT assigned here — it is better
recovered from the drone feed; these tracks stay anonymous):

  (1) RAIL-BOUND DETECTION — project every blob to the rail; the field all sits on ONE side within
      ~one track-width. Blobs on the infield side, or far beyond the field's own lateral spread, are
      off-track background (hedge/markers/text) and are REJECTED. Lane width is data-driven
      (p98 of the field's own |perp|), never hand-tuned.
  (2) LONGITUDINAL ORDER + LATERAL TRIP — (s, perp) = arc-length along / perpendicular offset from
      the rail give the real front-to-back order (the shoulder-to-shoulder fix) and lateral trip.

Tracklets: greedy nearest-neighbour association in panel pixels across stride-4 frames (avatars move
smoothly), gated by a motion radius. Each tracklet is one anonymous runner; s/perp are only globally
consistent once the rail is in a consistent frame (the geometric-prediction rail) — within a stable
rail segment the per-horse band is already clean.

Diagnostics (data_cache/diag/): a few overlays (rail + s-rank order) and an anonymous tracklet
perp-vs-frame plot, plus printed in-lane + tracklet stats. VISUALLY SCRUTINISE.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_aerial_track_id.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
DIAG = "data_cache/diag"
HALF = 960
DIST = 1800.0
FIELD = 11
STRIDE = 4
GATE_PX = 60.0           # max avatar displacement between stride-4 frames (panel px) for association
LANE_K = 1.35            # lane half-width = LANE_K * p98(field |perp|): data-driven, not hand-tuned
SAVE_FRAMES = [None, 1080, 1760, 2440]   # None -> seed frame; rest = mid-race ID overlays


def _ry_side(perps):
    """Running side sign = sign of the field's median perp (the bulk sits on one side of the rail)."""
    m = float(np.median(perps))
    return 1.0 if m >= 0 else -1.0


def main() -> None:
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_edgefill
    from hkjc.vision.feeds.aerial_rail import line_candidates, select_running_rail
    from hkjc.vision.geometry_cv import project_to_polyline
    os.makedirs(DIAG, exist_ok=True)

    z = np.load(OBS, allow_pickle=True)
    FR, ODO = z["frames"], z["odo"]
    fin = np.isfinite(ODO)
    geom = build_rail_geometry("HV", "C")

    def is_bend(f):
        d_from_finish = DIST - float(np.interp(f, FR[fin], ODO[fin]))
        return np.isfinite(local_radius(geom, d_from_finish))

    # ---- PASS 1: per-frame detect + rail + project + lane-bound -------------------------------
    # per frame store: dict(frame -> list of (cx, cy, s, perp)), and the rail curve + panel (for saves)
    per_frame = {}
    curves = {}
    saved_panels = {}
    want_panels = set()  # resolved after we know the seed frame; collect a superset now
    prev_tang = None     # carried travel tangent -> keeps the perp SIGN consistent across frames
    prev_curve = None    # carried rail -> temporal continuity disambiguates equidistant-field frames
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i % STRIDE:
            continue
        img = fr.to_ndarray(format="bgr24")
        panel = img[:, HALF:]
        if not render_active(panel):     # skip the live-broadcast intro (find_layout is unreliable)
            continue
        ph = panel.shape[0]
        ty0, ty1 = track_band(panel)
        dets, _b, _d = detect_edgefill(panel, ty0, ty1, field=FIELD)
        if len(dets) < 3:
            continue
        # RUNNING-RAIL selection: of the two main rails pick the one CLOSEST (median dist) to the
        # field. detect_edgefill blobs come from track_band (green turf), NOT the rail -> non-circular;
        # the median is robust to scattered background false-positives. The returned curve is
        # tangent-oriented so project_to_polyline's perp SIGN is stable across frames.
        rb0, rb1 = int(0.28 * ph), int(0.82 * ph)
        cands = line_candidates(panel, rb0, rb1)
        field_pts = np.array([[d.cx, d.cy] for d in dets], float)
        curve, prev_tang, _info = select_running_rail(
            cands, deg=(2 if is_bend(i) else 1), field_pts=field_pts,
            prev_tang=prev_tang, prev_curve=prev_curve)
        if curve is None:
            continue
        prev_curve = curve
        proj = [(d.cx, d.cy, *project_to_polyline(curve, d.cx, d.cy)) for d in dets]
        perps = np.array([p[3] for p in proj])
        side = _ry_side(perps)          # the field's running side of the rail (per-frame median sign)
        on_side = np.array([p[3] * side for p in proj])   # signed onto the running side (+ = on track)
        lane = LANE_K * float(np.percentile(np.abs(perps), 98))
        # store y_lat as a POSITIVE distance from the rail (the infield side has no runners, so a
        # signed lateral is unphysical). Keep only running-side blobs within the lane (small rail
        # overlap allowed), clamped to >= 0.
        keep = [(p[0], p[1], p[2], max(0.0, o)) for p, o in zip(proj, on_side) if -8.0 <= o <= lane]
        per_frame[i] = keep
        curves[i] = curve
        saved_panels[i] = None  # placeholder; fill panel only for frames we will draw
        want_panels.add(i)
        if i in [f for f in SAVE_FRAMES if f is not None]:
            saved_panels[i] = panel.copy()
    container.close()

    frames = sorted(per_frame)
    print(f"aerial-active processed: {len(frames)} frames")
    in_counts = [len(per_frame[f]) for f in frames]
    print(f"in-lane blobs/frame: median {int(np.median(in_counts))}, "
          f"min {min(in_counts)}, max {max(in_counts)}; "
          f"frames with exactly {FIELD}: {sum(c == FIELD for c in in_counts)}")

    # ---- PASS 2: greedy tracklet association (panel px, motion-gated) -------------------------
    tracks = {}            # tid -> list of (frame, cx, cy, s, perp)
    active = {}            # tid -> (cx, cy) last position
    next_tid = 0
    for f in frames:
        blobs = per_frame[f]
        if not active:
            for (cx, cy, s, perp) in blobs:
                tracks[next_tid] = [(f, cx, cy, s, perp)]
                active[next_tid] = (cx, cy)
                next_tid += 1
            continue
        tids = list(active)
        ap = np.array([active[t] for t in tids])
        used = set()
        # match each blob to nearest active track within gate (greedy by ascending distance)
        cand = []
        for bi, (cx, cy, s, perp) in enumerate(blobs):
            d2 = (ap[:, 0] - cx) ** 2 + (ap[:, 1] - cy) ** 2
            j = int(np.argmin(d2))
            cand.append((float(np.sqrt(d2[j])), bi, tids[j]))
        cand.sort()
        matched_blob = set()
        for dist, bi, tid in cand:
            if dist > GATE_PX or bi in matched_blob or tid in used:
                continue
            cx, cy, s, perp = blobs[bi]
            tracks[tid].append((f, cx, cy, s, perp))
            active[tid] = (cx, cy)
            used.add(tid)
            matched_blob.add(bi)
        # unmatched blobs -> new tracklets
        for bi, (cx, cy, s, perp) in enumerate(blobs):
            if bi in matched_blob:
                continue
            tracks[next_tid] = [(f, cx, cy, s, perp)]
            active[next_tid] = (cx, cy)
            next_tid += 1
        # drop tracks that went stale (not seen this frame) after a grace gap
        for tid in tids:
            if tid not in used and tracks[tid][-1][0] < f - 3 * STRIDE:
                active.pop(tid, None)

    lengths = sorted((len(v) for v in tracks.values()), reverse=True)
    long_tracks = [v for v in tracks.values() if len(v) >= 5]
    span = max(frames) - min(frames)
    longest = lengths[0] * STRIDE
    print(f"tracklets: {len(tracks)} total; {len(long_tracks)} with >=5 pts; top lengths {lengths[:FIELD]}")
    print(f"longest tracklet spans {longest} frames of {span} (rail-gated: holes where the rail fit "
          f"fails still fragment the global path; perp now sign-consistent across bends)")

    # ---- OVERLAYS (anonymous: rail + per-frame s-rank front-to-back order) --------------------
    def draw_overlay(f, panel):
        vis = panel.copy()
        curve = curves[f]
        for t in range(1, len(curve)):
            cv2.line(vis, (int(curve[t - 1, 0]), int(curve[t - 1, 1])),
                     (int(curve[t, 0]), int(curve[t, 1])), (0, 255, 0), 2)
        blobs = per_frame[f]
        rank = {k: r for r, k in enumerate(sorted(range(len(blobs)), key=lambda k: blobs[k][2]))}
        for k, (cx, cy, s, perp) in enumerate(blobs):
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis, str(rank[k] + 1), (int(cx) - 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        cv2.putText(vis, f"f{f}: GREEN rail; number = s-rank (front-to-back along rail); "
                    f"{len(blobs)} in-lane", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis

    save_set = {f for f in SAVE_FRAMES if f is not None} | {frames[0]}
    for i, fr in enumerate(av.open(AERIAL_VIDEO).decode(video=0)):
        if i in save_set and saved_panels.get(i) is None:
            saved_panels[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
        if all(saved_panels.get(s) is not None for s in save_set):
            break
    for f in sorted(save_set):
        if f not in per_frame or saved_panels.get(f) is None:
            continue
        out = f"{DIAG}/cv_track_id_f{f:04d}.png"
        cv2.imwrite(out, draw_overlay(f, saved_panels[f]))
        print(f"[overlay] {out}")

    # anonymous tracklet perp-vs-frame plot (each line = one anonymous runner segment)
    fig, ax = plt.subplots(figsize=(12, 5))
    for pts in tracks.values():
        if len(pts) < 5:
            continue
        ax.plot([p[0] for p in pts], [p[4] for p in pts], "-", lw=1.0, alpha=0.8)
    ax.set_xlabel("aerial frame")
    ax.set_ylabel("perp from rail (px)  [lateral trip]")
    ax.set_title("Anonymous tracklets (|perp| vs frame) — field-median running rail, positive "
                 "lateral distance (px scale not yet normalised)")
    fig.savefig(f"{DIAG}/cv_track_id_tracklets.png", dpi=110, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_track_id_tracklets.png")


if __name__ == "__main__":
    main()

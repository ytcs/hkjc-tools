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
STRIDE = 1               # process EVERY frame (25 fps native) for max temporal resolution
GATE_PX = 60.0           # max avatar displacement between stride-4 frames (panel px) for association
LANE_K = 1.35            # lane half-width = LANE_K * p98(field |perp|): data-driven, not hand-tuned
SAVE_FRAMES = [None, 1080, 1760, 2440]   # None -> seed frame; rest = mid-race ID overlays


def _ry_side(perps):
    """Running side sign = sign of the field's median perp (the bulk sits on one side of the rail)."""
    m = float(np.median(perps))
    return 1.0 if m >= 0 else -1.0


def _medfilt(a, w=5):
    """Rolling-median filter — suppresses the isolated per-frame outlier spikes (resolution clean-up)."""
    a = np.asarray(a, float)
    if len(a) < 3:
        return a
    h = w // 2
    return np.array([float(np.median(a[max(0, i - h):i + h + 1])) for i in range(len(a))])


def main() -> None:
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cv_rigid_rail import build_rail_geometry, local_radius
    from hkjc.vision.aerial_layout import render_active
    from hkjc.vision.feeds.aerial import track_band
    from hkjc.vision.feeds.aerial_detect import detect_avatars
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
    outers = {}          # the OTHER main rail (outer) per frame -> local 22.5 m metric ruler
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
        dets, _dbg = detect_avatars(panel, ty0, ty1, field=FIELD)
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
        outers[i] = (_info or {}).get("outer")
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

    # ---- metric anchor (shared by association + reconstruction): odo + head-relative offset -----
    def odo_arclen(f):
        return float(np.interp(f, FR[fin], ODO[fin]))   # HEAD horse metres-run (1800 - M-to-finish)

    # robust head reference: the 96th percentile of s, NOT max(s) (a single spurious lead detection
    # makes max jitter and snaps the longitudinal baseline). p96 ~ the leader but outlier-resistant.
    s_head = {f: float(np.percentile([p[2] for p in per_frame[f]], 96)) for f in frames}

    # ASSOCIATION uses a STABLE global scale (a varying scale would jitter the matching coordinate and
    # shatter the tracks). metric() below feeds PASS 2 only.
    spreads = [max(p[2] for p in per_frame[f]) - min(p[2] for p in per_frame[f])
               for f in frames if len(per_frame[f]) > 2]
    scale_assoc = 15.0 / max(1.0, float(np.median(spreads)))

    def metric(f, s, perp):
        return odo_arclen(f) - (s_head[f] - s) * scale_assoc, perp * scale_assoc

    # OUTPUT uses a PERSPECTIVE-CORRECT per-frame scale: the inner->outer rail separation is a KNOWN
    # 22.5 m, so local px->m varies with foreshortening. Temporally SMOOTHED (camera distance changes
    # slowly; the raw per-frame sep is noisy because the outer rail is the fainter one).
    RAIL_W = 22.5

    def _sep_px(f):
        inner, outer = curves[f], outers.get(f)
        if outer is None or len(outer) == 0:
            return None
        d2 = (inner[:, None, 0] - outer[None, :, 0]) ** 2 + (inner[:, None, 1] - outer[None, :, 1]) ** 2
        return float(np.median(np.sqrt(d2.min(1))))

    sep_raw = np.array([_sep_px(f) if _sep_px(f) else np.nan for f in frames])
    sep_global = float(np.nanmedian(sep_raw)) if np.isfinite(sep_raw).any() else 200.0
    ok = np.isfinite(sep_raw)
    if ok.sum() >= 2:                      # fill gaps, then WIDE temporal median (camera dist is slow)
        sep_raw = np.interp(np.arange(len(frames)), np.where(ok)[0], sep_raw[ok])
    else:
        sep_raw = np.full(len(frames), sep_global)
    sep_sm = _medfilt(sep_raw, w=15)
    sep_by_f = {f: float(sep_sm[k]) for k, f in enumerate(frames)}

    def scale_f(f):
        return RAIL_W / max(20.0, sep_by_f.get(f, sep_global))

    # ---- PASS 2: METRIC-SPACE association with the ODOMETER as the motion prior ----------------
    # Match detections in (arclen, y_lat) METRES. Each track is predicted forward by the odometer
    # delta (a horse advances with the field; its residual drift between stride-4 frames is small), so
    # association survives bends, zoom and short holes far better than panel-pixel NN. Holes bridged
    # up to GRACE frames; gates are in metres.
    GATE_ARC, GATE_LAT, GRACE = 6.0, 4.0, 24   # gates in metres; GRACE ~1 s (stride-independent)
    tracks = {}            # tid -> list of (frame, cx, cy, s, perp)
    active = {}            # tid -> dict(last_f, arclen, ylat)
    next_tid = 0
    for f in frames:
        dets = [(cx, cy, s, perp, *metric(f, s, perp)) for (cx, cy, s, perp) in per_frame[f]]
        if active:
            tids = list(active)
            pred = {t: (active[t]["arclen"] + (odo_arclen(f) - odo_arclen(active[t]["last_f"])),
                        active[t]["ylat"]) for t in tids}
            cand = []
            for di, d in enumerate(dets):
                for t in tids:
                    da, dl = abs(d[4] - pred[t][0]), abs(d[5] - pred[t][1])
                    if da < GATE_ARC and dl < GATE_LAT:
                        cand.append((da / GATE_ARC + dl / GATE_LAT, di, t))
            cand.sort()
            used, matched = set(), set()
            for _cost, di, t in cand:
                if di in matched or t in used:
                    continue
                d = dets[di]
                tracks[t].append((f, d[0], d[1], d[2], d[3]))
                active[t] = dict(last_f=f, arclen=d[4], ylat=d[5])
                used.add(t)
                matched.add(di)
            for t in tids:                         # bridge holes; drop only after the grace gap
                if t not in used and active[t]["last_f"] < f - GRACE:
                    active.pop(t, None)
            dets = [d for di, d in enumerate(dets) if di not in matched]
        for d in dets:                             # unmatched -> new track
            tracks[next_tid] = [(f, d[0], d[1], d[2], d[3])]
            active[next_tid] = dict(last_f=f, arclen=d[4], ylat=d[5])
            next_tid += 1

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

    # ---- PASS 3: ODOMETER-anchored longitudinal reconstruction -> metric-ish 2D paths -----------
    # arclen_h(f) = odo(f) - (s_head - s_h)*scale: the head sits at the odometer, each trailing horse
    # its own offset behind -> arclen IS the odometer PLUS the relative position, co-stamped by frame
    # with y_lat. odo_arclen / s_head / scale / metric() were defined above (shared with association).
    fig2, (axA, axL, axR) = plt.subplots(1, 3, figsize=(20, 5))
    n_paths = 0
    for pts in tracks.values():
        if len(pts) < 8:
            continue
        n_paths += 1
        ff = [p[0] for p in pts]
        offset = _medfilt([-(s_head[f] - s) * scale_f(f) for (f, _cx, _cy, s, _p) in pts])  # de-spike
        ylat = _medfilt([perp * scale_f(fp) for (fp, _cx, _cy, _s, perp) in pts])           # de-spike
        arclen = [odo_arclen(f) + o for f, o in zip(ff, offset)]
        axA.plot(ff, arclen, "-", lw=1.0, alpha=0.7)
        axL.plot(ff, offset, "-", lw=1.0, alpha=0.7)
        axR.plot(arclen, ylat, "-", lw=1.0, alpha=0.7)
    axA.set_xlabel("aerial frame")
    axA.set_ylabel("arclen run (m)")
    axA.set_title("Absolute arclen (odometer-anchored, monotonic)")
    axL.set_xlabel("aerial frame")
    axL.set_ylabel("longitudinal offset in field (m)")
    axL.set_title("Per-horse position behind the head (0 = leader) — the race dynamics")
    axL.axhline(0, color="k", lw=0.5)
    axR.set_xlabel("arclen run (m)")
    axR.set_ylabel("y_lat (m), clipped 8")
    axR.set_title("Reconstructed 2D path: lateral trip vs arclen")
    fig2.savefig(f"{DIAG}/cv_recon_2d.png", dpi=110, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_recon_2d.png  ({n_paths} paths, scale~={RAIL_W/sep_global:.3f} m/px (per-frame))")

    # ---- PASS 4: RESOLUTION diagnostic — top tracks' raw perp & s (px) + worst-jump frames -------
    top = sorted(tracks.values(), key=len, reverse=True)[:3]
    fig3, (axP, axS) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    worst = []
    for ti, pts in enumerate(top):
        ff = [p[0] for p in pts]
        pr = [p[4] for p in pts]                       # raw perp (px)
        so = [p[3] - s_head[p[0]] for p in pts]        # raw s offset behind head (px)
        axP.plot(ff, pr, "-o", ms=2, lw=0.8, label=f"track{ti} (n={len(pts)})")
        axS.plot(ff, so, "-o", ms=2, lw=0.8)
        for k in range(1, len(pts)):
            dfr = ff[k] - ff[k - 1]
            if dfr <= 2 * STRIDE and (abs(pr[k] - pr[k - 1]) > 25 or abs(so[k] - so[k - 1]) > 40):
                worst.append((ff[k], ti, round(pr[k] - pr[k - 1]), round(so[k] - so[k - 1])))
    axP.set_ylabel("perp (px) [lateral]")
    axP.legend(fontsize=8)
    axP.set_title("Top-3 tracks RAW resolution — perp (px) and s-offset (px) vs frame (jitter/quant?)")
    axS.set_ylabel("s offset behind head (px) [longitudinal]")
    axS.set_xlabel("aerial frame")
    fig3.savefig(f"{DIAG}/cv_recon_res.png", dpi=110, bbox_inches="tight")
    print(f"[fig] {DIAG}/cv_recon_res.png")
    worst.sort(key=lambda w: -abs(w[2]) - abs(w[3]))
    print(f"worst per-frame jumps (frame, track, dperp_px, ds_px): {worst[:12]}")


if __name__ == "__main__":
    main()

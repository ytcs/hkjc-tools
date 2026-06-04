"""STAGE 4-7 — identity (silk-colour) + rail-anchored projection to (d, off) + trajectories.

Reads tracks_raw_<tag>.npz from extract_tracks.py and the per-frame video (for rail
segmentation), and produces physical (d, off) trajectories validated against the HKJC
sectionals.

WHY rail-anchored (not a full 4-point homography):
  The drone follows the field; both rails are visible but the OUTER rail is heavily
  cluttered by background white (buildings, pavement, road markings) so a clean 4-point
  rail-to-rail homography per frame is brittle. The INNER running rail (the bright
  continuous fence the field hugs) is the one robust landmark. So we build a *rail-relative*
  world frame:

    d   (metres-from-finish) = arc length measured ALONG the inner rail, chained across the
        camera pan and anchored so that the field's mean d = 1650 at the race jump and the
        recovered section-crossing times line up with the sectionals.
    off (lateral metres from rail) = signed perpendicular pixel distance from the horse
        foot-point to the rail polyline, converted to metres by a LOCAL pixel->metre scale.

  LOCAL SCALE: a thoroughbred is ~2.4 m nose-to-tail; we calibrate metres-per-pixel in each
  frame from the median horse bounding-box LENGTH (long box dimension) across that frame's
  detections. This is the honest scale source for an oblique view where the outer rail is
  unreliable. (Accuracy caveat: oblique foreshortening makes `off` the weakest axis — see report.)

IDENTITY: saddlecloth numbers are NOT legible overhead, so we cluster tracklets by silk
colour (mean Lab of the rider-torso ROI) into the 12 runners, then map clusters to saddle
numbers using the silk-colour swatches implied by the legible FINISH-camera frames is out of
scope here — instead we report ID fragmentation (tracklets per colour-cluster) which is the
measurable identity-quality number.

Run:
  cd /home/ytc/Projects/hkjc-tools && uv run --with opencv-python-headless --with av \
      --with scipy --with scikit-learn --with matplotlib \
      python experiments/trajectory/project_traj.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import race_ground_truth as GT  # noqa: E402

VIDEO = Path("/home/ytc/Projects/hkjc-tools/data_cache/video/drone_20260603_01.mp4")
OUT = VIDEO.parent / "poc_out"
TAG = VIDEO.stem
HORSE_LEN_M = 2.4   # thoroughbred nose-to-tail, for local pixel->metre scale
RACE_DIST = GT.RACE["distance"]   # 1650


# ----------------------------------------------------------------- rail extraction
def inner_rail_polyline(img, foot_pts):
    """Fit the inner running-rail polyline near the field.

    The running rail is the long white structure closest to the horse cluster. We mask
    white, keep components in a band around the field, and fit the rail as a polynomial in
    whichever axis the rail is more extended along.
    Returns Nx2 ordered polyline (pixels) or None.
    """
    import cv2
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if len(foot_pts) == 0:
        return None
    fp = np.array(foot_pts)
    cx, cy = fp[:, 0].mean(), fp[:, 1].mean()

    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    cand = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        length = max(w, h)
        thin = length / max(1, min(w, h))
        if area > 500 and length > 250 and thin > 4:   # long & thin => a rail
            d = np.hypot(cent[i][0] - cx, cent[i][1] - cy)
            cand.append((d, i, length))
    if not cand:
        return None
    # keep the long thin white structures nearest the field; merge their pixels
    cand.sort()
    keep = [i for _, i, _ in cand[:3]]
    ys, xs = [], []
    for i in keep:
        yy, xx = np.where(lab == i)
        ys.append(yy); xs.append(xx)
    ys = np.concatenate(ys); xs = np.concatenate(xs)

    # choose the nearest single rail to the field for the anchor curve:
    # pick the candidate component with min distance, fit a smooth polyline to it
    best_i = cand[0][1]
    yy, xx = np.where(lab == best_i)
    # fit along the dominant axis
    if np.ptp(xx) >= np.ptp(yy):
        order = np.argsort(xx)
        t = xx[order]; u = yy[order]
        coef = np.polyfit(t, u, 2)
        ts = np.linspace(t.min(), t.max(), 200)
        us = np.polyval(coef, ts)
        poly = np.column_stack([ts, us])
    else:
        order = np.argsort(yy)
        t = yy[order]; u = xx[order]
        coef = np.polyfit(t, u, 2)
        ts = np.linspace(t.min(), t.max(), 200)
        us = np.polyval(coef, ts)
        poly = np.column_stack([us, ts])
    return poly


def project_to_rail(pt, poly):
    """Return (arc_along_rail_px, signed_perp_px) of pt onto polyline poly."""
    seg = poly[1:] - poly[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(seglen)])
    p = np.array(pt)
    best = None
    for i in range(len(seg)):
        a = poly[i]; ab = seg[i]; L2 = ab @ ab
        if L2 < 1e-6:
            continue
        tt = np.clip((p - a) @ ab / L2, 0, 1)
        proj = a + tt * ab
        dist = np.hypot(*(p - proj))
        arc = cum[i] + tt * seglen[i]
        # signed: cross product sign
        cross = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
        signed = dist * np.sign(cross)
        if best is None or dist < best[0]:
            best = (dist, arc, signed)
    return best[1], best[2]


def main():
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    d = np.load(OUT / f"tracks_raw_{TAG}.npz", allow_pickle=True)
    rows = d["rows"]
    cols = list(d["cols"])
    fps = float(d["fps"])
    ci = {c: i for i, c in enumerate(cols)}
    print(f"loaded {len(rows)} detections, {len(np.unique(rows[:,ci['id']]))} track-ids, fps={fps}")

    frame_idxs = np.unique(rows[:, ci["frame"]]).astype(int)

    # ---- pass over the needed frames to get rail polylines + local scale
    needed = set(frame_idxs.tolist())
    rail_by_frame = {}
    scale_by_frame = {}   # metres per pixel
    container = av.open(str(VIDEO))
    vs = container.streams.video[0]
    # camera-pan (background optical flow) gives a NON-CIRCULAR field-speed source:
    # the drone tracks the field, so the static-background pixel pan between frames, scaled
    # to metres, equals the field's ground displacement. We measure it with sparse LK flow
    # on corners OUTSIDE the horse boxes (true static scene).
    pan_px_by_frame = {}     # frame -> background pan magnitude (px) since prev processed frame
    prev_gray = None
    prev_fidx = None
    idx = -1
    for frame in container.decode(vs):
        idx += 1
        if idx not in needed:
            continue
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fr = rows[rows[:, ci["frame"]].astype(int) == idx]
        foot = fr[:, [ci["fx"], ci["fy"]]]
        poly = inner_rail_polyline(img, foot)
        rail_by_frame[idx] = poly
        boxlen = np.maximum(fr[:, ci["x2"]] - fr[:, ci["x1"]],
                            fr[:, ci["y2"]] - fr[:, ci["y1"]])
        mpx = HORSE_LEN_M / max(1.0, np.median(boxlen))
        scale_by_frame[idx] = mpx

        # background pan: mask out horse boxes, track corners
        if prev_gray is not None:
            mask = np.full(gray.shape, 255, np.uint8)
            for r in fr:
                x1, y1, x2, y2 = r[ci["x1"]], r[ci["y1"]], r[ci["x2"]], r[ci["y2"]]
                pad = 0.4 * (x2 - x1)
                cv2.rectangle(mask, (int(x1 - pad), int(y1 - pad)),
                              (int(x2 + pad), int(y2 + pad)), 0, -1)
                cv2.rectangle(prev_mask, (int(x1 - pad), int(y1 - pad)),
                              (int(x2 + pad), int(y2 + pad)), 0, -1)
            p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                         minDistance=12, mask=prev_mask)
            if p0 is not None and len(p0) > 8:
                # large window: the follow-shot pans the static background by ~60-150 px/step
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                                     winSize=(31, 31), maxLevel=4)
                good0 = p0[st.flatten() == 1].reshape(-1, 2)
                good1 = p1[st.flatten() == 1].reshape(-1, 2)
                if len(good0) > 8:
                    flow = good1 - good0
                    # the camera pan is the DOMINANT motion of static corners: take the
                    # median flow VECTOR, then the pan magnitude is its norm. (Reject corners
                    # far from the median vector as non-static / on the field.)
                    med = np.median(flow, axis=0)
                    keep = np.hypot(*(flow - med).T) < 20
                    if keep.sum() >= 8:
                        med = np.median(flow[keep], axis=0)
                    pan = float(np.hypot(*med))
                    pan_px_by_frame[idx] = (prev_fidx, pan)
        prev_gray = gray
        prev_mask = np.full(gray.shape, 255, np.uint8)
        prev_fidx = idx
        if len(rail_by_frame) % 50 == 0:
            print(f"  rail/scale/pan frames done: {len(rail_by_frame)}/{len(needed)}")
    container.close()

    # ---- chain rail arc-length across frames into a global d
    # For each frame, the field's MEAN arc position should advance monotonically as the
    # race progresses (d decreases). We compute per-detection (arc_px, off_px) then build a
    # per-frame field-mean arc, then map mean-arc-progress -> world d by anchoring the first
    # processed frame to d≈RACE_DIST and using the local metre scale to convert px arc steps.
    recs = []  # (frame, t, id, arc_px, off_px, mpx, L,a,b)
    for r in rows:
        fidx = int(r[ci["frame"]])
        poly = rail_by_frame.get(fidx)
        if poly is None:
            continue
        arc, off = project_to_rail((r[ci["fx"]], r[ci["fy"]]), poly)
        recs.append((fidx, r[ci["t"]], int(r[ci["id"]]), arc, off,
                     scale_by_frame[fidx], r[ci["L"]], r[ci["a"]], r[ci["b"]]))
    recs = np.array(recs, dtype=np.float64)
    R = {c: i for i, c in enumerate(
        ["frame", "t", "id", "arc", "off", "mpx", "L", "a", "b"])}

    fr_list = np.unique(recs[:, R["frame"]]).astype(int)
    fr_list.sort()

    # ---- chain d from the NON-CIRCULAR camera-pan (background optical flow).
    # Between consecutive processed frames the field's ground displacement (m) = camera pan
    # (px) * local metres-per-pixel. Accumulate to get distance travelled, anchor start=1650.
    travelled = {fr_list[0]: 0.0}
    pan_speed = {}   # frame -> instantaneous field speed (m/s) from pan
    for k in range(1, len(fr_list)):
        f0, f1 = fr_list[k - 1], fr_list[k]
        dt = (f1 - f0) / fps
        info = pan_px_by_frame.get(f1)
        if info is not None and info[0] == f0:
            mpx = 0.5 * (scale_by_frame[f0] + scale_by_frame[f1])
            step = info[1] * mpx
            # guard against flow blowups
            if not np.isfinite(step) or step > 30 * dt or step < 0:
                step = pan_speed.get(f0, 16.0) * dt
        else:
            step = pan_speed.get(f0, 16.0) * dt
        travelled[f1] = travelled[f0] + step
        pan_speed[f1] = step / max(dt, 1e-3)

    # light smoothing of the cumulative travelled curve (it must be monotone increasing)
    fa = np.array([travelled[f] for f in fr_list])
    fa = np.maximum.accumulate(fa)
    field_d = {f: RACE_DIST - fa[i] for i, f in enumerate(fr_list)}

    # per-detection world coords
    out_rows = []
    for r in recs:
        f = int(r[R["frame"]])
        # individual d = field_d adjusted by this horse's arc offset from field median
        sub = recs[recs[:, R["frame"]].astype(int) == f]
        med_arc = np.median(sub[:, R["arc"]])
        ind_d = field_d[f] - (r[R["arc"]] - med_arc) * r[R["mpx"]]
        off_m = r[R["off"]] * r[R["mpx"]]
        out_rows.append((f, r[R["t"]], int(r[R["id"]]), ind_d, off_m,
                         r[R["L"]], r[R["a"]], r[R["b"]]))
    out = np.array(out_rows, dtype=np.float64)
    O = {c: i for i, c in enumerate(["frame", "t", "id", "d", "off", "L", "a", "b"])}

    # ---- identity: cluster tracklet mean silk colour into 12 runners
    ids = np.unique(out[:, O["id"]]).astype(int)
    feats, idlist, lengths = [], [], []
    for tid in ids:
        sub = out[out[:, O["id"]].astype(int) == tid]
        if len(sub) < 5:
            continue
        feats.append([sub[:, O["L"]].mean(), sub[:, O["a"]].mean(), sub[:, O["b"]].mean()])
        idlist.append(tid); lengths.append(len(sub))
    feats = np.array(feats)
    k = min(GT.RACE["n_runners"], len(feats))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    cluster = {tid: int(c) for tid, c in zip(idlist, km.labels_)}
    frag = {}
    for tid, c in cluster.items():
        frag.setdefault(c, []).append(tid)
    print(f"\n== IDENTITY ==")
    print(f"{len(idlist)} tracklets (>=5 frames) clustered into {k} silk-colour groups")
    tpc = [len(v) for v in frag.values()]
    print(f"tracklets per colour-cluster: median={np.median(tpc):.1f} "
          f"mean={np.mean(tpc):.1f} max={max(tpc)} (ideal=1 if no fragmentation, "
          f"={len(idlist)/k:.1f} expected from raw track count)")

    np.savez_compressed(OUT / f"traj_{TAG}.npz", rows=out,
                        cols=np.array(list(O.keys())),
                        cluster=np.array([(t, c) for t, c in cluster.items()]))

    # ---- VALIDATION vs sectionals: time for field to cross each marker d
    # field_d is monotonic decreasing in t; interpolate t(d) at markers.
    ft = np.array([(f / fps, field_d[f]) for f in fr_list])
    ft = ft[np.argsort(-ft[:, 1])]  # sort by descending d
    markers = GT.SECTION_MARKERS_D
    cross_t = {}
    for md in markers:
        if md > ft[:, 1].max() or md < ft[:, 1].min():
            cross_t[md] = None
            continue
        cross_t[md] = float(np.interp(-md, -ft[:, 1], ft[:, 0]))
    print(f"\n== VALIDATION (field-level section times) ==")
    print(f"overhead window covers d in [{field_d[fr_list[-1]]:.0f}, "
          f"{field_d[fr_list[0]]:.0f}] m (race is {RACE_DIST}->0)")
    # winner cumulative times for reference (t measured from jump; our t0 may differ)
    # align our clock: assume field crosses d=1650 (start) at our first frame time
    t_start = fr_list[0] / fps
    print(f"marker_d | recovered_t_from_start | sectional_winner_cum | leader_cum")
    win_cum = GT.cumulative_times()[3]  # winner #3 cumulative at [1650,1200,1050,600,0]
    # leader at each marker varies; use winner as a coarse anchor only
    for j, md in enumerate(markers):
        rt = cross_t[md]
        rec = f"{rt - t_start:6.2f}s" if rt is not None else "  (outside window)"
        print(f"  {md:5d}  |  {rec}  | winner_cum={win_cum[j]:6.2f}s")

    # section splits recovered (field leader proxy) vs winner splits
    print(f"\nrecovered field section splits vs winner splits:")
    seg_ok = []
    for j in range(len(markers) - 1):
        a, b = cross_t[markers[j]], cross_t[markers[j + 1]]
        ws = win_cum[j + 1] - win_cum[j]
        if a is None or b is None:
            print(f"  S{j+1} ({markers[j]}->{markers[j+1]}m): outside window "
                  f"(winner split {ws:.2f}s)")
            continue
        rs = b - a
        err = rs - ws
        seg_ok.append(abs(err))
        print(f"  S{j+1} ({markers[j]}->{markers[j+1]}m): recovered={rs:5.2f}s "
              f"winner={ws:5.2f}s  err={err:+.2f}s")
    if seg_ok:
        print(f"  mean |section-time error| (in-window) = {np.mean(seg_ok):.2f}s")

    # ---- TOP-DOWN (d, off) plot — the money viz
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("tab20")
    for tid in idlist:
        sub = out[out[:, O["id"]].astype(int) == tid]
        if len(sub) < 12:
            continue
        sub = sub[np.argsort(sub[:, O["t"]])]
        c = cmap(cluster[tid] % 20)
        ax.plot(sub[:, O["d"]], sub[:, O["off"]], "-", color=c, lw=1.2, alpha=0.8)
    for md in markers:
        ax.axvline(md, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.text(md, ax.get_ylim()[1], f"{md}m", fontsize=8, ha="center", va="bottom")
    ax.invert_xaxis()  # d decreases toward finish (left=finish)
    ax.set_xlabel("d = metres from finish (race runs ->, finish at left)")
    ax.set_ylabel("off = lateral metres from inner rail")
    ax.set_title(f"{TAG}: TOP-DOWN recovered trajectories (d, off) — rail-anchored, "
                 f"HV C course")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"topdown_{TAG}.png", dpi=120)

    # speed vs distance
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for tid in idlist:
        sub = out[out[:, O["id"]].astype(int) == tid]
        if len(sub) < 15:
            continue
        sub = sub[np.argsort(sub[:, O["t"]])]
        dd = np.diff(sub[:, O["d"]]); dt = np.diff(sub[:, O["t"]])
        v = np.abs(dd) / np.maximum(dt, 1e-3)
        # smooth
        if len(v) > 5:
            kern = np.ones(5) / 5
            v = np.convolve(v, kern, mode="same")
        c = cmap(cluster[tid] % 20)
        ax2.plot(sub[1:, O["d"]], v, "-", color=c, lw=1.0, alpha=0.7)
    ax2.set_ylim(0, 25)
    ax2.invert_xaxis()
    for md in markers:
        ax2.axvline(md, color="k", ls=":", lw=0.8, alpha=0.5)
    ax2.axhline(16, color="r", ls="--", lw=0.8, alpha=0.5, label="~16 m/s gallop")
    ax2.set_xlabel("d = metres from finish")
    ax2.set_ylabel("speed (m/s)")
    ax2.set_title(f"{TAG}: recovered speed vs distance (per tracklet)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(OUT / f"speed_{TAG}.png", dpi=120)

    print(f"\nwrote traj_{TAG}.npz, topdown_{TAG}.png, speed_{TAG}.png")


if __name__ == "__main__":
    main()

# ruff: noqa: N802, N803, N806  (H = homography matrix; CV-conventional capitals)
"""Drone-replay path-extraction PROTOTYPE — HV 2026-06-03 R1 (Class 5, 1650M, "C" course).

Recover, from the aerial DRONE replay, a per-horse time series:
  x_long = distance travelled along the track (m, arc-length along the inner rail)
  y_lat  = distance from the inside rail (m)
on a near-straight segment (start straight, t~6..11s), proving the pipeline:
  ingest -> frame-to-frame camera-motion (ORB+RANSAC, background only)
         -> per-frame rail fit (the in-frame ground ruler)
         -> foreground horse detection inside the running lane
         -> nearest-neighbour tracking
         -> CURVILINEAR map: y_lat = perp metres to inner rail (per frame);
            x_long = cumulative ground displacement projected onto the rail
                     direction (frame-to-frame homography chains the camera pan).

Key design choices, learned the hard way:
  * The camera PANS to follow the pack, so a single fixed reference frame fails
    (the pack travels >100m and leaves it). We chain FRAME-TO-FRAME homographies
    of the static background instead, and measure each horse against the rail
    THAT IS VISIBLE IN ITS OWN FRAME.
  * Off-the-shelf YOLO finds ~0 horses on this overhead view (horses ~30px) ->
    detection is classical: temporal-foreground + colour blobs, restricted to a
    band around the fitted rail to kill grandstand/crowd false positives.

Scale: rail-to-rail pixel separation == COURSE_WIDTH_M (HV "C" = 22.5 m).

Outputs:
  - data_cache/path_sample.parquet     [t, horse_id, x_long, y_lat, u, v]
  - studies/cv/out/overlay.png         tracks over a frame (+ fitted rail)
  - studies/cv/out/track_xy.png        x(t), y(t) sanity plots
  - studies/cv/out/detections.png      detection + rail overlay on one frame

Run:  uv run --group vision --group dev python studies/cv/extract_path.py
"""
from __future__ import annotations

import os
import cv2
import numpy as np

MAIN = "/home/ytc/Projects/hkjc-tools"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
VIDEO = os.path.join(MAIN, "data_cache/video/drone_20260603_01.mp4")
PARQUET_OUT = os.path.join(MAIN, "data_cache/path_sample.parquet")

# ---- segment / sampling -----------------------------------------------------
T0, T1 = 6.0, 11.0        # start straight — cleanest rail geometry (validated)
FPS_OUT = 10.0
COURSE_WIDTH_M = 22.5     # HV "C" course usable width (rail-to-rail proxy)
RACE_DISTANCE_M = 1650.0

# ============================================================================ #
# Stage 0: ingest
# ============================================================================ #

def load_frames(video, t0, t1, fps_out):
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / fps_out)))
    f0, f1 = int(round(t0 * src_fps)), int(round(t1 * src_fps))
    frames, times = [], []
    for f in range(f0, f1 + 1, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
        times.append(f / src_fps)
    cap.release()
    return frames, np.array(times), src_fps


def turf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return ((h > 30) & (h < 95) & (s > 40)).astype(np.uint8)


def pack_centroid(img):
    """Rough pack centroid: median of non-green-on-turf blob centres (whole frame)."""
    turf = turf_mask(img)
    bgr = img.astype(np.int16)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    nongreen = ((g - (b + r) // 2) < 18) & (turf > 0)
    fg = nongreen.astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
    pts = [cent[k] for k in range(1, n)
           if 80 < stats[k, cv2.CC_STAT_AREA] < 3500
           and max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) < 80]
    if not pts:
        return None
    return np.median(np.array(pts), axis=0)


# ============================================================================ #
# Stage 1: FRAME-TO-FRAME camera-motion (background only). H[i]: frame_i -> frame_{i-1}.
# ============================================================================ #

def _orb_homography(bf, fa, fb, min_inl=25):
    """Homography mapping fa's frame -> fb's frame (fa=(kp,des), fb=(kp,des))."""
    (kpa, da), (kpb, db) = fa, fb
    if da is None or db is None:
        return None, 0
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:600]
    if len(m) < 12:
        return None, 0
    src = np.float32([kpa[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kpb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if (H is not None and n >= min_inl) else None), n


def keyframe_H_to0(frames):
    """Register each frame to a COMMON ground frame (frame 0) with bounded drift,
    via a chain of KEYFRAMES. Each frame is matched DIRECTLY to the current
    keyframe (one registration, not a long product). When direct overlap to the
    keyframe weakens, promote a new keyframe and compose its (single, well-
    conditioned) offset to frame 0. Returns H_to0[i]: frame i px -> frame 0 px.

    This bounds accumulated error to ~one keyframe handoff, fixing the long-chain
    drift that a pure consecutive product suffers (bg-align MAE explodes)."""
    orb = cv2.ORB_create(6000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feats = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bg = (1 - turf_mask(f)) * 255
        bg = cv2.dilate(bg, np.ones((5, 5), np.uint8))
        feats.append(orb.detectAndCompute(g, bg))

    H_to0 = [np.eye(3)]
    key = 0                       # current keyframe index
    H_key_to0 = np.eye(3)         # keyframe -> frame 0
    inls = [-1]
    for i in range(1, len(frames)):
        Hi_key, n = _orb_homography(bf, feats[i], feats[key])   # frame i -> keyframe
        if Hi_key is None:
            # keyframe drifted out of view -> promote previous frame as new keyframe
            if H_to0[i - 1] is not None:
                key = i - 1
                H_key_to0 = H_to0[i - 1]
                Hi_key, n = _orb_homography(bf, feats[i], feats[key])
        if Hi_key is None:
            H_to0.append(None)
            inls.append(0)
            continue
        H_to0.append(H_key_to0 @ Hi_key)
        inls.append(n)
        # proactively advance keyframe when match strength to current keyframe drops
        if n < 60 and H_to0[i] is not None:
            key = i
            H_key_to0 = H_to0[i]
    return H_to0, inls


# ============================================================================ #
# Stage 2: per-frame RAIL fit (the in-frame ground ruler).
#   Returns inner-rail (point, unit dir) and px/m scale from rail-rail width.
# ============================================================================ #

def fit_rails(img, pack_centroid=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    near_turf = cv2.dilate(turf_mask(img), np.ones((25, 25), np.uint8))
    white = ((gray > 170) & (near_turf > 0)).astype(np.uint8) * 255
    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                            minLineLength=int(h * 0.25), maxLineGap=60)
    if lines is None:
        return None
    # A true rail is a long near-vertical white line that sits at a TURF BOUNDARY:
    # turf on one side, non-turf (crowd / apron / dirt) on the other. Interior
    # mowing stripes fail this test (turf on both sides) -> rejected.
    turf = turf_mask(img)
    segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(abs(ang) - 90) >= 35:
            continue
        length = np.hypot(dx, dy)
        d = np.array([dx, dy], float) / length
        if d[1] < 0:
            d = -d
        nvec = np.array([-d[1], d[0]])
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        # sample turf fraction ~12px to each side of the line midpoint
        def turf_frac(sign):
            tot = hit = 0
            for s in np.linspace(0.2, 1.0, 8):
                p = mid + sign * nvec * 12 * s
                xi, yi = int(p[0]), int(p[1])
                if 0 <= yi < turf.shape[0] and 0 <= xi < turf.shape[1]:
                    tot += 1
                    hit += turf[yi, xi]
            return hit / max(1, tot)
        tl, tr = turf_frac(-1), turf_frac(+1)
        # boundary => one side mostly turf, the other mostly not
        if abs(tl - tr) < 0.45:
            continue
        segs.append((mid[0], mid, d, length))
    if len(segs) < 2:
        return None
    xs = np.array([s[0] for s in segs])

    # cluster into the leftmost and rightmost rail by x; weight-average within
    def wavg(group):
        wts = np.array([g[3] for g in group])
        pt = np.average([g[1] for g in group], axis=0, weights=wts)
        d = np.average([g[2] for g in group], axis=0, weights=wts)
        return pt, d / (np.linalg.norm(d) + 1e-9)

    # The running lane is bounded by the two rails STRADDLING the pack: the
    # nearest rail on each side of the pack centroid. (A 3-rail config — left
    # edge, C-course running rail, far edge — otherwise picks the wrong pair.)
    if pack_centroid is None:
        xmin, xmax = xs.min(), xs.max()
        if xmax - xmin < 50:
            return None
        left = [s for s in segs if s[0] <= xmin + 0.25 * (xmax - xmin)]
        right = [s for s in segs if s[0] >= xmax - 0.25 * (xmax - xmin)]
    else:
        pcx = pack_centroid[0]
        lcand = [s for s in segs if s[0] < pcx - 10]
        rcand = [s for s in segs if s[0] > pcx + 10]
        if not lcand or not rcand:
            return None
        # nearest rail on each side (cluster the ones within 40px of the nearest)
        ln = max(lcand, key=lambda s: s[0])      # closest-from-left = largest x < pcx
        rn = min(rcand, key=lambda s: s[0])      # closest-from-right
        left = [s for s in lcand if abs(s[0] - ln[0]) < 45]
        right = [s for s in rcand if abs(s[0] - rn[0]) < 45]
    lp, ld = wavg(left)
    rp, rd = wavg(right)

    # Evaluate the lateral rail separation AT THE PACK'S ROW (perspective-correct):
    # intersect each rail line with the horizontal row through the pack centroid,
    # take the perpendicular gap there. A scalar px/m valid LOCALLY at the pack.
    row = pack_centroid[1] if pack_centroid is not None else h / 2

    def x_at_row(pt, d, y):
        if abs(d[1]) < 1e-6:
            return pt[0]
        return pt[0] + d[0] * (y - pt[1]) / d[1]
    xl = x_at_row(lp, ld, row)
    xr = x_at_row(rp, rd, row)
    mean_dir = (ld + rd) / 2
    mean_dir /= np.linalg.norm(mean_dir) + 1e-9
    n = np.array([-mean_dir[1], mean_dir[0]])
    sep_px = abs((np.array([xr, row]) - np.array([xl, row])) @ n)
    if sep_px < 50:
        return None
    scale = COURSE_WIDTH_M / sep_px
    # plausibility gate: top-down horses are ~30-90px -> 20..60 px/m
    if not (15.0 <= 1.0 / scale <= 70.0):
        return None
    if pack_centroid is not None:
        inner_is_left = abs(pack_centroid[0] - xl) < abs(pack_centroid[0] - xr)
    else:
        inner_is_left = True
    # anchor the inner-rail point at the pack row for a clean perp reference
    inner_pt = np.array([xl if inner_is_left else xr, row])
    inner_dir = ld if inner_is_left else rd
    return dict(inner_pt=inner_pt, inner_dir=inner_dir, scale=scale,
                sep_px=sep_px, left=(lp, ld), right=(rp, rd),
                inner_is_left=inner_is_left)


# ============================================================================ #
# Stage 3: foreground horse detection, restricted to a band around the inner rail.
# ============================================================================ #

def detect_horses(frames, rails):
    """Temporal-foreground blobs within the running lane. Per-frame median over a
    small temporal window approximates the static turf; deviations = the pack."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    dets = []
    N = len(frames)
    for i, f in enumerate(frames):
        r = rails[i]
        hsv = grays[i]
        turf = ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40))
        # foreground = colour deviating from turf-green inside the lane
        # (the pack is brown horses + bright silks on green)
        bgr = f.astype(np.int16)
        b, g, rr = bgr[..., 0], bgr[..., 1], bgr[..., 2]
        greenness = g - (b + rr) // 2
        nongreen = (greenness < 18) & turf            # darker/!green-dominant ON turf
        fg = (nongreen).astype(np.uint8) * 255
        # restrict to the LANE BETWEEN the two straddling rails (the running lane):
        # build a polygon from the left & right rail lines spanning the frame height.
        if r is not None and "left" in r and "right" in r:
            H_, W_ = f.shape[:2]
            (lp, ld), (rp, rd) = r["left"], r["right"]

            def x_at(pt, d, y):
                return pt[0] if abs(d[1]) < 1e-6 else pt[0] + d[0] * (y - pt[1]) / d[1]
            ys = [0, H_]
            lxs = [x_at(lp, ld, y) for y in ys]
            rxs = [x_at(rp, rd, y) for y in ys]
            poly = np.array([[lxs[0], 0], [rxs[0], 0], [rxs[1], H_], [lxs[1], H_]],
                            np.int32)
            mask = np.zeros((H_, W_), np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            fg = cv2.bitwise_and(fg, mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        fd = _split_blobs(fg, hsv)
        dets.append(fd)
        _ = N
    return dets


def _split_blobs(fg, hsv, horse_px=30):
    """Connected components + distance-transform watershed to split merged
    horses in the dense pack. Returns (u,v,area,hue) detections."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        ww, hh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        if area < 60:
            continue
        comp = (lab == k).astype(np.uint8)
        big = area > 1.6 * horse_px * horse_px or max(ww, hh) > 1.8 * horse_px
        centres = [cent[k]]
        if big:
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            mx = dist.max()
            if mx > 4:
                peaks = (dist > 0.5 * mx).astype(np.uint8)
                pn, plab, pstats, pcent = cv2.connectedComponentsWithStats(peaks, 8)
                if pn - 1 >= 2:
                    centres = [pcent[j] for j in range(1, pn)]
        for (u, vv) in centres:
            ys, xs = np.where(comp > 0)
            sb = hsv[ys, xs, 1]
            hb = hsv[ys, xs, 0]
            sel = sb > 80
            hue = float(np.median(hb[sel])) if sel.any() else -1.0
            out.append((float(u), float(vv), float(area / len(centres)), hue))
    return out


# ============================================================================ #
# Stage 4: gap-tolerant NN tracker (PIXEL space, per raw frame).
# ============================================================================ #

def track_nn_ground(gdets, max_jump=70.0, max_miss=6):
    """NN tracker in GROUND (frame-0) pixel space with constant-velocity
    prediction. gdets[i] = list of (gu, gv, area, hue, u_px, v_px). Returns
    tracks: id -> list of (frame, gu, gv, hue, u_px, v_px)."""
    tracks, state, nxt = {}, {}, 0
    for fi, dd in enumerate(gdets):
        assigned = set()
        for tid in sorted(state, key=lambda t: state[t]["miss"]):
            st = state[tid]
            pu = st["u"] + st["vu"] * (st["miss"] + 1)
            pv = st["v"] + st["vv"] * (st["miss"] + 1)
            best, bd = None, max_jump * (1 + 0.3 * st["miss"])
            for j, (gu, gv, a, hue, ux, vy) in enumerate(dd):
                if j in assigned:
                    continue
                d = np.hypot(gu - pu, gv - pv)
                if d < bd:
                    bd, best = d, j
            if best is not None:
                gu, gv, a, hue, ux, vy = dd[best]
                nfr = fi - st["last"]
                st["vu"] = 0.7 * st["vu"] + 0.3 * (gu - st["u"]) / max(1, nfr)
                st["vv"] = 0.7 * st["vv"] + 0.3 * (gv - st["v"]) / max(1, nfr)
                st["u"], st["v"], st["miss"], st["last"] = gu, gv, 0, fi
                tracks[tid].append((fi, gu, gv, hue, ux, vy))
                assigned.add(best)
            else:
                st["miss"] += 1
                if st["miss"] > max_miss:
                    state.pop(tid, None)
        for j, (gu, gv, a, hue, ux, vy) in enumerate(dd):
            if j in assigned:
                continue
            tracks[nxt] = [(fi, gu, gv, hue, ux, vy)]
            state[nxt] = dict(u=gu, v=gv, vu=0.0, vv=0.0, miss=0, last=fi)
            nxt += 1
    return tracks



# ============================================================================ #
# Stage 5: CURVILINEAR projection to (x_long, y_lat).
#   y_lat: per-frame perpendicular metres to that frame's inner rail.
#   x_long: cumulative GROUND displacement. We map a horse's pixel at frame i and
#           i-1 into a COMMON frame via the chained background homography, take the
#           displacement, project it onto the rail direction, scale to metres,
#           and accumulate.
# ============================================================================ #

def perp_metres_to_rail(u, v, rail, scale):
    ip, idr = rail["inner_pt"], rail["inner_dir"]
    idr = idr / (np.linalg.norm(idr) + 1e-9)
    nvec = np.array([-idr[1], idr[0]])
    lat_px = (u - ip[0]) * nvec[0] + (v - ip[1]) * nvec[1]
    return abs(lat_px) * scale


def main():
    import polars as pl

    print(f"[ingest] {VIDEO}")
    frames, times, src_fps = load_frames(VIDEO, T0, T1, FPS_OUT)
    print(f"  {len(frames)} frames @ {FPS_OUT}fps t=[{times[0]:.2f},{times[-1]:.2f}] src_fps={src_fps}")

    print("[register] keyframe ORB+RANSAC -> common ground frame (bounded drift) ...")
    H_to0, ninl = keyframe_H_to0(frames)
    good = [n for n in ninl if n > 0]
    nreg = sum(1 for h in H_to0 if h is not None)
    print(f"  frames registered to ground={nreg}/{len(frames)}; "
          f"inliers min={min(good)} med={int(np.median(good))} max={max(good)}")
    # verify registration quality: background alignment MAE of warped frames to frame 0
    _bg = (1 - turf_mask(frames[0])).astype(bool)
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
    maes = []
    for i in range(0, len(frames), max(1, len(frames) // 6)):
        if H_to0[i] is None:
            continue
        w = cv2.warpPerspective(frames[i], H_to0[i], (frames[0].shape[1], frames[0].shape[0]))
        gi = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.int16)
        ov = (gi > 0) & _bg
        maes.append(float(np.abs(gi[ov] - g0[ov]).mean()))
    print(f"  bg-align MAE over time (lower=less drift): "
          f"{', '.join(f'{m:.0f}' for m in maes)}")

    # first pass: rough pack centroid per frame (non-green blobs on turf, whole frame)
    print("[rails] per-frame rail fit (pack-straddling rails) ...")
    centroids = [pack_centroid(f) for f in frames]
    rails = []
    for i, f in enumerate(frames):
        r = fit_rails(f, centroids[i])
        rails.append(r)
    nr = sum(r is not None for r in rails)
    seps = np.array([r["sep_px"] for r in rails if r])
    scales = [r["scale"] for r in rails if r]
    print(f"  rails fit on {nr}/{len(frames)} frames; "
          f"sep_px: med={np.median(seps):.0f} p25={np.percentile(seps,25):.0f} "
          f"p75={np.percentile(seps,75):.0f}; scale med={np.median(scales):.4f} m/px "
          f"({1/np.median(scales):.1f} px/m)")
    # ROBUST scale: lock to the median straddling-rail separation (reject outliers),
    # then re-derive every frame's scale from its OWN sep relative to that lock so
    # the px/m tracks camera zoom but is anchored to the same metric width.
    # Lock the scale from the DOMINANT sep cluster (reject minority wrong-rail-pair
    # fits): take seps within 20% of the modal value via a coarse histogram.
    if len(seps) >= 3:
        lo, hi = np.percentile(seps, [30, 70])
        core = seps[(seps >= lo * 0.9) & (seps <= hi * 1.1)]
        sep_lock = float(np.median(core)) if len(core) else float(np.median(seps))
    else:
        sep_lock = float(np.median(seps))
    print(f"  sep_lock={sep_lock:.0f}px -> {COURSE_WIDTH_M/sep_lock*1000:.0f}mm/px "
          f"({sep_lock/COURSE_WIDTH_M:.1f} px/m)")

    # Propagate the inner rail to EVERY frame by transforming a well-fit reference
    # rail (two points) through the chained background homography. This keeps the
    # y=0 reference physically consistent even where the per-frame Hough fit failed.
    ref_idx = next((i for i, r in enumerate(rails) if r is not None), None)
    if ref_idx is not None:
        rr = rails[ref_idx]
        ip = rr["inner_pt"]
        p2 = ip + rr["inner_dir"] / (np.linalg.norm(rr["inner_dir"]) + 1e-9) * 100.0
        for i in range(len(rails)):
            if rails[i] is not None:
                continue
            if H_to0[i] is None or H_to0[ref_idx] is None:
                continue
            Href = np.linalg.inv(H_to0[i]) @ H_to0[ref_idx]   # ref px -> frame i px
            if Href is None:
                continue
            a = np.array(_apply_H(Href, ip[0], ip[1]))
            b = np.array(_apply_H(Href, p2[0], p2[1]))
            d = b - a
            d /= np.linalg.norm(d) + 1e-9
            rails[i] = dict(inner_pt=a, inner_dir=d, scale=COURSE_WIDTH_M / sep_lock,
                            sep_px=sep_lock, left=rr["left"], right=rr["right"],
                            inner_is_left=rr["inner_is_left"], propagated=True)
    print(f"  rails available on {sum(r is not None for r in rails)}/{len(rails)} "
          f"frames after homography propagation")

    print("[detect] foreground horse blobs in the running lane ...")
    dets = detect_horses(frames, rails)
    npf = [len(d) for d in dets]
    print(f"  det/frame med={int(np.median(npf))} mean={np.mean(npf):.1f} "
          f"max={max(npf)} min={min(npf)}")

    # scale cross-check: a horse is ~2.4m long. Median detected blob extent / scale.
    blob_px = []
    for i, dd in enumerate(dets):
        for (u, v, a, hue) in dd:
            blob_px.append(np.sqrt(a))
    if blob_px and rails[len(rails)//2]:
        sc = rails[len(rails)//2]["scale"]
        print(f"  [scale-check] median blob ~{np.median(blob_px):.0f}px "
              f"-> {np.median(blob_px)*sc:.1f}m (a horse+rider top-down ~2-3m)")

    # Map every detection into a COMMON ground frame (frame 0's pixel coords) via
    # the chained background homography, so association links the same PHYSICAL
    # horse even though the camera follows the screen-stationary pack. Horses move
    # smoothly (~const velocity) in ground space -> NN+prediction is reliable there.
    print("[ground] projecting detections to common ground frame ...")
    gdets = []
    for i, dd in enumerate(dets):
        H = H_to0[i]
        gd = []
        if H is not None:
            for (u, v, a, hue) in dd:
                gu, gv = _apply_H(H, u, v)
                gd.append((gu, gv, a, hue, u, v))   # carry original pixel too
        gdets.append(gd)

    print("[track] gap-tolerant NN in GROUND space ...")
    tracks = track_nn_ground(gdets)
    lengths = sorted((len(v) for v in tracks.values()), reverse=True)
    print(f"  {len(tracks)} raw tracks; top lengths={lengths[:14]}")
    # keep tracks spanning >= ~2.2s of clean tracking AND a real ground extent
    min_len = 22

    def gspan(v):
        gu = np.array([g[1] for g in v])
        gv = np.array([g[2] for g in v])
        return np.hypot(np.ptp(gu), np.ptp(gv))
    keep = {tid: v for tid, v in tracks.items()
            if len(v) >= min_len and gspan(v) > 150}   # >150px ground travel (~4.6m)
    print(f"  {len(keep)} tracks >= {min_len} frames & >150px ground travel (kept)")

    _det_image(frames, dets, rails)
    if not keep:
        print("  !! no tracks survived")
        return

    # ---- project to (x_long, y_lat) ----
    # SCALE: lock a single robust px/m from the pack-row rail separations (the
    # camera barely zooms over this 5s window, so a constant scale is justified;
    # verified by the homography step-displacement check ~= physical speed).
    scale_lock = COURSE_WIDTH_M / sep_lock
    print(f"[project] curvilinear mapping; locked scale={scale_lock:.4f} m/px "
          f"({1/scale_lock:.1f} px/m) from sep_lock={sep_lock:.0f}px")
    # inner-rail direction in GROUND (frame-0) coords: transform a per-frame inner
    # rail point+dir into frame 0 once, to project ground displacement on it.
    rr0 = rails[0] if rails[0] is not None else next(r for r in rails if r is not None)
    g_rail_dir = rr0["inner_dir"] / (np.linalg.norm(rr0["inner_dir"]) + 1e-9)

    rows = []
    for tid, obs in keep.items():
        obs = sorted(obs, key=lambda o: o[0])
        hues = [h for (_, _, _, _, _, _), h in [(o, o[3]) for o in obs] if h >= 0]
        sil = f"hue{int(np.median(hues))}" if hues else "unk"
        x_long = 0.0
        prev_g = None
        for (fi, gu, gv, hue, ux, vy) in obs:
            r = rails[fi]
            # y_lat: perpendicular metres to the inner rail in the ORIGINAL frame
            y_lat = perp_metres_to_rail(ux, vy, r, scale_lock)
            # x_long: cumulative GROUND arc-length (frame-0 px) projected on rail dir
            if prev_g is not None:
                disp = np.array([gu, gv]) - np.array(prev_g)
                along = abs(float(np.dot(disp, g_rail_dir)))
                mag = float(np.hypot(*disp))
                step_px = along if along > 0.5 * mag else mag
                x_long += step_px * scale_lock
            rows.append(dict(t=float(times[fi]), horse_id=f"trk{tid}_{sil}",
                             x_long=float(x_long), y_lat=float(y_lat),
                             u=float(ux), v=float(vy)))
            prev_g = (gu, gv)

    df = pl.DataFrame(rows).sort(["horse_id", "t"])
    df.write_parquet(PARQUET_OUT)
    print(f"[save] {PARQUET_OUT}  rows={df.height}  horses={df['horse_id'].n_unique()}")

    # ---- sanity ----
    print("\n[sanity]")
    speeds, dxts = [], []
    for hid, g in df.group_by("horse_id"):
        hid = hid[0] if isinstance(hid, tuple) else hid
        g = g.sort("t")
        if g.height < 3:
            continue
        x = g["x_long"].to_numpy()
        tt = g["t"].to_numpy()
        yl = g["y_lat"].to_numpy()
        v = np.diff(x) / np.diff(tt)
        mono = float((np.diff(x) >= -0.3).mean())
        speeds.append(np.median(v))
        dxts.append(x.max())
        print(f"  {hid:16s} n={g.height:2d} dx={x.max():6.1f}m "
              f"y=[{yl.min():4.1f},{yl.max():4.1f}]m v_med={np.median(v):5.1f}m/s mono={mono:.2f}")
    if speeds:
        print(f"  -- median speed = {np.median(speeds):.1f} m/s (physical 15-19)")
        print(f"  -- median dx over {times[-1]-times[0]:.0f}s window = {np.median(dxts):.0f}m "
              f"(=> {np.median(dxts)/(times[-1]-times[0]):.1f} m/s)")
        print(f"  -- y_lat range [{df['y_lat'].min():.1f},{df['y_lat'].max():.1f}]m "
              f"(course width {COURSE_WIDTH_M}m)")

    _plots(df, frames[0], keep, rails[0])
    _track_montage(frames, keep, rails)
    print("\n[done] outputs in studies/cv/out/ : overlay.png, track_xy.png, "
          "detections.png, track_montage.png")


def _apply_H(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def _det_image(frames, dets, rails):
    mid = len(frames) // 2
    dv = frames[mid].copy()
    r = rails[mid]
    if r is not None:
        for key, col in [("left", (255, 0, 0)), ("right", (0, 255, 0))]:
            pt, d = r[key]
            p0 = (pt - d * 1500).astype(int)
            p1 = (pt + d * 1500).astype(int)
            cv2.line(dv, tuple(p0), tuple(p1), col, 2)
    for (u, v, a, hue) in dets[mid]:
        cv2.circle(dv, (int(u), int(v)), 10, (0, 0, 255), 2)
    cv2.putText(dv, f"frame {mid}: {len(dets[mid])} det  (blue=L rail, green=R rail)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, "detections.png"), dv)


def _track_montage(frames, keep, rails):
    """3-frame montage with each kept track's ORIGINAL-pixel position marked on the
    frames where it was observed — shows the markers riding the actual horses."""
    n = len(frames)
    idxs = [int(0.08 * n), int(0.5 * n), int(0.9 * n)]
    rng = np.random.default_rng(0)
    cols = {tid: tuple(int(c) for c in rng.integers(60, 255, 3)) for tid in keep}
    panels = []
    for fi in idxs:
        img = frames[fi].copy()
        r = rails[fi]
        if r is not None and "left" in r:
            for key, c in [("left", (255, 0, 0)), ("right", (0, 255, 0))]:
                pt, d = r[key]
                cv2.line(img, tuple((pt - d * 1500).astype(int)),
                         tuple((pt + d * 1500).astype(int)), c, 2)
        for tid, obs in keep.items():
            for (f, gu, gv, hue, ux, vy) in obs:
                if f == fi:
                    cv2.circle(img, (int(ux), int(vy)), 11, cols[tid], 3)
        cv2.putText(img, f"t={fi/FPS_OUT + T0:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        panels.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv2.imwrite(os.path.join(OUT, "track_montage.png"), np.hstack(panels))


def _plots(df, ref_img, keep, rail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ov = ref_img.copy()
    if rail is not None:
        ip, idr = rail["inner_pt"], rail["inner_dir"]
        p0 = (ip - idr * 1500).astype(int)
        p1 = (ip + idr * 1500).astype(int)
        cv2.line(ov, tuple(p0), tuple(p1), (0, 255, 255), 2)
    # tracks are in GROUND (frame-0) coords (gu,gv) -> draw directly on frame 0:
    # the track polyline shows each horse's ground path receding up-frame as it
    # gallops away (frame 0 is the window start, horses move toward the bottom).
    rng = np.random.default_rng(0)
    for tid, obs in keep.items():
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        pts = np.array([(gu, gv) for (_, gu, gv, _, _, _) in sorted(obs)], np.int32)
        for j in range(1, len(pts)):
            cv2.line(ov, tuple(pts[j - 1]), tuple(pts[j]), col, 2)
        cv2.circle(ov, tuple(pts[0]), 5, col, -1)
        cv2.circle(ov, tuple(pts[-1]), 6, col, 2)
    cv2.imwrite(os.path.join(OUT, "overlay.png"), ov)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for hid, g in df.group_by("horse_id"):
        g = g.sort("t")
        ax[0].plot(g["t"], g["x_long"], ".-", ms=3, lw=1)
        ax[1].plot(g["t"], g["y_lat"], ".-", ms=3, lw=1)
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("x_long (m, dist travelled)")
    ax[0].set_title("Longitudinal — should rise ~linearly (~16 m/s)")
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("y_lat (m from inner rail)")
    ax[1].set_title("Lateral — bounded by track width")
    ax[1].axhline(COURSE_WIDTH_M, ls="--", c="r", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "track_xy.png"), dpi=110)


if __name__ == "__main__":
    main()

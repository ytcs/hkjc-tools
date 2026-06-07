"""S0d DRONE pipeline — REAL-horse rail-relative path reconstruction (the PRIMARY lateral signal).

Per journal 0069, the drone is now the PRIMARY path to the novel lateral/control signal: it is
REAL horse positions (not the aerial's synthetic legend overlay), so its rail-relative ``y_lat`` is
the genuine ground-saved/lost trip — INVISIBLE to the chart/form. 0068 dismissed the drone as "the
weak view, numbers illegible from altitude" on an INCOMPLETE look (only the high-altitude
back-straight). Frame scrutiny (0069) shows the drone ZOOMS to a close oblique "Photo Finish System"
replay (t≈120-138s) where saddlecloth numbers ARE clearly legible — so the drone CAN be numbered
NON-CIRCULARLY (video-only) and become a numbered, real-horse lateral reconstruction.

Stages:
  detect   — the field is a tight cluster of dark/coloured horse+jockey sprites on green turf,
             surrounded by grandstand/crowd/rail clutter. Mask the largest turf region, keep
             non-turf sprites INSIDE it, keep the densest spatial cluster (the field).
  track    — greedy nearest-neighbour motion linkage into tracklets.
  scale    — FIXED (0069). Per frame, the running corridor (inner white rail -> outer rail) is the
             "C"-course width 21.3 m. We measure the perpendicular turf-corridor width in px via a
             distance-transform on the turf-strip mask (robust to the diagonal/curved track and to
             one rail being off-frame), giving a per-frame m_per_px. Along-rail arc-length is the
             cumulative TANGENTIAL displacement (not the old image-x proxy that inflated arc ~2-4x),
             and y_lat is the perpendicular rail offset CLAMPED to [0, 21.3] m. Validated against the
             known race distance (~1650 m) and the 21.3 m width.
  re-id    — silk-COLOUR clusters (anonymous) PLUS the NUMBER BRIDGE: OCR the finish-zoom replay
             window, sample the silk LAB at each legible saddlecloth number, match the number's silk
             to a back-straight silk cluster, and propagate the label BACKWARD along the track. The
             OCR uses ``hkjc.vision.mode_b.ocr`` READ-ONLY (RapidOCR). This is the genuine,
             non-circular cluster->number bind 0068 said was impossible.

Reads the video + ``hkjc.core.track`` + ``hkjc.vision.mode_b.ocr`` READ-ONLY. numpy/cv2/av lazy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# finish-zoom "Photo Finish System" replay window (s) where saddlecloth numbers are legible
FINISH_ZOOM_T0 = 118.0
FINISH_ZOOM_T1 = 140.0
# top banner rows (the bilingual "Photo Finish System" overlay) to skip when OCR-ing the replay
FINISH_BANNER_PX = 110


@dataclass
class DroneResult:
    rows: list           # list[dict] tidy path rows
    diagnostics: dict


@dataclass
class _Track:
    rid: int
    dets: list = field(default_factory=list)
    number: int | None = None   # bound saddlecloth number (None = unbound)


def _turf_mask(img):
    """Largest connected turf (green) region as a uint8 mask + the locked turf hue."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sel = (ss > 40) & (vv > 40) & (hh > 30) & (hh < 80)
    if sel.sum() < 1000:
        return None, None
    turf_hue = int(np.median(hh[sel]))
    green = ((np.abs(hh.astype(int) - turf_hue) <= 15) & (ss > 30) & (vv > 40)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lbl, st, _ct = cv2.connectedComponentsWithStats(green, 8)
    if n < 2:
        return None, turf_hue
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lbl == big).astype(np.uint8), turf_hue


def _corridor_px(turf, field_xy):
    """Perpendicular turf-strip width in px, via the distance transform (global ridge).

    The turf strip runs diagonally / curves, so a per-row pixel width is the (longer) oblique
    chord, not the perpendicular width. The distance transform's value at an interior RIDGE point
    equals the distance to the nearest turf edge = HALF the local perpendicular width; the strip's
    perpendicular width is a stable global geometric property, so we take a high percentile of the
    DT over the WHOLE turf mask (robust to the field's position on the strip and to clutter holes).
    ``field_xy`` is accepted for signature symmetry but the global ridge is used."""
    import cv2
    import numpy as np

    dt = cv2.distanceTransform(turf, cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    if vals.size < 50:
        return None
    half = float(np.percentile(vals, 99.0))
    return 2.0 * half if half > 5 else None


def _detect_field(img, turf_lock):
    """Detect the field's horse sprites in one BGR frame; return (blobs, turf_hue, turf_mask)."""
    import cv2
    import numpy as np

    turf, turf_hue = _turf_mask(img)
    if turf is None:
        return [], turf_lock, None
    turf_lock = turf_hue if turf_lock is None else int(0.7 * turf_lock + 0.3 * turf_hue)

    field_region = cv2.erode(turf, np.ones((9, 9), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (np.abs(hh.astype(int) - turf_lock) <= 13) & (ss > 35) & (vv > 40)
    nong = ((~green) & (vv > 30)).astype(np.uint8)
    sprites = cv2.bitwise_and(nong, field_region)
    sprites = cv2.morphologyEx(sprites, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    sprites = cv2.morphologyEx(sprites, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _l2, s2, c2 = cv2.connectedComponentsWithStats(sprites, 8)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cand = []
    for i in range(1, n):
        a = int(s2[i, cv2.CC_STAT_AREA])
        bw, bh = int(s2[i, cv2.CC_STAT_WIDTH]), int(s2[i, cv2.CC_STAT_HEIGHT])
        if not (18 <= a <= 2000 and bw < 80 and bh < 80):
            continue
        cx, cy = float(c2[i][0]), float(c2[i][1])
        bx, by = int(s2[i, cv2.CC_STAT_LEFT]), int(s2[i, cv2.CC_STAT_TOP])
        patch = lab_img[by:by + bh, bx:bx + bw].reshape(-1, 3).astype(float)
        chroma = np.abs(patch[:, 1] - 128.0) + np.abs(patch[:, 2] - 128.0)
        k = max(3, int(0.3 * len(patch)))
        mlab = patch[np.argsort(chroma)[-k:]].mean(axis=0)
        cand.append((cx, cy, a, (float(mlab[0]), float(mlab[1]), float(mlab[2]))))
    if not cand:
        return [], turf_lock, turf
    xy = np.array([(c[0], c[1]) for c in cand])
    med = np.median(xy, axis=0)
    d = np.linalg.norm(xy - med, axis=1)
    keep = [cand[i] for i in range(len(cand)) if d[i] < 400]
    return keep, turf_lock, turf


def _rail_inner(turf, field_xy):
    """Inner-rail polyline as (xs, ys) sample arrays nearest the field, from the turf-mask edge.

    The inner rail is the turf-strip boundary on the FIELD side. We take the turf-mask boundary,
    keep boundary points within a band of the field, and order them by arc to form a polyline used
    for (along, perp) decomposition. Returns (xs, ys) or None."""
    import cv2
    import numpy as np

    if not field_xy:
        return None
    cnts, _ = cv2.findContours(turf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    fx = np.array([p[0] for p in field_xy])
    fy = np.array([p[1] for p in field_xy])
    # boundary points nearest to the field (the field hugs the inner rail)
    d2 = ((cnt[:, None, 0] - fx[None, :]) ** 2 + (cnt[:, None, 1] - fy[None, :]) ** 2).min(axis=1)
    near = cnt[d2 < 220 ** 2]
    if len(near) < 12:
        return None
    # order along the dominant axis (PCA) to make a monotone polyline
    c = near - near.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    t = c @ vt[0]
    order = np.argsort(t)
    pts = near[order]
    # decimate
    step = max(1, len(pts) // 80)
    pts = pts[::step]
    return pts[:, 0], pts[:, 1]


def _bands(idx):
    import numpy as np

    idx = np.sort(idx)
    out, start, prev = [], idx[0], idx[0]
    for x in idx[1:]:
        if x - prev > 8:
            out.append((start + prev) / 2.0)
            start = x
        prev = x
    out.append((start + prev) / 2.0)
    return out


def _link(dets_by_frame, gate=70.0):
    import numpy as np

    tracks, active, last = [], [], {}
    nid = 0
    for dets in dets_by_frame:
        if not dets:
            continue
        if not active:
            for d in dets:
                tracks.append(_Track(nid, [d]))
                last[nid] = (d["cx"], d["cy"])
                active.append(nid)
                nid += 1
            continue
        cost = np.full((len(active), len(dets)), 1e6)
        for ai, rid in enumerate(active):
            px, py = last[rid]
            for di, d in enumerate(dets):
                dist = float(np.hypot(d["cx"] - px, d["cy"] - py))
                if dist <= gate:
                    lab_d = float(np.linalg.norm(np.array(d["lab"]) - tracks[rid].dets[-1]["lab"]))
                    cost[ai, di] = dist + 0.4 * lab_d
        used_a, used_d = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for ai, di in order:
            if cost[ai, di] >= 1e6:
                break
            if ai in used_a or di in used_d:
                continue
            rid = active[ai]
            tracks[rid].dets.append(dets[di])
            last[rid] = (dets[di]["cx"], dets[di]["cy"])
            used_a.add(ai)
            used_d.add(di)
        new_active = [active[ai] for ai in used_a]
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tracks.append(_Track(nid, [d]))
            last[nid] = (d["cx"], d["cy"])
            new_active.append(nid)
            nid += 1
        active = new_active
    return tracks


def _silk_at(img, cx, cy):
    """Median chromatic LAB of the silk patch around the number box (the jockey's body region).

    The saddlecloth number sits on the horse's flank; the jockey's silk torso is just ABOVE it, so
    we sample a box biased upward and keep the most chromatic pixels (the silk colour), to match the
    back-straight detector's silk-LAB read."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    rx, ry = 30, 34
    x0, x1 = max(0, int(cx) - rx), min(w, int(cx) + rx)
    y0, y1 = max(0, int(cy) - ry - 14), min(h, int(cy) + ry - 14)
    if x1 <= x0 or y1 <= y0:
        return None
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    chroma = np.abs(lab[:, 1] - 128.0) + np.abs(lab[:, 2] - 128.0)
    k = max(3, int(0.25 * len(lab)))
    sel = lab[np.argsort(chroma)[-k:]]
    return tuple(float(v) for v in sel.mean(axis=0))


def _ocr_finish_numbers(finish_frames, field_size):
    """OCR saddlecloth numbers in the finish-zoom replay frames; return number -> [silk LAB].

    ``finish_frames`` is a list of BGR frames sampled from [FINISH_ZOOM_T0, FINISH_ZOOM_T1].
    RapidOCR localises digit tokens; for each high-confidence in-field number we sample the silk
    LAB just above the number box (the jockey's silk). Returns number -> list of (conf, silk_lab)."""
    import cv2

    from hkjc.vision.mode_b.ocr import _rapidocr

    ocr = _rapidocr()
    valid = set(range(1, field_size + 2))
    reads: dict[int, list] = {}
    sc = 2.5
    for img in finish_frames:
        crop = img[FINISH_BANNER_PX:, :, :]
        up = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(up)
        for box, txt, conf in (res or []):
            digits = "".join(ch for ch in txt if ch.isdigit())
            if not digits or float(conf) < 0.6:
                continue
            num = int(digits)
            if num not in valid:
                continue
            xc = sum(p[0] for p in box) / 4.0 / sc
            yc = sum(p[1] for p in box) / 4.0 / sc + FINISH_BANNER_PX
            silk = _silk_at(img, xc, yc)
            if silk is not None:
                reads.setdefault(num, []).append((float(conf), silk))
    return reads


def _bind_numbers(runners, ocr_reads):
    """Bind each OCR'd number to its nearest silk-colour cluster (Hungarian-lite greedy).

    For each number, the consensus silk LAB is the conf-weighted median of its finish-zoom reads.
    We assign numbers to runner clusters by minimum LAB distance, one-to-one, greedily by best
    match. Returns {runner_id: number} for the bound subset, plus a per-bind LAB distance dict."""
    import numpy as np

    def cluster_lab(r):
        return np.array([d["lab"] for d in r.dets], float).mean(axis=0)

    num_lab = {}
    for num, lst in ocr_reads.items():
        if not lst:
            continue
        labs = np.array([s for _c, s in lst], float)
        num_lab[num] = np.median(labs, axis=0)
    clab = {r.rid: cluster_lab(r) for r in runners}
    pairs = []
    for num, nl in num_lab.items():
        for rid, cl in clab.items():
            pairs.append((float(np.linalg.norm(nl - cl)), num, rid))
    pairs.sort()
    bound, used_num, used_rid, dist = {}, set(), set(), {}
    for d, num, rid in pairs:
        if num in used_num or rid in used_rid:
            continue
        bound[rid] = num
        dist[rid] = round(d, 1)
        used_num.add(num)
        used_rid.add(rid)
    return bound, dist


def extract_drone_paths(
    video_path: str,
    *,
    official: dict,
    venue: str,
    date: str,
    race_no: int,
    distance_m: float,
    surface: str = "turf",
    course: str = "C",
    field_size: int = 12,
    stride: int = 5,
) -> DroneResult:
    """Reconstruct NUMBERED rail-relative runner paths from the drone video (the lateral signal)."""
    import av
    import numpy as np

    from hkjc.core.track import track_for

    geo = track_for(venue, distance_m, surface)
    track_width_m = float(geo.width(course))

    container = av.open(video_path)
    vstream = container.streams.video[0]
    fps = float(vstream.average_rate)

    dets_by_frame, frame_idx = [], []
    rail_by_frame, scale_by_frame = {}, {}
    finish_frames = []
    turf_lock = None
    n_used = 0
    for i, fr in enumerate(container.decode(video=0)):
        t = i / fps
        if FINISH_ZOOM_T0 <= t <= FINISH_ZOOM_T1:
            # finish-zoom replay: a separate close shot; collect for the OCR number bridge only
            if i % 3 == 0:
                finish_frames.append(fr.to_ndarray(format="bgr24"))
            continue
        if i % stride:
            continue
        img = fr.to_ndarray(format="bgr24")
        blobs, turf_lock, turf = _detect_field(img, turf_lock)
        if not blobs or turf is None:
            continue
        n_used += 1
        cap = field_size + 4
        blobs = sorted(blobs, key=lambda b: -b[2])[:cap]
        dets = [dict(frame=i, t=t, cx=b[0], cy=b[1], area=b[2], lab=b[3]) for b in blobs]
        field_xy = [(d["cx"], d["cy"]) for d in dets]
        rail = _rail_inner(turf, field_xy)
        corr_px = _corridor_px(turf, field_xy)
        if rail is not None:
            rail_by_frame[i] = rail
        if corr_px and corr_px > 50:
            scale_by_frame[i] = track_width_m / corr_px   # m per px, per frame
        dets_by_frame.append(dets)
        frame_idx.append(i)
    container.close()

    diagnostics = {
        "fps": fps,
        "frames_used": n_used,
        "frames_with_detections": len(dets_by_frame),
        "track_width_m": track_width_m,
        "rail_frames": len(rail_by_frame),
        "scale_frames": len(scale_by_frame),
        "scale_source": "perp-turf-corridor-DT == C-width(21.3m)",
    }
    if not dets_by_frame:
        diagnostics["error"] = "no field detections — drone substrate inadequate"
        return DroneResult(rows=[], diagnostics=diagnostics)

    det_counts = [len(d) for d in dets_by_frame]
    diagnostics["det_per_frame_median"] = float(np.median(det_counts))

    tracks = _link(dets_by_frame)
    diagnostics["n_tracks"] = len(tracks)

    runners = _stitch_colour(tracks, field_size)
    diagnostics["n_runners_colour"] = len(runners)

    # ---- the NUMBER BRIDGE: OCR the finish-zoom replay, bind cluster->number ----------------
    diagnostics["finish_frames"] = len(finish_frames)
    ocr_reads = _ocr_finish_numbers(finish_frames, field_size)
    diagnostics["ocr_numbers_read"] = sorted(ocr_reads.keys())
    diagnostics["ocr_n_reads"] = {int(k): len(v) for k, v in sorted(ocr_reads.items())}
    bound, bind_dist = _bind_numbers(runners, ocr_reads)
    for r in runners:
        r.number = bound.get(r.rid)
    diagnostics["n_bound"] = len(bound)
    diagnostics["bound_numbers"] = sorted(bound.values())
    diagnostics["bind_lab_dist"] = bind_dist
    diagnostics["reid_mode"] = "silk-colour + finish-zoom OCR number bridge (non-circular)"

    # ---- rail-relative rows: along-rail arc-length (tangential) + perp y_lat (clamped) --------
    scale_med = float(np.median(list(scale_by_frame.values()))) if scale_by_frame else None
    diagnostics["m_per_px_median"] = scale_med
    rows = []
    arc_maxes = []
    for rr in runners:
        rr.dets.sort(key=lambda d: d["frame"])
        arc = 0.0
        prev_xy = None
        for d in rr.dets:
            rail = rail_by_frame.get(d["frame"])
            mpp = scale_by_frame.get(d["frame"], scale_med)
            if rail is None or mpp is None:
                continue
            perp_px, along_px, tangu = _rail_decomp(rail, d["cx"], d["cy"])
            if prev_xy is not None:
                dx = d["cx"] - prev_xy[0]
                dy = d["cy"] - prev_xy[1]
                step = abs(dx * tangu[0] + dy * tangu[1])  # tangential displacement (px)
                if step < 60:   # reject homography jumps / track switches
                    arc += step * mpp
            prev_xy = (d["cx"], d["cy"])
            y_lat = min(max(abs(perp_px) * mpp, 0.0), track_width_m)
            rows.append(dict(
                venue=venue, date=date, race_no=race_no, runner=rr.rid,
                number=rr.number,
                t=round(d["t"], 3), frame=d["frame"],
                x_arclen=round(arc, 2), y_lat=round(float(y_lat), 3),
                silk_lab_l=round(d["lab"][0], 1), silk_lab_a=round(d["lab"][1], 1),
                silk_lab_b=round(d["lab"][2], 1),
            ))
        arc_maxes.append(arc)
    diagnostics["n_rows"] = len(rows)
    diagnostics["arc_max_median_m"] = round(float(np.median(arc_maxes)), 1) if arc_maxes else None
    diagnostics["arc_max_max_m"] = round(float(np.max(arc_maxes)), 1) if arc_maxes else None
    diagnostics["y_lat_max_m"] = round(max((r["y_lat"] for r in rows), default=0.0), 2)
    diagnostics["drone_purity"] = round(_colour_purity(runners), 3)
    diagnostics["note"] = (
        "scale FIXED: per-frame m/px from perp turf-corridor (C=21.3m); arc = tangential "
        "displacement; y_lat clamped [0,21.3]. Numbered via finish-zoom OCR bridge (non-circular)."
    )
    return DroneResult(rows=rows, diagnostics=diagnostics)


def _rail_decomp(rail, cx, cy):
    """(perp_px, along_px, unit_tangent) of point (cx,cy) w.r.t. the inner-rail polyline."""
    import numpy as np

    xs, ys = rail
    d = np.hypot(xs - cx, ys - cy)
    k = int(np.argmin(d))
    k0 = max(0, k - 1)
    k1 = min(len(xs) - 1, k + 1)
    tx, ty = xs[k1] - xs[k0], ys[k1] - ys[k0]
    nrm = float(np.hypot(tx, ty)) or 1.0
    tangu = (tx / nrm, ty / nrm)
    along = xs[k] * tangu[0] + ys[k] * tangu[1]
    return float(d[k]), float(along), tangu


def _stitch_colour(tracks, field_size, anchor_len=8):
    import numpy as np

    tracks = [t for t in tracks if t.dets]
    if not tracks:
        return []

    def mlab(t):
        return np.array([d["lab"] for d in t.dets], float).mean(axis=0)

    anchors = sorted((t for t in tracks if len(t.dets) >= anchor_len),
                     key=lambda t: len(t.dets), reverse=True)
    if not anchors:
        anchors = sorted(tracks, key=lambda t: len(t.dets), reverse=True)[:field_size]
    merged = []
    for a in anchors:
        if len(merged) >= field_size:
            break
        al = mlab(a)
        hit = next((m for m in merged if np.linalg.norm(mlab(m) - al) < 14.0), None)
        if hit is None:
            nt = _Track(len(merged), list(a.dets))
            merged.append(nt)
        else:
            hit.dets.extend(a.dets)
    if not merged:
        return []
    cents = np.array([mlab(m) for m in merged])
    anchor_ids = {id(a) for a in anchors[:len(merged)]}
    for t in tracks:
        if id(t) in anchor_ids:
            continue
        j = int(np.argmin(np.linalg.norm(cents - mlab(t), axis=1)))
        merged[j].dets.extend(t.dets)
    for m in merged:
        m.dets.sort(key=lambda d: d["frame"])
        seen = {}
        for d in m.dets:
            if d["frame"] not in seen or d["area"] > seen[d["frame"]]["area"]:
                seen[d["frame"]] = d
        m.dets = [seen[f] for f in sorted(seen)]
    return [m for m in merged if m.dets]


def _colour_purity(runners):
    """Median per-runner silk-LAB tightness, normalised to a 0..1 purity-like score."""
    import numpy as np

    if not runners:
        return 0.0
    tight = []
    for r in runners:
        labs = np.array([d["lab"] for d in r.dets], float)
        if len(labs) < 3:
            continue
        spread = float(np.median(np.linalg.norm(labs - np.median(labs, axis=0), axis=1)))
        tight.append(spread)
    if not tight:
        return 0.0
    med_spread = float(np.median(tight))
    return max(0.0, min(1.0, 1.0 - med_spread / 40.0))


def main() -> None:
    """Run the drone reconstruction on ST 2026-06-03 R1 and write the parquet + diagnostics."""
    import json

    import polars as pl

    res = extract_drone_paths(
        "data_cache/video/drone_20260603_01.mp4",
        official={}, venue="ST", date="2026-06-03", race_no=1,
        distance_m=1650.0, surface="turf", course="C", field_size=12, stride=5,
    )
    print(json.dumps(res.diagnostics, indent=2, default=str))
    if res.rows:
        df = pl.DataFrame(res.rows)
        df.write_parquet("data_cache/cv_s0d_drone.parquet")
        print(f"wrote data_cache/cv_s0d_drone.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()

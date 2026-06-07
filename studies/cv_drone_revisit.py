"""DRONE REVISIT (#87) — fold the drone in as the 3rd lateral feed with the NEW toolkit.

Journal 0070 (#79) NULLED the drone lateral, but with the OLD per-frame-rail approach
(``_rail_inner`` re-fits the turf-mask boundary EVERY frame -> the y_lat of a single horse swings
0.8->6.3m across clean frames; #12 the wire-to-wire RAIL leader read WIDE). Since then two things
have worked on the OTHER feeds:
  * 0075 (aerial #82): a STABLE RAIL SPLINE (NOT per-frame refit) + GROUND-CONTACT (shadow base, not
    body centroid). This made the aerial render between-runner lateral FAITHFUL (rp +0.45, ac1 +0.90,
    stewards-comment match).
  * 0076 (patrol #83): a CONSTANT homography from a fixed mount.

The drone is a REAL moving high camera (pan/zoom/cut). It has NO global longitudinal axis, but it
DOES have clean near-static STRAIGHT windows where a per-WINDOW stable rail is a good analog of the
0075 stable spline: fit ONE rail line for the window (not per frame), measure each horse's
ground-contact perpendicular offset to it, scale by the 21.3m "C" width. This is the 0075 recipe
ported to the panning camera.

THREE STEPS (the task):
  1. COVERAGE CENSUS on the SHARED CLOCK. Anchor drone_vid_t -> race_t by two real-time events
     (the gate jump ~t=3.0s -> race_t 0; the run-in finish-cut ~t=113.7s -> winner crosses
     race_t 98.85s) -> an affine map; then race_t -> arclen via the official-section curve (same
     curve the fusion spine uses). Classify each window STRAIGHT vs BEND by the in-frame rail
     curvature. Report which arclen band(s) the drone sees cleanly and whether it OVERLAPS the
     aerial's clean body/home-straight and/or the run-in.
  2. VALIDATED LATERAL on a clean STRAIGHT window, NEW toolkit (stable per-window rail + ground
     contact). Identity via the 0070 silk->number bridge (reuse the existing drone panel's numbered
     tracks for identity colour anchors). Within/between dispersion ratio per clean zone -- does the
     new toolkit beat the 0070 ratio>1 null?
  3. BACK-STRAIGHT / RUN-IN REDUNDANT CROSS-CHECK vs the aerial render. On the STRAIGHT where the
     drone overlaps the aerial, run the per-horse lateral AGREEMENT test (Spearman of the two feeds'
     tight<->wide ordering + rail/wide class-agree) the 0077 bend overlap could not deliver.

Official data (finish order, sections, stewards comments) is VALIDATION-ONLY, never a model input.
Study-only; src/hkjc/model|core|vision/mode_b NOT touched; the existing drone/aerial panels are read
read-only; new parquet written to data_cache/cv_drone_numbered.parquet.
"""
from __future__ import annotations

import json

import av
import cv2
import numpy as np
import polars as pl

DRONE_VIDEO = "data_cache/video/drone_20260603_01.mp4"
DRONE_PANEL = "data_cache/cv_s0d_drone.parquet"          # read-only (identity colour anchors)
AERIAL_PANEL = "data_cache/cv_s0d_aerial_render.parquet"  # read-only (#82/0075 lateral)
OUT_PANEL = "data_cache/cv_drone_numbered.parquet"
FIG = "data_cache/figs/cv_drone_revisit_qc.png"

DIST = 1650.0
TRACK_WIDTH_M = 21.3  # "C" course width (validated 0068/0070)
FPS = 25.0

# Shared-clock anchors (real-time drone events, frame-verified above this study).
DRONE_START_VID_T = 3.0     # gate jump (centroid speed surges 2->12 px/frame ~t=2.7-3.6s)
DRONE_FINISHCUT_VID_T = 113.68  # hard CUT to the photo-finish replay (centroid jump 350px); winner
#                                 # crosses the line just before the cut.
WINNER_FINISH_RACE_T = 98.85  # official winner 1:38.85

# Official sections (results HTML) -> arclen->race_t curve (same as the fusion spine, 0077).
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]

# Stewards' Comments-on-Running (VALIDATION ONLY) — the external lateral ground-truth (0075/0076).
RAIL_HORSES = {12, 8, 4, 11}   # 12 wire-to-wire rail; 8 improved inside
WIDE_HORSES = {2, 6, 9, 1}     # 2 wide barrier; 6 caught wide; 9 bumped; 1 wide
# Official running positions (mean lower == nearer lead/rail trip) — VALIDATION ONLY.
OFFICIAL_RP = {
    3: [7, 7, 6, 1], 12: [1, 1, 1, 2], 6: [11, 11, 10, 3], 2: [9, 9, 9, 4],
    9: [4, 4, 3, 5], 8: [6, 2, 2, 6], 11: [8, 8, 8, 7], 4: [5, 6, 5, 8],
    7: [12, 12, 11, 9], 10: [3, 3, 4, 10], 5: [2, 5, 7, 11], 1: [10, 10, 12, 12],
}


# --------------------------------------------------------------------------------------------
# SHARED CLOCK
# --------------------------------------------------------------------------------------------
def drone_vid_to_race_t(vid_t):
    """Affine map drone video-time -> race_t from the two frame-verified anchors."""
    v0, v1 = DRONE_START_VID_T, DRONE_FINISHCUT_VID_T
    r0, r1 = 0.0, WINNER_FINISH_RACE_T
    return r0 + (np.asarray(vid_t, float) - v0) * (r1 - r0) / (v1 - v0)


def _scalar_race_t(vid_t: float) -> float:
    return float(drone_vid_to_race_t(np.asarray(vid_t, float)))


def _scalar_arclen(race_t: float) -> float:
    return float(race_t_to_arclen(np.asarray(race_t, float)))


def arclen_to_race_t(arclen_m: np.ndarray) -> np.ndarray:
    d_bounds = np.array([0.0, 450.0, 850.0, 1250.0, 1650.0])
    t_cum = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    return np.interp(np.asarray(arclen_m, float), d_bounds, t_cum)


def race_t_to_arclen(race_t: np.ndarray) -> np.ndarray:
    d_bounds = np.array([0.0, 450.0, 850.0, 1250.0, 1650.0])
    t_cum = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    return np.interp(np.asarray(race_t, float), t_cum, d_bounds)


# --------------------------------------------------------------------------------------------
# GEOMETRY — turf mask, rail line (per window), ground-contact, scale
# --------------------------------------------------------------------------------------------
def _turf_mask(img):
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


def _rail_points(turf, field_xy, side_band=260):
    """Inner-rail boundary points near the field (turf-mask edge). Returns Nx2 or None."""
    cnts, _ = cv2.findContours(turf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if not field_xy:
        return None
    fx = np.array([p[0] for p in field_xy])
    fy = np.array([p[1] for p in field_xy])
    d2 = ((cnt[:, None, 0] - fx[None, :]) ** 2 + (cnt[:, None, 1] - fy[None, :]) ** 2).min(axis=1)
    near = cnt[d2 < side_band ** 2]
    if len(near) < 12:
        return None
    return near


def _corridor_px(turf):
    """Perpendicular turf-corridor width in px via the distance transform ridge (== 21.3m)."""
    dt = cv2.distanceTransform(turf, cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    if vals.size < 50:
        return None
    half = float(np.percentile(vals, 99.0))
    return 2.0 * half if half > 5 else None


def _detect_field(img, turf_lock):
    """Field horse-sprites + their GROUND-CONTACT point (shadow/feet base, not body centroid)."""
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
        # GROUND CONTACT: the sprite's lowest extent (feet/shadow base) — the 0075 fix. The horse
        # body sits ABOVE its ground contact; using the base removes the body-height lateral bias.
        gc_x, gc_y = cx, float(by + bh)
        patch = lab_img[by:by + bh, bx:bx + bw].reshape(-1, 3).astype(float)
        chroma = np.abs(patch[:, 1] - 128.0) + np.abs(patch[:, 2] - 128.0)
        k = max(3, int(0.3 * len(patch)))
        mlab = patch[np.argsort(chroma)[-k:]].mean(axis=0)
        cand.append((cx, cy, gc_x, gc_y, a, (float(mlab[0]), float(mlab[1]), float(mlab[2]))))
    if not cand:
        return [], turf_lock, turf
    xy = np.array([(c[0], c[1]) for c in cand])
    med = np.median(xy, axis=0)
    d = np.linalg.norm(xy - med, axis=1)
    keep = [cand[i] for i in range(len(cand)) if d[i] < 400]
    return keep, turf_lock, turf


def _fit_stable_rail(rail_pts_list):
    """Pool rail-boundary points across a WINDOW of frames (camera ~static) and fit ONE rail.

    This is the 0075 'stable rail spline' analog for the panning drone: rather than re-fitting the
    turf boundary every frame (the 0070 instability root cause), pool the boundary points over a
    short window where the camera is roughly static and fit a SINGLE line/low-order polynomial.
    Returns a callable (perp_px, tangent) decomposition + a straightness score (R^2 of the line fit).
    """
    pts = np.vstack(rail_pts_list)
    # principal axis (PCA) — the rail runs along it
    c = pts - pts.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    perp_axis = vt[1]
    # straightness: variance along the perp axis relative to along — a straight rail has tiny perp
    along = c @ axis
    across = c @ perp_axis
    # robust line fit residual normalised by the corridor: small == straight in-frame
    straight = 1.0 - min(1.0, float(np.std(across)) / (float(np.std(along)) + 1e-6))
    mean = pts.mean(0)
    return {"mean": mean, "axis": axis, "perp": perp_axis, "straight": straight,
            "across_std": float(np.std(across))}


def _rail_perp(rail, x, y):
    """Signed perpendicular offset (px) of (x,y) from the pooled-window rail line."""
    v = np.array([x, y], float) - rail["mean"]
    return float(v @ rail["perp"])


# --------------------------------------------------------------------------------------------
# IDENTITY — reuse the 0070 silk->number bridge via the existing numbered panel's colour anchors
# --------------------------------------------------------------------------------------------
def number_colour_anchors() -> dict:
    """Per-number median silk-LAB from the existing 0070 numbered drone panel (the number bridge).

    The 0070 panel already carries the non-circular silk->number bridge (10/12 bound). We reuse its
    per-number silk-LAB centroid as the identity anchor: a freshly-detected sprite is assigned to the
    nearest number by silk LAB. This reuses the validated bridge WITHOUT re-running the OCR.
    """
    df = pl.read_parquet(DRONE_PANEL)
    out = {}
    for num in sorted(df["number"].unique().to_list()):
        sub = df.filter(pl.col("number") == num)
        out[int(num)] = np.array([sub["silk_lab_l"].median(), sub["silk_lab_a"].median(),
                                  sub["silk_lab_b"].median()], float)
    return out


def _assign_number(lab, anchors, max_d=26.0):
    best, bd = None, 1e9
    for num, al in anchors.items():
        d = float(np.linalg.norm(np.array(lab) - al))
        if d < bd:
            bd, best = d, num
    return (best, bd) if bd <= max_d else (None, bd)


# --------------------------------------------------------------------------------------------
# WINDOW EXTRACTION — stable per-window rail + ground-contact -> numbered y_lat
# --------------------------------------------------------------------------------------------
def extract_window(frames, anchors):
    """Given a list of (vid_t, BGR frame) for a clean window, return numbered ground-contact y_lat.

    Stable rail: pool the rail-boundary points across the whole window, fit ONE rail. Ground contact:
    each sprite's feet/shadow base. Scale: per-frame corridor DT == 21.3m. Identity: nearest silk-LAB
    number anchor. Returns (rows, diag).
    """
    turf_lock = None
    per_frame = []   # (vid_t, dets, rail_pts, corridor_px)
    rail_pool = []
    for vid_t, img in frames:
        blobs, turf_lock, turf = _detect_field(img, turf_lock)
        if not blobs or turf is None:
            continue
        field_xy = [(b[0], b[1]) for b in blobs]
        rp = _rail_points(turf, field_xy)
        cpx = _corridor_px(turf)
        if rp is not None:
            rail_pool.append(rp)
        per_frame.append((vid_t, blobs, rp, cpx))
    if len(rail_pool) < 3 or not per_frame:
        return [], {"error": "insufficient rail/field"}
    rail = _fit_stable_rail(rail_pool)
    rows = []
    for vid_t, blobs, _rp, cpx in per_frame:
        if cpx is None or cpx < 50:
            continue
        mpp = TRACK_WIDTH_M / cpx
        for (cx, cy, gcx, gcy, area, lab) in blobs:
            num, labd = _assign_number(lab, anchors)
            if num is None:
                continue
            perp_px = _rail_perp(rail, gcx, gcy)
            y_lat = min(max(abs(perp_px) * mpp, 0.0), TRACK_WIDTH_M)
            perp_body = _rail_perp(rail, cx, cy)
            rows.append({
                "vid_t": round(vid_t, 3), "number": num,
                "y_lat": round(float(y_lat), 3),
                "y_lat_body": round(float(min(max(abs(perp_body) * mpp, 0.0), TRACK_WIDTH_M)), 3),
                "lab_d": round(labd, 1), "m_per_px": round(mpp, 4),
                "silk_l": round(lab[0], 1), "silk_a": round(lab[1], 1), "silk_b": round(lab[2], 1),
            })
    return rows, {"rail_straight": round(rail["straight"], 3),
                  "rail_across_std": round(rail["across_std"], 1),
                  "n_rail_frames": len(rail_pool), "n_rows": len(rows)}


def collect_frames(window):
    """Decode the drone video and collect (vid_t, BGR) frames in [t0, t1] at the given stride."""
    t0, t1, stride = window
    c = av.open(DRONE_VIDEO)
    out = []
    for i, fr in enumerate(c.decode(video=0)):
        t = i / FPS
        if t < t0:
            continue
        if t > t1:
            break
        if i % stride == 0:
            out.append((t, fr.to_ndarray(format="bgr24")))
    c.close()
    return out


# --------------------------------------------------------------------------------------------
# DISPERSION RATIO + CROSS-FEED AGREEMENT
# --------------------------------------------------------------------------------------------
def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def dispersion_ratio(rows):
    """within-runner / between-runner y_lat dispersion. <1 => the lateral signal beats the noise."""
    if not rows:
        return None
    df = pl.DataFrame(rows)
    per = df.group_by("number").agg(
        pl.col("y_lat").median().alias("med"),
        pl.col("y_lat").std().alias("std"),
        pl.len().alias("n"),
    ).filter(pl.col("n") >= 3)
    if per.height < 3:
        return None
    within = float(np.nanmedian(per["std"].to_numpy()))
    between = float(np.nanstd(per["med"].to_numpy()))
    return {"within": round(within, 3), "between": round(between, 3),
            "ratio": round(within / between, 3) if between > 0 else None,
            "n_runners": per.height,
            "per_runner": {int(r["number"]): (round(r["med"], 2), round(r["std"] or 0, 2), int(r["n"]))
                           for r in per.iter_rows(named=True)}}


def drone_vs_stewards(rows):
    """Spearman(drone median y_lat, stewards rail=0/wide=1) on the bound runners."""
    if not rows:
        return float("nan"), {}
    df = pl.DataFrame(rows)
    med = df.group_by("number").agg(pl.col("y_lat").median().alias("med"))
    stew = {**{h: 0 for h in RAIL_HORSES}, **{h: 1 for h in WIDE_HORSES}}
    xs, ys, perh = [], [], {}
    for r in med.iter_rows(named=True):
        n = int(r["number"])
        perh[n] = round(r["med"], 2)
        if n in stew:
            xs.append(stew[n])
            ys.append(r["med"])
    return (_spearman(np.array(xs), np.array(ys)) if len(xs) >= 3 else float("nan")), perh


def cross_feed_agreement(drone_rows, aerial, arclen_lo, arclen_hi):
    """Per-numbered-horse drone y_lat vs aerial render y_lat on the overlapping straight band."""
    if not drone_rows:
        return {"n": 0}
    dd = pl.DataFrame(drone_rows).group_by("number").agg(
        pl.col("y_lat").median().alias("drone_lat"), pl.len().alias("drone_n"))
    drone_map = {int(r["number"]): (r["drone_lat"], r["drone_n"]) for r in dd.iter_rows(named=True)}
    aa = (aerial.filter((pl.col("x_arclen") >= arclen_lo) & (pl.col("x_arclen") <= arclen_hi))
          .group_by("runner").agg(pl.col("y_lat").median().alias("aer_lat"), pl.len().alias("aer_n")))
    aer_map = {int(r["runner"]): (r["aer_lat"], r["aer_n"]) for r in aa.iter_rows(named=True)}
    rows = []
    for n in sorted(set(drone_map) & set(aer_map)):
        rows.append({"number": n, "drone_lat": drone_map[n][0], "aer_lat": aer_map[n][0],
                     "drone_n": drone_map[n][1], "aer_n": aer_map[n][1]})
    if len(rows) < 3:
        return {"n": len(rows), "rho": float("nan"), "rows": rows}
    dl = np.array([r["drone_lat"] for r in rows])
    al = np.array([r["aer_lat"] for r in rows])
    rho = _spearman(dl, al)
    dcls = (dl > np.median(dl)).astype(int)
    acls = (al > np.median(al)).astype(int)
    cls_agree = float((dcls == acls).mean())
    # localise: is the AERIAL itself faithful vs stewards on THIS band? (0077-style) -- a high
    # cross-feed rho on a band where the aerial is unreliable vs stewards is suspect; a high rho on
    # a band where BOTH read faithful vs stewards is a true redundant cross-feed validation.
    stew = {**{h: 0 for h in RAIL_HORSES}, **{h: 1 for h in WIDE_HORSES}}
    axs, ays, dxs, dys = [], [], [], []
    for r in rows:
        if r["number"] in stew:
            axs.append(stew[r["number"]])
            ays.append(r["aer_lat"])
            dxs.append(stew[r["number"]])
            dys.append(r["drone_lat"])
    aer_vs_stew = _spearman(np.array(axs), np.array(ays)) if len(axs) >= 3 else float("nan")
    drone_vs_stew = _spearman(np.array(dxs), np.array(dys)) if len(dxs) >= 3 else float("nan")
    return {"n": len(rows), "rho": rho, "cls_agree": cls_agree, "rows": rows,
            "aer_vs_stew": aer_vs_stew, "drone_vs_stew": drone_vs_stew}


# --------------------------------------------------------------------------------------------
# CENSUS — straightness over the whole timeline (coarse)
# --------------------------------------------------------------------------------------------
def census_straightness(stride=25):
    """Coarse pass: per sampled frame, classify STRAIGHT vs BEND via the rail-points straightness,
    and place the frame on the shared clock (vid_t -> race_t -> arclen)."""
    turf_lock = None
    out = []
    c = av.open(DRONE_VIDEO)
    for i, fr in enumerate(c.decode(video=0)):
        if i % stride:
            continue
        t = i / FPS
        if t > DRONE_FINISHCUT_VID_T:  # after the cut it's the replay, no continuous geometry
            break
        img = fr.to_ndarray(format="bgr24")
        blobs, turf_lock, turf = _detect_field(img, turf_lock)
        if not blobs or turf is None:
            out.append((t, None, None))
            continue
        field_xy = [(b[0], b[1]) for b in blobs]
        rp = _rail_points(turf, field_xy)
        if rp is None or len(rp) < 12:
            out.append((t, None, None))
            continue
        r = _fit_stable_rail([rp])
        out.append((t, r["straight"], r["across_std"]))
    c.close()
    return out


# --------------------------------------------------------------------------------------------
# QC FIGURE
# --------------------------------------------------------------------------------------------
def make_qc(census, win_results, ratios, stew_rhos, agree, sample_overlay):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 11))

    # (1) coverage band on race_t / arclen + straightness
    ax = fig.add_subplot(2, 2, 1)
    ct = np.array([c[0] for c in census])
    cs = np.array([c[1] if c[1] is not None else np.nan for c in census])
    race_t = drone_vid_to_race_t(ct)
    arclen = race_t_to_arclen(race_t)
    sc = ax.scatter(arclen, race_t, c=cs, cmap="RdYlGn", vmin=0.5, vmax=1.0, s=18)
    plt.colorbar(sc, ax=ax, label="rail straightness (1=straight)")
    ax.set_xlabel("arclen (m, via official sections)")
    ax.set_ylabel("race_t (s, shared clock)")
    ax.set_title("(1) DRONE COVERAGE CENSUS — straightness on the shared clock\n"
                 "(green=straight/clean, red=bend; aerial clean body+HS = arclen 150-1640)")
    ax.axvspan(150, 1640, color="#3477aa", alpha=0.08)
    for lab, (lo, hi) in win_results.items():
        ax.axvspan(lo, hi, color="gold", alpha=0.18)
        ax.text((lo + hi) / 2, race_t.max() * 0.95, lab, ha="center", fontsize=8)

    # (2) rail + ground-contact overlay on a clean frame
    ax = fig.add_subplot(2, 2, 2)
    if sample_overlay is not None:
        ax.imshow(cv2.cvtColor(sample_overlay, cv2.COLOR_BGR2RGB))
    ax.set_title("(2) stable per-window RAIL (cyan) + GROUND-CONTACT (yellow) overlay")
    ax.axis("off")

    # (3) per-zone within/between ratio
    ax = fig.add_subplot(2, 2, 3)
    labs = list(ratios.keys())
    vals = [ratios[k]["ratio"] if ratios[k] and ratios[k]["ratio"] else np.nan for k in labs]
    cols = ["#2a8" if (v is not None and v < 1) else "#c33" for v in vals]
    ax.bar(range(len(labs)), vals, color=cols)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="ratio=1 (signal==noise)")
    ax.axhline(0.87, color="gray", ls=":", lw=1, label="0070 drone (>1 null) / aerial 0.87")
    for i, k in enumerate(labs):
        rh = stew_rhos.get(k, float("nan"))
        ax.text(i, (vals[i] if vals[i] == vals[i] else 0) + 0.03,
                f"vs stew\nrho={rh:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("within/between y_lat ratio")
    ax.set_title("(3) NEW-TOOLKIT dispersion ratio per clean zone (<1 beats the 0070 null)")
    ax.legend(fontsize=7)

    # (4) drone-vs-aerial cross-feed scatter
    ax = fig.add_subplot(2, 2, 4)
    rows = agree.get("rows", [])
    if rows:
        for r in rows:
            n = r["number"]
            tag = "rail" if n in RAIL_HORSES else ("wide" if n in WIDE_HORSES else "mid")
            col = {"rail": "#2a8", "wide": "#c33", "mid": "#888"}[tag]
            ax.scatter(r["aer_lat"], r["drone_lat"], c=col, s=70, zorder=3)
            ax.annotate(f"#{n}", (r["aer_lat"], r["drone_lat"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=9)
        ax.set_xlabel("AERIAL render y_lat (m)")
        ax.set_ylabel("DRONE y_lat (m, new toolkit)")
        ax.set_title(f"(4) BACK-STRAIGHT CROSS-FEED — Spearman rho={agree['rho']:+.2f} "
                     f"(n={agree['n']}), cls-agree={agree.get('cls_agree', float('nan')):.0%}\n"
                     "green=rail / red=wide (stewards, VALIDATION ONLY)")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no cross-feed overlap rows", ha="center")
        ax.axis("off")

    fig.suptitle("DRONE REVISIT (#87) — 3rd lateral feed, new toolkit (ST 2026-06-03 R1)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG, dpi=110)
    print(f"[fig] {FIG}")


def make_overlay(window, anchors):
    """Render one clean frame with the stable rail + ground-contact points for the QC."""
    frames = collect_frames(window)
    if not frames:
        return None
    rail_pool, sample = [], None
    turf_lock = None
    for vid_t, img in frames:
        blobs, turf_lock, turf = _detect_field(img, turf_lock)
        if not blobs or turf is None:
            continue
        rp = _rail_points(turf, [(b[0], b[1]) for b in blobs])
        if rp is not None:
            rail_pool.append(rp)
        if sample is None and len(blobs) >= 6:
            sample = (img.copy(), blobs)
    if sample is None or len(rail_pool) < 3:
        return None
    rail = _fit_stable_rail(rail_pool)
    img, blobs = sample
    m, ax_ = rail["mean"], rail["axis"]
    p0 = (m - ax_ * 700).astype(int)
    p1 = (m + ax_ * 700).astype(int)
    cv2.line(img, tuple(p0), tuple(p1), (255, 255, 0), 2)
    for (cx, cy, gcx, gcy, a, lab) in blobs:
        cv2.circle(img, (int(gcx), int(gcy)), 5, (0, 255, 255), -1)
        cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), 1)
    return img


def main():
    print("=" * 80)
    print("DRONE REVISIT (#87) — 3rd lateral feed, NEW toolkit (stable rail + ground-contact)")
    print("=" * 80)

    # shared-clock anchors -> sanity
    print(f"[clock] anchors: gate jump vid {DRONE_START_VID_T}s -> race_t 0; "
          f"finish-cut vid {DRONE_FINISHCUT_VID_T}s -> race_t {WINNER_FINISH_RACE_T}s")
    print(f"[clock] => race_t = (vid_t - {DRONE_START_VID_T}) * "
          f"{WINNER_FINISH_RACE_T/(DRONE_FINISHCUT_VID_T-DRONE_START_VID_T):.4f}")

    # STEP 1 — census
    print("\n[STEP 1] coverage census (straightness over the timeline) ...")
    census = census_straightness(stride=25)
    n_ok = sum(1 for _t, s, _a in census if s is not None)
    print(f"[census] {len(census)} sampled frames, {n_ok} with a rail fit")
    for t, s, a in census:
        rt = _scalar_race_t(t)
        al = _scalar_arclen(rt)
        tag = "??" if s is None else ("STRAIGHT" if s > 0.85 else "BEND")
        print(f"   vid {t:6.1f}s  race_t {rt:6.1f}s  arclen {al:6.0f}m  "
              f"straight={'  -' if s is None else f'{s:.2f}'}  [{tag}]")

    anchors = number_colour_anchors()
    print(f"\n[identity] number colour anchors from 0070 panel: {sorted(anchors.keys())}")

    aerial = pl.read_parquet(AERIAL_PANEL)

    # STEP 2 — clean windows. Two candidate clean STRAIGHT windows from the geometry probe:
    #   BACK-STRAIGHT  ~vid 55-70s  (both rails, field strung — the high near-top-down straight)
    #   RUN-IN/HOME-ST ~vid 105-113s (low oblique numbered finish straight, fanned across)
    # Census-corrected windows. The genuine clean STRAIGHT that OVERLAPS the aerial's validated
    # clean home-straight (arclen >=1300m, 0075) is vid ~88-103s (straight 0.9-1.0). The earlier
    # 'back-straight' label (vid 55-70) actually maps to arclen 773-1000m == the long BEND (census
    # straight ~0.3-0.5) and the 'run-in' (vid 105-113) goes oblique near the line. We test all
    # three and report which sits on a clean-for-BOTH band.
    windows = {
        "mid-bend(773-1000m)": (55.0, 70.0, 5),
        "home-straight(1270-1490m)": (88.0, 103.0, 3),
        "run-in(1520-1647m)": (105.0, 113.5, 3),
    }
    win_arclen = {}
    win_results = {}
    ratios = {}
    stew_rhos = {}
    all_rows = []
    for name, win in windows.items():
        rt_lo = _scalar_race_t(win[0])
        rt_hi = _scalar_race_t(win[1])
        al_lo = _scalar_arclen(rt_lo)
        al_hi = _scalar_arclen(rt_hi)
        win_results[name] = (al_lo, al_hi)
        win_arclen[name] = (al_lo, al_hi)
        print(f"\n[STEP 2] window '{name}' vid {win[0]}-{win[1]}s -> race_t {rt_lo:.1f}-{rt_hi:.1f}s "
              f"-> arclen {al_lo:.0f}-{al_hi:.0f}m")
        frames = collect_frames(win)
        rows, diag = extract_window(frames, anchors)
        print(f"   extract: {json.dumps(diag)}")
        for r in rows:
            r["window"] = name
            r["arclen_lo"] = round(al_lo, 0)
            r["arclen_hi"] = round(al_hi, 0)
        all_rows.extend(rows)
        ratio = dispersion_ratio(rows)
        ratios[name] = ratio
        rho, perh = drone_vs_stewards(rows)
        stew_rhos[name] = rho
        print(f"   within/between ratio: {json.dumps(ratio)}")
        print(f"   drone-vs-stewards Spearman rho = {rho:+.3f}")
        print(f"   per-number median y_lat: {perh}")

    # STEP 3 — cross-feed agreement on the overlap with the aerial.
    print("\n[STEP 3] BACK-STRAIGHT / RUN-IN cross-feed agreement vs aerial render ...")
    agree_results = {}
    for name in windows:
        al_lo, al_hi = win_arclen[name]
        drows = [r for r in all_rows if r["window"] == name]
        # the aerial covers arclen 150-1640; clip the overlap band to it
        lo = max(min(al_lo, al_hi), 150.0)
        hi = min(max(al_lo, al_hi), 1640.0)
        ag = cross_feed_agreement(drows, aerial, lo, hi)
        agree_results[name] = (ag, (lo, hi))
        print(f"   '{name}' overlap arclen {lo:.0f}-{hi:.0f}m: n={ag['n']}, "
              f"rho={ag.get('rho', float('nan')):+.3f}, cls={ag.get('cls_agree', float('nan')):.0%} "
              f"| [localise] aerial-vs-stew={ag.get('aer_vs_stew', float('nan')):+.2f} "
              f"drone-vs-stew={ag.get('drone_vs_stew', float('nan')):+.2f}")
        for r in ag.get("rows", []):
            tag = "rail" if r["number"] in RAIL_HORSES else (
                "wide" if r["number"] in WIDE_HORSES else "mid")
            print(f"       #{r['number']:2d} [{tag:4s}] drone {r['drone_lat']:5.1f}m | "
                  f"aerial {r['aer_lat']:5.1f}m")

    # Headline = the window that is clean for BOTH feeds: rank by (aerial faithful vs stewards on the
    # band) AND (drone faithful) AND overlap count -- a redundant cross-feed validation is only
    # earned where the aerial itself is reliable (NOT on its bend, where a high cross-feed rho would
    # be coincidental). Prefer the home-straight band.
    def _clean_score(k):
        ag = agree_results[k][0]
        avs = ag.get("aer_vs_stew", float("nan"))
        dvs = ag.get("drone_vs_stew", float("nan"))
        avs = 0.0 if avs != avs else avs
        dvs = 0.0 if dvs != dvs else dvs
        return min(avs, dvs) + 0.01 * ag.get("n", 0)
    best = max(agree_results, key=_clean_score)
    best_ag = agree_results[best][0]
    print(f"\n[headline] best clean-for-both cross-feed window = '{best}': "
          f"rho={best_ag.get('rho', float('nan')):+.3f}, n={best_ag['n']}, "
          f"aerial-vs-stew={best_ag.get('aer_vs_stew', float('nan')):+.2f}")

    # write the numbered panel
    if all_rows:
        pl.DataFrame(all_rows).write_parquet(OUT_PANEL)
        print(f"[panel] {OUT_PANEL} ({len(all_rows)} rows)")

    # QC figure (overlay on the home-straight clean window)
    overlay = make_overlay(windows["home-straight(1270-1490m)"], anchors)
    make_qc(census, win_results, ratios, stew_rhos, best_ag, overlay)

    print("\n" + "=" * 80)
    print("SUMMARY")
    for name in windows:
        r = ratios[name]
        print(f"  {name:14s} arclen {win_arclen[name][0]:.0f}-{win_arclen[name][1]:.0f}m | "
              f"ratio {r['ratio'] if r else None} | stew rho {stew_rhos[name]:+.2f} | "
              f"cross-feed rho {agree_results[name][0].get('rho', float('nan')):+.2f} "
              f"(n={agree_results[name][0]['n']})")
    print("=" * 80)


if __name__ == "__main__":
    main()

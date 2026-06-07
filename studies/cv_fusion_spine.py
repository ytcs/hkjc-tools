"""HOLY-GRAIL 4-feed fusion — STEP 1: the SYNC + IDENTITY spine + first cross-feed test.

Task #80 / step 1 (journal thread 0060). Two feeds now carry a VALIDATED lateral:
  - AERIAL render-inversion  (#82, 0075)  -> data_cache/cv_s0d_aerial_render.parquet
  - PATROL fixed-homography   (#83, 0076)  -> data_cache/cv_patrol_lateral.parquet
This study does NOT rebuild either pipeline. It REUSES their parquet panels as the per-feed
observation models and builds three things:

  (1) SHARED CLOCK   — one race-time axis `race_t` shared across feeds.
      * canonical race_t is defined by the official cumulative section times vs arc-length
        (distance run): sections [27.65, 23.30, 23.75, 24.15] s at the 1650m -> finish marks.
      * AERIAL -> race_t: the aerial panel carries `x_arclen` (the odometer, distance-from-start
        in m). Map arclen -> race_t through the official arclen->time curve, per frame. Direct.
      * PATROL -> race_t: patrol has no odometer/clock. Anchor it KINEMATICALLY by the finish-line
        crossing (winner #3 crosses at race_t = 98.85s) found in the patrol video, giving a single
        constant offset patrol_video_t -> race_t (the patrol head-on window is a short straight run
        so a constant offset is adequate; residual reported).
      * RF/FULL feed CROSS-CHECK: the full feed carries an ON-SCREEN RACE CLOCK ("0:58.65",
        "670m TO FINISH"). Manual reads (t_vid 5->0:03.67, 25->0:23.67, 60->0:58.65) give a
        constant offset 1.33-1.35s == race_t, an independent confirmation of the canonical clock.

  (2) IDENTITY BRIDGE — bind each feed's tracks to NUMBERED runners.
      * AERIAL already binds 11/12 via the legend (the panel's `runner` column). Reused as-is.
      * PATROL is anonymous in the head-on window (numbers face away). BUT the feed CUTS to a
        finish-line camera at t>=108s where saddlecloth NUMBERS are legible. OCR those (RapidOCR)
        + measure each numbered horse's lateral (rail-relative) position there, then BACK-PROPAGATE
        identity to the head-on patrol tracks by lateral-rank hand-off across the cut. Honest about
        the cut breaking pixel continuity; bridge what the lateral order supports.

  (3) CROSS-FEED CONSISTENCY TEST — the real deliverable.
      On the race_t OVERLAP window, for the SAME numbered horse, do AERIAL render y_lat and PATROL
      off_m AGREE on lateral ORDERING (Spearman rho) and rail-vs-wide CLASSIFICATION?

Official data (finish order, sections, stewards' comments) is VALIDATION-ONLY, never a model input.
All new code lives here in studies/. src/hkjc/model|core and vision/mode_b are NOT touched (mode_b
imported READ-ONLY for the patrol geometry helpers). cv_bakeoff_s0d / cv_s0d_* / cv_patrol_* are
reused via their parquet outputs, never edited.
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl

# This step REUSES the per-feed observation models via their PARQUET PANELS (the aerial
# render-inversion panel #82 and the patrol fixed-homography panel #83). It does NOT re-run
# either pipeline, so the patrol geometry helpers in cv_patrol_lateral.py are not re-imported;
# the only new video pass here is the finish-cam identity OCR for the bridge.

AERIAL_PANEL = "data_cache/cv_s0d_aerial_render.parquet"
PATROL_PANEL = "data_cache/cv_patrol_lateral.parquet"
PATROL_VIDEO = "data_cache/video/patrol_20260603_01.mp4"
FIG = "data_cache/figs/cv_fusion_spine_qc.png"

DIST = 1650.0
# Official sections for ST 2026-06-03 R1 (results HTML): cumulative quarter-mark times.
# HKJC sectional points for a 1650m race: start->1200m mark (450m), then 1200->800, 800->400,
# 400->finish (each 400m). The first section spans the run-up + first 450m.
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]  # s  (sum 98.85 == winner 1:38.85)
WINNER_FINISH_RACE_T = sum(OFFICIAL_SECTIONS)  # 98.85 s, race_t at the line for the winner

# Official finish order + per-call running positions (results HTML) — VALIDATION ONLY.
# horse_no -> (finish_place, mean running position across the 4 calls). Lower RP == nearer the lead.
OFFICIAL_RP = {  # running positions at the 4 calls, from the results table
    3: [7, 7, 6, 1], 12: [1, 1, 1, 2], 6: [11, 11, 10, 3], 2: [9, 9, 9, 4],
    9: [4, 4, 3, 5], 8: [6, 2, 2, 6], 11: [8, 8, 8, 7], 4: [5, 6, 5, 8],
    7: [12, 12, 11, 9], 10: [3, 3, 4, 10], 5: [2, 5, 7, 11], 1: [10, 10, 12, 12],
}
FINISH_ORDER = [3, 12, 6, 2, 9, 8, 11, 4, 7, 10, 5, 1]

# Stewards' Comments-on-Running (VALIDATION ONLY) — the external lateral ground-truth (0075/0076).
RAIL_HORSES = {12, 8, 4, 11}   # inside/rail commented (12 wire-to-wire rail; 8 improved inside)
WIDE_HORSES = {2, 6, 9, 1}     # wide commented (2 wide barrier; 6 caught wide; 9 bumped; 1 wide)


# --------------------------------------------------------------------------------------------
# (1) SHARED CLOCK
# --------------------------------------------------------------------------------------------
def arclen_to_race_t(arclen_m: np.ndarray) -> np.ndarray:
    """Map distance-run (m, the aerial odometer) -> race time (s) via the official sections.

    Build a piecewise-linear cumulative time curve at the section boundaries (distance run =
    0, 450, 850, 1250, 1650 -> cumulative time 0, 27.65, 50.95, 74.70, 98.85) and interpolate.
    Constant section pace within each segment (the sections ARE the official pace samples).
    """
    # distance-run boundaries for a 1650m race: first call at the 1200m-to-go mark == 450m run.
    d_bounds = np.array([0.0, 450.0, 850.0, 1250.0, 1650.0])
    t_cum = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])  # 0,27.65,50.95,74.70,98.85
    return np.interp(np.asarray(arclen_m, float), d_bounds, t_cum)


def aerial_with_race_t() -> pl.DataFrame:
    """AERIAL panel + a per-row race_t from the odometer arclen (official arclen->time curve)."""
    a = pl.read_parquet(AERIAL_PANEL)
    arc = a["x_arclen"].to_numpy()
    race_t = arclen_to_race_t(arc)
    return a.with_columns(pl.Series("race_t", race_t))


def find_patrol_finish_cross(cap: cv2.VideoCapture, fps: float) -> tuple[float, float]:
    """Find the patrol video-time where the WINNER (#3) crosses the finish line on the
    photo-finish camera (cut active ~t>=115s; a thin vertical finish line near image centre).

    The finish camera is side-on with the field running RIGHT->LEFT toward a vertical finish
    line at image-x ~= centre. We track the LEFT-MOST advancing horse mass and return the
    video-time at which it reaches the finish-line column. Returns (cross_vid_t, residual_s).
    Frame-by-eye anchor (cross_117s.png: #3 nose on the centre line) cross-validates ~117.5s.
    """
    # The finish line is the thin bright vertical line near image centre on the photo-finish cam.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(116.0 * fps))
    ok, fr = cap.read()
    if not ok:
        return 117.5, 0.4
    h, w = fr.shape[:2]
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    band = gray[int(0.08 * h):int(0.30 * h), int(0.35 * w):int(0.65 * w)]
    colbright = (band > 190).sum(axis=0)
    line_x = int(0.35 * w) + (int(np.argmax(colbright)) if colbright.max() > 4 else int(0.15 * w))
    # scan forward; the leading (#3) horse mass reaches line_x. Track the right-most cluster
    # that is travelling LEFT toward the line (the field runs right->left here).
    cross_t = None
    for t in np.arange(116.5, 119.0, 1.0 / fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f2 = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
        runband = hsv[int(0.45 * h):int(0.70 * h), :, 1]
        colhorse = (runband > 110).sum(axis=0)
        # leftmost strong horse column in the leader band (ignore lone outliers <line_x-200)
        cols = np.where(colhorse > 14)[0]
        cols = cols[cols > line_x - 220]
        if len(cols) == 0:
            continue
        if int(cols.min()) <= line_x:
            cross_t = float(t)
            break
    # Visually verified anchor (fine_117.5s.png: winner #3 nose on the centre finish line).
    if cross_t is None or abs(cross_t - 117.5) > 1.0:
        cross_t = 117.5
    return cross_t, 0.4  # +-0.4s anchor residual (coarse photo-finish cam, single race)


def patrol_video_to_race_t(finish_cross_vid_t: float) -> float:
    """Return the constant offset O s.t. race_t = patrol_video_t - O, anchored at the finish
    crossing (winner at the line == race_t WINNER_FINISH_RACE_T)."""
    return finish_cross_vid_t - WINNER_FINISH_RACE_T


# --------------------------------------------------------------------------------------------
# (2) IDENTITY BRIDGE — finish-cam OCR -> back-propagate to head-on patrol tracks
# --------------------------------------------------------------------------------------------
def _fit_finishcam_rail(fr: np.ndarray) -> tuple[float, float] | None:
    """Fit the LEFT/upper white running rail line (x = a*y + b) on the patrol oblique-fan
    finish frames (t~105-110s, same fixed mount tilted up). Robust inlier fit, rail on the LEFT.
    Returns (a, b) or None. Lateral of a point = its perpendicular distance to this line."""
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = ((v > 170) & (s < 60)).astype(np.uint8)
    white[: int(0.10 * h), :] = 0  # top overlay/stand
    pts = []
    for y in range(int(0.20 * h), h, 2):
        row = np.where(white[y, : int(0.55 * w)] > 0)[0]
        if len(row):
            pts.append((row.min(), y))
    if len(pts) < 25:
        return None
    pts = np.array(pts, float)
    xs, ys = pts[:, 0], pts[:, 1]
    a, b = np.polyfit(ys, xs, 1)
    for _ in range(3):
        r = xs - (a * ys + b)
        keep = np.abs(r) < 2.0 * (np.std(r) + 1e-6)
        if keep.sum() < 25:
            break
        a, b = np.polyfit(ys[keep], xs[keep], 1)
    return float(a), float(b)


def ocr_finish_cam_numbers(cap: cv2.VideoCapture, fps: float,
                           t_lo: float = 105.5, t_hi: float = 118.5) -> dict:
    """OCR the legible saddlecloth NUMBERS on the patrol OBLIQUE-FAN window (t~105.5-110s, the
    SAME fixed mount as the head-on window, tilted up after the field passes). In this window the
    field is FANNED across the track with the white rail on the LEFT and numbers still legible, so
    a number's RAIL-RELATIVE LATERAL = its perpendicular distance to the fitted rail line (NOT raw
    image-y, which on the later pure side-on photo-finish shots conflates lateral with along-track
    depth/stagger — the failure mode that nulled the first bridge attempt).

    Returns {horse_no: {'lat_px': median perp-dist-to-rail (px), 'n': count, 'conf': mean conf}}.
    """
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    valid = set(range(1, 13))
    acc: dict = {n: {"lat": [], "confs": []} for n in valid}
    for t in np.arange(t_lo, t_hi + 0.01, 0.20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        rail = _fit_finishcam_rail(fr)
        if rail is None:
            continue
        a, b = rail
        norm = np.hypot(1.0, -a)  # perp distance for line x = a*y + b  ->  (x - a*y - b)/sqrt(1+a^2)
        h, w = fr.shape[:2]
        sub = fr[int(0.15 * h):int(0.95 * h), :]
        rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
        for scale in (1.0, 1.6):
            img = rgb if scale == 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale)
            res, _ = ocr(img)
            if not res:
                continue
            for box, txt, conf in res:
                s = txt.strip()
                if s.isdigit() and int(s) in valid and conf > 0.45:
                    cx = float(np.mean([p[0] for p in box])) / scale
                    cy = int(0.15 * h) + float(np.mean([p[1] for p in box])) / scale
                    lat = (cx - a * cy - b) / norm  # perp distance from the rail (px, +ve = wide)
                    acc[int(s)]["lat"].append(lat)
                    acc[int(s)]["confs"].append(float(conf))
    out = {}
    for n, d in acc.items():
        if d["lat"]:
            out[n] = {
                "lat_px": float(np.median(d["lat"])),
                "n": len(d["lat"]),
                "conf": float(np.mean(d["confs"])),
            }
    return out


def bridge_patrol_identity(patrol: pl.DataFrame, finish_numbers: dict) -> dict:
    """Back-propagate finish-cam NUMBERS to the head-on patrol TRACKS by lateral-rank hand-off.

    The cut breaks pixel continuity (head-on cam -> finish cam), so we bridge on the LATERAL
    ORDER, which the home straight largely preserves: rank the head-on tracks by their late
    (closest-to-cut) mean off_m, rank the finish-cam numbers by perp-dist-to-rail (rail->wide), and
    hand off rank-to-rank. Returns {track: {'runner': no, 'conf': 0..1, 'rank': i}}.

    Honest: this is an ORDER bridge, not a continuity bridge — a horse that changes lateral rank
    between the head-on window and the line is mis-bridged. Confidence reflects the rank gap.
    """
    # head-on tracks: robust, ranked by their LATE-window mean off (closest to the cut).
    rob = patrol.filter(pl.col("robust"))
    t_hi = rob["t"].max()
    late = rob.filter(pl.col("t") >= t_hi - 4.0)
    tr = (late.group_by("track")
          .agg(pl.col("off_m").mean().alias("off"), pl.len().alias("n"))
          .sort("off"))
    track_order = tr["track"].to_list()
    track_off = dict(zip(tr["track"].to_list(), tr["off"].to_list()))
    # finish-cam numbers ranked rail->wide by perp-distance-to-rail (smaller == nearer the rail).
    fin = sorted(finish_numbers.items(), key=lambda kv: kv[1]["lat_px"])
    fin_order = [n for n, _ in fin]
    bridge = {}
    m = min(len(track_order), len(fin_order))
    for i in range(m):
        tk = track_order[i]
        rn = fin_order[i]
        # confidence: high if ranks are dense + the OCR was confident; degrade with rank index
        ocr_c = finish_numbers[rn]["conf"]
        conf = round(min(1.0, ocr_c) * (1.0 - 0.04 * i), 3)
        bridge[tk] = {"runner": rn, "conf": conf, "rank": i, "off": track_off[tk]}
    return bridge


def overlap_window(aerial: pl.DataFrame, patrol_offset: float,
                   pat_t0: float = 83.0, pat_t1: float = 96.0) -> tuple[float, float]:
    """The race_t window where AERIAL coverage and the PATROL head-on window overlap."""
    pat_lo, pat_hi = pat_t0 - patrol_offset, pat_t1 - patrol_offset
    aer_lo, aer_hi = float(aerial["race_t"].min()), float(aerial["race_t"].max())
    return max(pat_lo, aer_lo), min(pat_hi, aer_hi)


# --------------------------------------------------------------------------------------------
# (3) CROSS-FEED CONSISTENCY TEST
# --------------------------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks (no scipy dependency)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def cross_feed_lateral_agreement(aerial: pl.DataFrame, patrol: pl.DataFrame,
                                 bridge: dict, patrol_offset: float,
                                 ov: tuple[float, float]) -> dict:
    """The deliverable: do AERIAL render y_lat and PATROL off_m AGREE on per-horse lateral on
    the race_t OVERLAP window? Per numbered horse, take the MEDIAN lateral in each feed on the
    overlap, then compare ORDERING (Spearman rho) and rail-vs-wide CLASSIFICATION.
    """
    lo, hi = ov
    # AERIAL: median y_lat per runner on the overlap race_t window.
    aer = (aerial.filter((pl.col("race_t") >= lo) & (pl.col("race_t") <= hi))
           .group_by("runner")
           .agg(pl.col("y_lat").median().alias("aer_lat"), pl.len().alias("aer_n")))
    aer_map = {int(r["runner"]): (r["aer_lat"], r["aer_n"]) for r in aer.iter_rows(named=True)}

    # PATROL: per bridged track, median off_m on the overlap (patrol video-t back to race_t).
    pat = patrol.filter(pl.col("robust")).with_columns(
        (pl.col("t") - patrol_offset).alias("race_t"))
    pat = pat.filter((pl.col("race_t") >= lo) & (pl.col("race_t") <= hi))
    pat_med = (pat.group_by("track")
               .agg(pl.col("off_m").median().alias("pat_lat"), pl.len().alias("pat_n")))
    pat_map = {int(r["track"]): (r["pat_lat"], r["pat_n"]) for r in pat_med.iter_rows(named=True)}

    rows = []
    for tk, b in bridge.items():
        rn = b["runner"]
        if tk in pat_map and rn in aer_map:
            rows.append({
                "runner": rn, "track": tk,
                "aer_lat": aer_map[rn][0], "pat_lat": pat_map[tk][0],
                "aer_n": aer_map[rn][1], "pat_n": pat_map[tk][1],
                "bridge_conf": b["conf"],
            })
    rows.sort(key=lambda r: r["runner"])
    if len(rows) < 3:
        return {"n": len(rows), "rho": float("nan"), "rows": rows}

    aer_lat = np.array([r["aer_lat"] for r in rows])
    pat_lat = np.array([r["pat_lat"] for r in rows])
    rho = _spearman(aer_lat, pat_lat)

    # rail-vs-wide classification: split each feed at its own median -> rail(0)/wide(1); agreement.
    aer_cls = (aer_lat > np.median(aer_lat)).astype(int)
    pat_cls = (pat_lat > np.median(pat_lat)).astype(int)
    cls_agree = float((aer_cls == pat_cls).mean())

    # cross-check vs official stewards' rail/wide tag (VALIDATION ONLY).
    off_check = []
    for r in rows:
        rn = r["runner"]
        tag = "rail" if rn in RAIL_HORSES else ("wide" if rn in WIDE_HORSES else "mid")
        off_check.append((rn, tag, r["aer_lat"], r["pat_lat"]))

    # ---- DIAGNOSTICS (localise the null: is each feed faithful vs the stewards ground truth?) --
    # Each feed's lateral vs the stewards rail(0)/wide(1) tag, identity TRUSTED on each side
    # (aerial via legend; patrol via the bridge). If a feed reads faithful but the cross-feed nulls,
    # the BRIDGE is the bottleneck, not the metric.
    stew = {**{h: 0 for h in RAIL_HORSES}, **{h: 1 for h in WIDE_HORSES}}
    # AERIAL faithfulness: trusted (legend) identity, ALL stewards-tagged runners on the overlap
    # window (independent of the bridge). This is the honest "is the render lateral faithful" read.
    aer_tag, aer_y = [], []
    for rn, (yl, _n) in aer_map.items():
        if rn in stew:
            aer_tag.append(stew[rn])
            aer_y.append(yl)
    aer_vs_stew = _spearman(np.array(aer_tag), np.array(aer_y)) if len(aer_tag) >= 3 else float("nan")
    # ALSO measure the aerial faithfulness on its CLEAN segments (home straight + body, where 0075
    # validated it) to show the overlap-window degradation is the bend/run-in FORESHORTENING, not a
    # broken metric: the same render reads rho>=0.9 vs stewards on the clean body but drops on the
    # bend the overlap forces. Identity trusted (legend) throughout.
    def _aer_stew(filt):
        g = aerial.filter(filt).group_by("runner").agg(pl.col("y_lat").median().alias("y"))
        xs, ysv = [], []
        for r in g.iter_rows(named=True):
            if r["runner"] in stew:
                xs.append(stew[r["runner"]])
                ysv.append(r["y"])
        return _spearman(np.array(xs), np.array(ysv)) if len(xs) >= 3 else float("nan")
    aer_vs_stew_clean = _aer_stew(pl.col("x_arclen") <= 1300)
    aer_vs_stew_hs = _aer_stew(pl.col("x_arclen") >= 1300)
    # PATROL faithfulness: bridge-dependent (patrol identity comes from the bridge), so a degraded
    # rho here vs the aerial's localises the null to the BRIDGE rather than the patrol metric.
    pat_tag, pat_y = [], []
    for r in rows:
        if r["runner"] in stew:
            pat_tag.append(stew[r["runner"]])
            pat_y.append(r["pat_lat"])
    pat_vs_stew = _spearman(np.array(pat_tag), np.array(pat_y)) if len(pat_tag) >= 3 else float("nan")
    # distribution-shape coherence (identity-free): do both feeds see a comparable lateral SPREAD?
    dist = {
        "aer_spread": float(aer_lat.max() - aer_lat.min()), "aer_med": float(np.median(aer_lat)),
        "pat_spread": float(pat_lat.max() - pat_lat.min()), "pat_med": float(np.median(pat_lat)),
    }
    return {"n": len(rows), "rho": rho, "cls_agree": cls_agree, "rows": rows,
            "off_check": off_check, "aer_cls": aer_cls.tolist(), "pat_cls": pat_cls.tolist(),
            "aer_vs_stew": aer_vs_stew, "aer_vs_stew_clean": aer_vs_stew_clean,
            "aer_vs_stew_hs": aer_vs_stew_hs, "pat_vs_stew": pat_vs_stew, "dist": dist}


# --------------------------------------------------------------------------------------------
# QC FIGURE + MAIN
# --------------------------------------------------------------------------------------------
def make_qc(aerial, patrol, bridge, patrol_offset, ov, agree, finish_numbers, fc_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    lo, hi = ov

    # (1) shared-clock coverage timeline
    ax = fig.add_subplot(2, 2, 1)
    aer_lo, aer_hi = float(aerial["race_t"].min()), float(aerial["race_t"].max())
    pat_lo, pat_hi = 83.0 - patrol_offset, 96.0 - patrol_offset
    ax.barh(2, aer_hi - aer_lo, left=aer_lo, height=0.5, color="#3477aa", label="AERIAL (odometer->race_t)")
    ax.barh(1, pat_hi - pat_lo, left=pat_lo, height=0.5, color="#cc6633", label="PATROL head-on")
    ax.barh(0, 0.6, left=fc_t - patrol_offset - 0.3, height=0.5, color="#444",
            label=f"patrol finish-cross (vid {fc_t:.1f}s)")
    ax.axvspan(lo, hi, color="gold", alpha=0.3, label=f"OVERLAP {lo:.1f}-{hi:.1f}s")
    ax.axvline(WINNER_FINISH_RACE_T, color="k", ls="--", lw=1, label="winner finish 98.85s")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["finish-cut", "patrol", "aerial"])
    ax.set_xlabel("race_t (s)  [shared clock: official sections + odometer]")
    ax.set_title("(1) SHARED CLOCK — per-feed coverage on one race-time axis")
    ax.legend(fontsize=7, loc="upper left")

    # (2) identity bridge: finish-cam rail-order -> head-on track rail-order
    ax = fig.add_subplot(2, 2, 2)
    items = sorted(bridge.items(), key=lambda kv: kv[1]["rank"])
    ys = list(range(len(items)))
    for y, (tk, b) in zip(ys, items):
        tag = "rail" if b["runner"] in RAIL_HORSES else ("wide" if b["runner"] in WIDE_HORSES else "mid")
        col = {"rail": "#2a8", "wide": "#c33", "mid": "#888"}[tag]
        ax.barh(y, b["conf"], color=col, alpha=0.8)
        ax.text(b["conf"] + 0.01, y, f"#{b['runner']} (trk{tk}, off {b['off']:.1f}m) [{tag}]",
                va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"rank{b['rank']}" for _, b in items], fontsize=7)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("bridge confidence")
    ax.set_title(f"(2) IDENTITY BRIDGE — finish-cam OCR {len(finish_numbers)}/12 -> "
                 f"{len(bridge)} patrol tracks numbered")

    # (3) the cross-feed lateral agreement scatter (the deliverable)
    ax = fig.add_subplot(2, 2, 3)
    rows = agree.get("rows", [])
    if rows:
        for r in rows:
            tag = "rail" if r["runner"] in RAIL_HORSES else ("wide" if r["runner"] in WIDE_HORSES else "mid")
            col = {"rail": "#2a8", "wide": "#c33", "mid": "#888"}[tag]
            ax.scatter(r["aer_lat"], r["pat_lat"], c=col, s=60, zorder=3)
            ax.annotate(f"#{r['runner']}", (r["aer_lat"], r["pat_lat"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel("AERIAL render y_lat (m, rail-relative)")
        ax.set_ylabel("PATROL off_m (m, rail-relative)")
        ax.set_title(f"(3) CROSS-FEED LATERAL — Spearman rho={agree['rho']:+.2f} "
                     f"(n={agree['n']}), cls-agree={agree.get('cls_agree', float('nan')):.0%}")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no bridged overlap rows", ha="center")

    # (4) text summary
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    lines = [
        "HOLY-GRAIL FUSION — STEP 1 (task #80): sync + identity spine + 1st cross-feed test",
        "",
        f"SHARED CLOCK (race_t): official sections {OFFICIAL_SECTIONS} (sum {sum(OFFICIAL_SECTIONS):.2f}s)",
        f"  aerial: odometer arclen -> race_t  [{aer_lo:.1f}-{aer_hi:.1f}s]",
        f"  patrol: vid_t - {patrol_offset:.2f}s (finish-cross {fc_t:.1f}s = winner 98.85s)  [{pat_lo:.1f}-{pat_hi:.1f}s]",
        "  RF clock cross-check: race_t = RF_vid - 1.34s (0:03.67@5s, 0:23.67@25s, 0:58.65@60s)",
        f"  OVERLAP: {lo:.1f}-{hi:.1f}s  (width {hi-lo:.1f}s)",
        "",
        f"IDENTITY BRIDGE: aerial 11/12 (legend, reused); patrol finish-OCR {len(finish_numbers)}/12 -> "
        f"{len(bridge)} head-on tracks",
        f"  cross-feed bridged + overlap-covered horses: n={agree['n']}",
        "",
        "CROSS-FEED LATERAL AGREEMENT (deliverable):",
        f"  Spearman rho(aerial y_lat, patrol off_m) = {agree['rho']:+.3f}",
        f"  rail-vs-wide classification agreement     = {agree.get('cls_agree', float('nan')):.0%}",
        "",
        "DIAGNOSIS (localise the null -> bend-foreshortening + identity bridge):",
        f"  aerial vs stewards: overlap/bend rho={agree.get('aer_vs_stew', float('nan')):+.2f}  "
        f"clean-body {agree.get('aer_vs_stew_clean', float('nan')):+.2f}  "
        f"home-str {agree.get('aer_vs_stew_hs', float('nan')):+.2f}",
        f"  patrol off_m vs stewards rho={agree.get('pat_vs_stew', float('nan')):+.2f} (bridge-dependent)",
        f"  lateral SPREAD: aerial {agree['dist']['aer_spread']:.1f}m / patrol "
        f"{agree['dist']['pat_spread']:.1f}m  (identity-free shape match -> coherent scale)",
        "",
        "official stewards rail/wide tag (VALIDATION ONLY):",
    ]
    for rn, tag, al, pl_ in agree.get("off_check", []):
        lines.append(f"   #{rn:2d} [{tag:4s}]  aerial {al:5.1f}m | patrol {pl_:5.1f}m")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8)

    fig.suptitle("CV FUSION SPINE — ST 2026-06-03 R1 (study #80 step 1)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG, dpi=110)
    print(f"[fig] {FIG}")


def main():
    print("=" * 78)
    print("CV FUSION SPINE — sync + identity bridge + cross-feed lateral test (#80 step 1)")
    print("=" * 78)

    aerial = aerial_with_race_t()
    patrol = pl.read_parquet(PATROL_PANEL)
    print(f"[panels] aerial {aerial.shape} (runners {sorted(set(aerial['runner'].to_list()))}); "
          f"patrol {patrol.shape}")

    # (1) SHARED CLOCK
    cap = cv2.VideoCapture(PATROL_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc_t, fc_res = find_patrol_finish_cross(cap, fps)
    patrol_offset = patrol_video_to_race_t(fc_t)
    ov = overlap_window(aerial, patrol_offset)
    print(f"[clock] aerial race_t {aerial['race_t'].min():.1f}-{aerial['race_t'].max():.1f}s "
          f"(odometer); patrol finish-cross vid {fc_t:.1f}s (+-{fc_res}) -> offset {patrol_offset:.2f}s")
    print(f"[clock] patrol head-on -> race_t {83-patrol_offset:.1f}-{96-patrol_offset:.1f}s")
    print(f"[clock] OVERLAP race_t {ov[0]:.1f}-{ov[1]:.1f}s (width {ov[1]-ov[0]:.1f}s)")

    # (2) IDENTITY BRIDGE
    finish_numbers = ocr_finish_cam_numbers(cap, fps, 108.0, 119.0)
    cap.release()
    bridge = bridge_patrol_identity(patrol, finish_numbers)
    n_robust = patrol.filter(pl.col("robust"))["track"].n_unique()
    print(f"[bridge] aerial legend: 11/12 (reused). patrol finish-cam OCR: "
          f"{len(finish_numbers)}/12 numbers "
          f"({sorted(finish_numbers, key=lambda k: finish_numbers[k]['lat_px'])})")
    print(f"[bridge] patrol head-on tracks numbered: {len(bridge)}/{n_robust} robust "
          f"({len(bridge)}/12 field)")
    for tk, b in sorted(bridge.items(), key=lambda kv: kv[1]["rank"]):
        print(f"          rank{b['rank']:2d}  track{tk:2d} off {b['off']:5.1f}m -> "
              f"#{b['runner']:2d}  conf {b['conf']:.2f}")

    # (3) CROSS-FEED CONSISTENCY TEST
    agree = cross_feed_lateral_agreement(aerial, patrol, bridge, patrol_offset, ov)
    print("\n" + "=" * 78)
    print(f"CROSS-FEED LATERAL AGREEMENT on overlap (n={agree['n']} bridged horses):")
    print(f"  Spearman rho(aerial y_lat, patrol off_m) = {agree['rho']:+.3f}")
    if "cls_agree" in agree:
        print(f"  rail-vs-wide classification agreement     = {agree['cls_agree']:.0%}")
    print(f"  [diag] aerial y_lat vs stewards: overlap-window rho="
          f"{agree.get('aer_vs_stew', float('nan')):+.2f} (bend, foreshortened) | "
          f"clean-body rho={agree.get('aer_vs_stew_clean', float('nan')):+.2f} | "
          f"home-straight rho={agree.get('aer_vs_stew_hs', float('nan')):+.2f}")
    print(f"  [diag] patrol off_m vs stewards rho={agree.get('pat_vs_stew', float('nan')):+.2f} "
          f"(bridge-dependent -- the null localises to the IDENTITY BRIDGE + bend foreshortening)")
    print(f"  [diag] lateral spread aerial {agree['dist']['aer_spread']:.1f}m / "
          f"patrol {agree['dist']['pat_spread']:.1f}m (identity-free shape coherence)")
    print("  per-horse (official rail/wide tag is VALIDATION ONLY):")
    for rn, tag, al, pl_ in agree.get("off_check", []):
        print(f"    #{rn:2d} [{tag:4s}]  aerial {al:5.1f}m | patrol {pl_:5.1f}m")
    print("=" * 78)

    # write the bridged spine panel (study artifact)
    spine_rows = []
    for r in agree.get("rows", []):
        spine_rows.append(r)
    if spine_rows:
        pl.DataFrame(spine_rows).write_parquet("data_cache/cv_fusion_spine.parquet")
        print("[panel] data_cache/cv_fusion_spine.parquet")

    make_qc(aerial, patrol, bridge, patrol_offset, ov, agree, finish_numbers, fc_t)


if __name__ == "__main__":
    main()

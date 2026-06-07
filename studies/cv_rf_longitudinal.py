"""RF / RACE-FILM as the 4th fusion feed — per-horse LONGITUDINAL backbone + finishing MARGINS.

Task #89 (study only). The RF/race-film (`data_cache/video/full_20260603_01.mp4`) is the real
side-on broadcast feed. It is the natural source for the two things the 3-feed fusion smoother
(0084) found UNOBSERVED:

  STEP 1 — IDENTITY. Near the finish the camera is side-on at ~real scale, so saddlecloth NUMBERS
    OCR (RapidOCR, dual-scale) on the horses themselves (unlike the from-behind mid-race body and
    unlike the stylised AVV). Silk colour as a backup descriptor. Report the bound fraction.

  STEP 2 — FINISH-LINE MARGINS (the cleanest, highest-value deliverable). Near the line the camera
    is side-on == photo-finish geometry: a horse's HORIZONTAL pixel position is its along-track
    position. At a frame near the winner's line-cross, the horizontal pixel separation between two
    numbered horses, scaled by a known length (a horse body ~2.4m, measured in-frame from the
    detected horse blobs), is the FINISHING MARGIN in lengths/metres. This is pan-ROBUST because it
    is a WITHIN-FRAME relative measurement. VALIDATE vs the official chart LBW (lengths-behind-
    winner) + finishing order. This margins distribution IS the S3 likelihood surface (#60).

  STEP 3 — per-horse LONGITUDINAL position over race_t (the fusion backbone). The bottom LEGEND
    STRIP is a rendered running-ORDER data overlay (saddlecloth + silk + name, left->right ==
    lead->back) — a dense per-horse ORDER feed over the whole race (a data overlay, ~= the official
    chart, like the aerial legend; non-novel but it IS the per-horse order the fusion lacked). Where
    the horses are legible (the finish window) we ALSO recover per-horse RELATIVE along-track GAPS
    from horizontal pixel separation, anchored to the leader's known arclen(race_t) from the official
    sections. Honest about pan-induced uncertainty: absolute per-horse arclen across the whole race
    from horse pixels is NOT reliably recoverable (mid-race horses are tiny + from behind); the
    backbone is the finish-window per-horse gaps + the legend order.

Shared clock (0077): race_t = RF_vid_t - 1.34s. Winner (#3 FAMILY FORTUNE) crosses race_t 98.85s
(1:38.85) == RF_vid_t ~100.19s.

DISCIPLINE: study-only; new code here; do NOT edit src/hkjc; do NOT mutate existing parquets (writes
a NEW one). Official chart (LBW margins / order / sections) is VALIDATION ONLY, never a model input.
"""

from __future__ import annotations

import os
import re
import sys

import cv2
import numpy as np

VIDEO = "data_cache/video/full_20260603_01.mp4"
PANEL_OUT = "data_cache/cv_rf_longitudinal.parquet"
FIG = "data_cache/figs/cv_rf_qc.png"

# --- shared clock (0077) ---
RF_OFFSET = 1.34  # race_t = RF_vid_t - 1.34
# official sections [27.65,23.30,23.75,24.15] cumulative time at 0/412.5/825/1237.5/1650 m
SEC_T = np.array([0.0, 27.65, 50.95, 74.70, 98.85])
SEC_S = np.array([0.0, 412.5, 825.0, 1237.5, 1650.0])  # 1650m / 4 even splits

# winner crosses the line at race_t 98.85 -> RF_vid 100.19
WINNER_CROSS_VID = 98.85 + RF_OFFSET
# the clean side-on finish window (horses large, numbers legible, before the post-pass blur)
FIN_T0, FIN_T1, FIN_DT = 96.5, 100.5, 0.08  # vid seconds, frame-step (~50 frames)

# OCR runs on the TRACK BAND crop only (rows 0.30-0.82) — skips scenery/crowd/overlay + is faster.
BAND_Y0, BAND_Y1 = 0.30, 0.82
OCR_SCALES = (1.6, 2.4)
OCR_CONF = 0.40

# physical scale
LEN_M = 2.4  # a thoroughbred body length (nose-to-tail) in metres (HKJC convention ~2.4m)

# Official field (results.html) — NAMES/draw are PRE-RACE facts, reference only.
FIELD = {
    3: "FAMILY FORTUNE", 12: "THE WAY WE WIN", 6: "TELECOM POWER", 2: "SETANTA",
    9: "WAH MAY WAI WAI", 8: "MANYTHANKS FOREVER", 11: "CASA LEGEND", 4: "PRINCE ALEX",
    7: "ZETTA FORCE", 10: "SMART TRIO", 5: "HAPPY BUDDIES", 1: "MASTER OF HUMOR",
}
# normalised (uppercased, spaces removed) english horse-name -> runner number, for legend NAME
# OCR -> number bridge (the saddlecloth number box in the legend is too small to OCR; the english
# NAME OCRs cleanly and maps 1:1 to the runner via this fixed pre-race fact).
NAME2NUM = {v.replace(" ", "").upper(): k for k, v in FIELD.items()}
# Official chart (results.html) — VALIDATION ONLY: finishing order + LBW (lengths behind winner).
# LBW is the CUMULATIVE margin to the winner in lengths. 'N' = neck ~0.3L.
OFFICIAL_ORDER = [3, 12, 6, 2, 9, 8, 11, 4, 7, 10, 5, 1]
OFFICIAL_LBW = {  # lengths behind winner #3
    3: 0.0, 12: 0.3, 6: 1.5, 2: 2.25, 9: 2.25, 8: 4.25, 11: 4.5,
    4: 4.75, 7: 5.5, 10: 6.25, 5: 7.0, 1: 10.5,
}
OFFICIAL_TIME = {  # finish time (s) for the leaders, from the chart
    3: 98.85, 12: 98.90, 6: 99.09, 2: 99.19, 9: 99.20, 8: 99.53,
}


# ----------------------------------------------------------------------------
# clock helpers
# ----------------------------------------------------------------------------
def vid_to_race_t(vid_t: float) -> float:
    return vid_t - RF_OFFSET


def race_t_to_arclen(race_t: float) -> float:
    return float(np.interp(race_t, SEC_T, SEC_S))


def leader_velocity_at_line() -> float:
    """Leader (winner) ground speed at the line, m/s, from the official last-section pace."""
    # last section 24.15s over 412.5m -> ~17.08 m/s average; near the line the winner is
    # decelerating slightly but ~16.7 m/s is the standard line speed. Use the section average.
    return SEC_S[-1] - SEC_S[-2], SEC_T[-1] - SEC_T[-2]  # (dm, dt)


# ----------------------------------------------------------------------------
# horse / number detection in the finish window
# ----------------------------------------------------------------------------
def detect_horses(fr: np.ndarray) -> list[dict]:
    """Detect horse+jockey blobs (dark horse body below a saturated silk) on the turf.

    Returns per blob: cx, cy (centroid), bbox (x0,y0,x1,y1), area, and a horizontal extent
    (used as the body-length pixel scale). Side-on, large horses near the finish.
    """
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # turf = green, moderate sat. Horses = dark (low V) OR saturated non-green silks/tack.
    turf = (hh > 30) & (hh < 95) & (ss > 30) & (vv > 40)
    dark = vv < 80
    silk = (ss > 110) & (vv > 70) & (~turf)
    fg = ((dark | silk) & (~turf)).astype(np.uint8)
    # restrict to the track band (exclude top scenery / crowd, bottom overlay)
    band = np.zeros_like(fg)
    band[int(h * 0.30): int(h * 0.82), :] = 1
    fg = fg & band
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ct = cv2.connectedComponentsWithStats(fg, 8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 1200 or bw < 35 or bh < 25:
            continue
        if bw > w * 0.5:  # merged blob, skip (too wide to be one horse)
            continue
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        out.append({
            "cx": float(ct[i][0]), "cy": float(ct[i][1]),
            "x0": bx, "y0": by, "x1": bx + bw, "y1": by + bh,
            "bw": bw, "bh": bh, "area": area,
        })
    return out


def ocr_numbers(fr: np.ndarray, ocr) -> list[dict]:
    """OCR saddlecloth numbers on the horses. Crops to the TRACK BAND (faster + drops the bottom
    legend overlay and top scenery), dual-scale. Returns dicts in FULL-FRAME coords."""
    h = fr.shape[0]
    y0 = int(h * BAND_Y0)
    crop = fr[y0: int(h * BAND_Y1), :]
    out = []
    for scale in OCR_SCALES:
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        for box, txt, conf in res:
            txt = txt.strip()
            if not (re.fullmatch(r"\d{1,2}", txt) and conf > OCR_CONF and 1 <= int(txt) <= 12):
                continue
            cx = float(np.mean([p[0] for p in box])) / scale
            cy = float(np.mean([p[1] for p in box])) / scale + y0
            bh = (max(p[1] for p in box) - min(p[1] for p in box)) / scale
            out.append({"num": int(txt), "cx": cx, "cy": cy, "conf": float(conf), "h": bh})
    return out


def in_legend_strip(cy: float, frame_h: int) -> bool:
    """OCR hits in the bottom legend strip are the DATA OVERLAY, not the horse — exclude."""
    return cy > frame_h * 0.82


# ----------------------------------------------------------------------------
# STEP 2 — finish-line margins by within-frame horizontal pixel separation
# ----------------------------------------------------------------------------
def measure_margins(cap, fps, ocr):
    """Pool number detections across the finish window; per horse get its horizontal pixel
    position (leading edge) and a per-frame body-length scale; convert WITHIN-FRAME relative
    separation to lengths. The winner is the reference (margin 0).

    Pan-robust: the per-FRAME relative pixel gap / per-frame body-length scale is independent
    of where the camera is pointing. We aggregate the per-frame relative-to-winner gaps.
    """
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # per (frame) -> {num: leading_edge_x}, plus the frame's body-length scale (px/length)
    frame_recs = []  # list of (vid_t, {num: x_lead}, body_px, {num: cx})
    for vid_t in np.arange(FIN_T0, FIN_T1, FIN_DT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(vid_t * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        horses = detect_horses(fr)
        if len(horses) < 2:
            continue
        dets = [d for d in ocr_numbers(fr, ocr) if not in_legend_strip(d["cy"], h)]
        if len(dets) < 2:
            continue
        # de-dup numbers within the frame (keep highest conf per number)
        best = {}
        for d in dets:
            if d["num"] not in best or d["conf"] > best[d["num"]]["conf"]:
                best[d["num"]] = d
        # bind each numbered det to the nearest horse blob -> body length px + leading-edge x.
        # The horse runs RIGHT->LEFT at the finish (winner at left), so the LEADING edge is the
        # blob's LEFT edge (smaller x). Direction of travel verified from the frames (#3 at left).
        # per-frame body-length scale (px per 2.4m). A clean SIDE-ON single horse blob has a
        # width:height aspect ~1.6-3.0 (long body). Merged multi-horse blobs are much wider;
        # exclude them. Use the MEDIAN width of the aspect-clean blobs as the per-frame scale.
        clean = [hb for hb in horses if 1.3 <= hb["bw"] / max(hb["bh"], 1) <= 3.2]
        if not clean:
            clean = horses
        body_px = float(np.median([hb["bw"] for hb in clean]))
        # The saddlecloth number sits at a consistent body location, so the OCR number CENTROID x
        # is a stable per-horse along-track marker; its WITHIN-FRAME separation is the gap.
        numcx = {num: float(d["cx"]) for num, d in best.items()}
        if len(numcx) >= 2:
            frame_recs.append((float(vid_t), numcx, body_px))
    return frame_recs


def margins_from_frames(frame_recs):
    """Aggregate per-frame WITHIN-FRAME gaps (relative to the winner #3) into a length margin.

    For each frame that contains the winner (#3) AND horse k, gap_px = cx_k - cx_3 (k is BEHIND
    the winner so cx_k > cx_3 -> positive; horses run RIGHT->LEFT at the line). length =
    gap_px / body_px.

    The official LBW is the at-the-line margin, and the gap OPENS as the field crosses (the camera
    tracks the leaders and tends to lose back-markers from frame BEFORE the gap fully opens). So a
    plain median UNDER-reads the back markers. We take the OPEN-GAP estimate: the upper-quartile
    (75th pct) per-horse gap == the most-spread co-framed moment, the best proxy for the line
    margin; sd is the inter-quartile spread. Require n>=2 to drop single-frame OCR flukes.
    """
    from collections import defaultdict

    per = defaultdict(list)  # num -> [length gaps to winner]
    for vid_t, numcx, body_px in frame_recs:
        if 3 not in numcx or body_px <= 0:
            continue
        x3 = numcx[3]
        for num, x in numcx.items():
            per[num].append((x - x3) / body_px)
    res = {}
    for num, vals in per.items():
        v = np.array(vals)
        if num != 3 and len(v) < 2:
            continue  # drop single-frame flukes (keep the winner reference)
        est = 0.0 if num == 3 else float(np.percentile(v, 75))  # open-gap (line) proxy
        res[num] = {
            "margin_len": max(0.0, est),
            "sd_len": float(np.subtract(*np.percentile(v, [75, 25])) / 1.35)
            if len(v) > 2 else float("nan"),
            "n": int(len(v)),
        }
    return res


# ----------------------------------------------------------------------------
# STEP 3 — legend-strip running ORDER over the whole race (the backbone) + finish gaps
# ----------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def _match_name(txt: str) -> int | None:
    """Map an OCR'd legend english NAME to a runner number (fuzzy: exact, then best Jaccard of
    char-bigrams above a threshold — robust to RapidOCR dropping/merging a letter)."""
    q = _norm_name(txt)
    if len(q) < 5:
        return None
    if q in NAME2NUM:
        return NAME2NUM[q]

    def bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}

    qb = bigrams(q)
    best, bj = None, 0.0
    for name, num in NAME2NUM.items():
        nb = bigrams(name)
        j = len(qb & nb) / max(1, len(qb | nb))
        if j > bj:
            bj, best = j, num
    return best if bj >= 0.6 else None


def read_legend_order(cap, fps, ocr, t0=4.0, t1=95.0, dt=2.0):
    """Read the bottom LEGEND STRIP running order over the race. The strip is a rendered data
    overlay (~ official chart) listing the lead group left->right == lead->back; each entry is a
    coloured box with a (too-small-to-OCR) saddlecloth number + the english NAME, which OCRs
    cleanly. We read the NAMES (left->right) and bridge to runner numbers via NAME2NUM (a fixed
    pre-race fact). Returns a list of (vid_t, [numbers left->right]).
    """
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    seq = []
    for vid_t in np.arange(t0, t1, dt):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(vid_t * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        # the name band: rows ~0.855-0.93 (number box above, english name centred under it)
        strip = fr[int(h * 0.855): int(h * 0.93), : int(w * 0.92)]
        big = cv2.resize(strip, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        hits = []  # (cx, num)
        for box, txt, conf in res:
            num = _match_name(txt) if conf > 0.40 else None
            if num is not None:
                cx = float(np.mean([p[0] for p in box]))
                hits.append((cx, num))
        hits.sort()
        order = []
        for _, num in hits:
            if num not in order:
                order.append(num)
        if len(order) >= 3:
            seq.append((float(vid_t), order))
    return seq


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rapidocr_onnxruntime import RapidOCR

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = RapidOCR()

    print(f"RF race-film {VIDEO}  fps={fps}  winner-cross vid~{WINNER_CROSS_VID:.2f}s")

    # ============ STEP 1 + 2 — finish-window OCR identity + margins ============
    print("\n=== STEP 1/2 — finish-window detection + saddlecloth OCR + margins ===")
    frame_recs = measure_margins(cap, fps, ocr)
    seen = set()
    for _, numcx, _ in frame_recs:
        seen |= set(numcx)
    print(f"finish-window frames used: {len(frame_recs)}; numbers OCR'd on horses: "
          f"{sorted(seen)} ({len(seen)}/12)")

    margins = margins_from_frames(frame_recs)
    # order by margin (winner first)
    ordered = sorted(margins, key=lambda n: margins[n]["margin_len"])
    print("\n  RF finishing order + margins (lengths behind winner #3):")
    print(f"  {'rk':>2} {'num':>3} {'name':<18} {'RF_len':>7} {'sd':>5} {'n':>3} "
          f"{'off_LBW':>7} {'off_pos':>7}")
    rf_rank = {n: i + 1 for i, n in enumerate(ordered)}
    for i, num in enumerate(ordered):
        m = margins[num]
        off_lbw = m["margin_len"] - OFFICIAL_LBW.get(num, float("nan"))
        off_pos = rf_rank[num] - (OFFICIAL_ORDER.index(num) + 1 if num in OFFICIAL_ORDER else -1)
        print(f"  {i+1:>2} #{num:>2} {FIELD.get(num,'?'):<18} {m['margin_len']:>7.2f} "
              f"{m['sd_len']:>5.2f} {m['n']:>3} {off_lbw:>+7.2f} {off_pos:>+7d}")

    # ---- VALIDATION: RF margins vs official LBW ----
    print("\n=== VALIDATION — RF margins vs official chart (VALIDATION ONLY) ===")
    common = [n for n in ordered if n in OFFICIAL_LBW]
    rf_m = np.array([margins[n]["margin_len"] for n in common])
    off_m = np.array([OFFICIAL_LBW[n] for n in common])
    if len(common) >= 3:
        # Spearman rank corr of order
        from scipy.stats import spearmanr
        rho, _ = spearmanr(rf_m, off_m)
        mae = float(np.mean(np.abs(rf_m - off_m)))
        # linear fit RF = a*official + b (a~1 ideal)
        a, b = np.polyfit(off_m, rf_m, 1)
        # finishing-order agreement (RF rank vs official rank)
        off_rank = {n: OFFICIAL_ORDER.index(n) + 1 for n in common}
        rf_r = [rf_rank[n] for n in common]
        of_r = [off_rank[n] for n in common]
        rho_rank, _ = spearmanr(rf_r, of_r)
        n_adj = sum(1 for i in range(len(common)) for j in range(len(common))
                    if i < j and (rf_r[i] < rf_r[j]) == (of_r[i] < of_r[j]))
        n_pairs = len(common) * (len(common) - 1) // 2
        print(f"  common numbered horses: {common}")
        print(f"  margin MAE = {mae:.2f} lengths;  fit RF = {a:.2f}*official {b:+.2f}")
        print(f"  margin Spearman rho = {rho:+.3f};  ORDER Spearman rho = {rho_rank:+.3f}")
        print(f"  pairwise order agreement = {n_adj}/{n_pairs}")
    else:
        rho = rho_rank = mae = a = b = float("nan")
        print(f"  too few numbered horses ({len(common)}) for validation")

    # ============ STEP 3 — legend running-order backbone ============
    print("\n=== STEP 3 — legend-strip running ORDER over the race (the backbone) ===")
    seq = read_legend_order(cap, fps, ocr)
    print(f"  legend reads: {len(seq)} timepoints (vid {seq[0][0]:.0f}-{seq[-1][0]:.0f}s)"
          if seq else "  legend reads: NONE")
    # show a few
    for vid_t, order in seq[::max(1, len(seq) // 8)]:
        print(f"    race_t {vid_to_race_t(vid_t):5.1f}s  lead->back: {order}")

    # leader arclen anchor: legend leftmost == leader; its arclen = official arclen(race_t)
    # Per-horse arclen-or-gap on the shared grid:
    #  - finish window: per-horse arclen = leader_arclen(race_t) - margin_len*LEN_M (margins fixed
    #    near the line; we report them at the crossing race_t with the measured gap).
    #  - legend order: a per-horse ORDER (rank) backbone over the whole race (gap not metric).
    leader_v_dm, leader_v_dt = leader_velocity_at_line()
    v_line = leader_v_dm / leader_v_dt
    print(f"\n  leader line speed ~{v_line:.2f} m/s ({1.0/(v_line/LEN_M):.3f}s per {LEN_M}m length)")

    # ============ write the panel ============
    cols = {"runner": [], "race_t": [], "arclen_or_gap": [], "margin_len": [], "sd": [],
            "kind": []}
    # (A) finish-window margins, placed at the winner's crossing race_t, as gap-to-leader (metres)
    cross_rt = 98.85
    leader_arc = race_t_to_arclen(cross_rt)  # 1650
    for num in ordered:
        m = margins[num]
        gap_m = m["margin_len"] * LEN_M  # lengths -> metres behind leader
        cols["runner"].append(num)
        cols["race_t"].append(cross_rt)
        cols["arclen_or_gap"].append(leader_arc - gap_m)  # absolute arclen at the line
        cols["margin_len"].append(m["margin_len"])
        cols["sd"].append(m["sd_len"] if not np.isnan(m["sd_len"]) else 0.5)
        cols["kind"].append("finish_margin")
    # (B) legend-order backbone: per timepoint, per horse a RANK -> a coarse gap-to-leader proxy.
    #     We DO NOT fabricate metric gaps mid-race (pan + tiny horses); we record the ORDER as a
    #     rank field carried in arclen_or_gap as NEGATIVE rank (so it is unambiguous vs metric arc).
    for vid_t, order in seq:
        rt = vid_to_race_t(vid_t)
        for rank0, num in enumerate(order):
            cols["runner"].append(num)
            cols["race_t"].append(rt)
            cols["arclen_or_gap"].append(float(-(rank0 + 1)))  # -rank (lead=-1); ORDER, not metric
            cols["margin_len"].append(float("nan"))
            cols["sd"].append(float("nan"))
            cols["kind"].append("legend_order")
    pq.write_table(pa.table(cols), PANEL_OUT)
    print(f"\npanel: {PANEL_OUT} ({len(cols['runner'])} rows; "
          f"{sum(1 for k in cols['kind'] if k=='finish_margin')} margin rows, "
          f"{sum(1 for k in cols['kind'] if k=='legend_order')} legend-order rows)")

    make_qc(ordered, margins, rf_rank, seq, mae, rho, rho_rank, a, b)
    cap.release()
    return {"margins": margins, "order": ordered, "mae": mae, "rho_margin": rho,
            "rho_order": rho_rank, "n_id": len(seen)}


def make_qc(ordered, margins, rf_rank, seq, mae, rho, rho_rank, a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("data_cache/figs", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(17, 10))

    # (i) identity / detection count
    a0 = ax[0, 0]
    a0.axis("off")
    lines = ["STEP 1 — IDENTITY (finish-window saddlecloth OCR)\n"]
    seen = sorted(margins)
    lines.append(f"  numbers OCR'd on horses: {seen}")
    lines.append(f"  bound fraction: {len(seen)}/12 = {len(seen)/12:.0%}\n")
    lines.append("STEP 2 — MARGINS validation vs official LBW")
    lines.append(f"  margin MAE   = {mae:.2f} lengths")
    lines.append(f"  margin rho   = {rho:+.3f}")
    lines.append(f"  ORDER rho    = {rho_rank:+.3f}")
    lines.append(f"  fit RF = {a:.2f}*official {b:+.2f}  (slope 1 ideal)")
    a0.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    a0.set_title("(i) identity + margins headline")

    # (ii) finish-margins bar — RF vs official
    a1 = ax[0, 1]
    common = [n for n in ordered if n in OFFICIAL_LBW]
    x = np.arange(len(common))
    rf_v = [margins[n]["margin_len"] for n in common]
    of_v = [OFFICIAL_LBW[n] for n in common]
    a1.bar(x - 0.2, rf_v, 0.4, label="RF (pixel sep)", color="#1f77b4")
    a1.bar(x + 0.2, of_v, 0.4, label="official LBW", color="#ff7f0e")
    a1.set_xticks(x)
    a1.set_xticklabels([f"#{n}" for n in common])
    a1.set_ylabel("lengths behind winner")
    a1.set_title("(ii) FINISH MARGINS — RF vs official chart")
    a1.legend()
    a1.grid(alpha=0.3, axis="y")

    # (iii) RF margin vs official scatter
    a2 = ax[1, 0]
    a2.scatter(of_v, rf_v, c="#2ca02c", s=60)
    for n, ox, ry in zip(common, of_v, rf_v):
        a2.annotate(f"#{n}", (ox, ry), fontsize=9)
    lim = max(max(of_v), max(rf_v)) * 1.1
    a2.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
    a2.plot([0, lim], [b, a * lim + b], "r-", alpha=0.6, label=f"fit {a:.2f}x{b:+.2f}")
    a2.set_xlabel("official LBW (lengths)")
    a2.set_ylabel("RF margin (lengths)")
    a2.set_title("(iii) RF vs official margins")
    a2.legend()
    a2.grid(alpha=0.3)

    # (iv) legend-order backbone over race_t (per-horse rank tracks)
    a3 = ax[1, 1]
    from collections import defaultdict
    tracks = defaultdict(list)
    for vid_t, order in seq:
        rt = vid_to_race_t(vid_t)
        for rank0, num in enumerate(order):
            tracks[num].append((rt, rank0 + 1))
    for num, pts in tracks.items():
        pts.sort()
        rts = [p[0] for p in pts]
        rks = [p[1] for p in pts]
        a3.plot(rts, rks, "-o", ms=2, lw=1, alpha=0.8)
        a3.text(rts[-1], rks[-1], f"#{num}", fontsize=8, weight="bold")
    a3.invert_yaxis()
    a3.set_xlabel("race_t (s)")
    a3.set_ylabel("running rank (1=lead) from legend strip")
    a3.set_title("(iv) LEGEND ORDER backbone (lead group; data overlay)")
    a3.grid(alpha=0.3)

    fig.suptitle("RF / race-film 4th feed — identity + finish MARGINS + longitudinal backbone "
                 "(study #89, ST 2026-06-03 R1)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    plt.close(fig)
    print(f"QC figure: {FIG}")


if __name__ == "__main__":
    sys.exit(0 if main() else 0)

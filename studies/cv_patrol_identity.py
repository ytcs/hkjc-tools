"""Patrol head-on lateral IDENTITY by SILK COLOUR (study only — task #86).

Gives the validated PATROL head-on lateral (0076 / #83) an IDENTITY, by SILK COLOUR,
via a number<->silk LUT (no temporal bridge across the camera cut — 0077 showed that is
unreliable). Two stages:

  STAGE A — BUILD the silk->number LUT from the PATROL FINISH-CAM (vid t>=105s, the feed
    cuts to a finish-line tracking camera where saddlecloth NUMBERS are legible). For each
    numbered horse: OCR the number (RapidOCR, pooled across frames) and sample its JOCKEY
    SILK colour (the saturated torso patch the jockey wears, located relative to the OCR'd
    number box). Pool per-number; the LUT entry is the robust (saturated-cluster) LAB.

  STAGE B — LABEL the anonymous head-on tracks (0076's panel cv_patrol_lateral.parquet).
    Each head-on track carries a clean, camera-FACING silk colour. Re-sample each track's
    silk LAB signature with the SAME _silk_mask machinery as cv_patrol_lateral, then assign
    the nearest LUT number one-to-one (Hungarian on a hue-weighted LAB distance). Report
    per-assignment confidence (margin to the 2nd-best) + any ties.

OUTPUT: data_cache/cv_patrol_numbered.parquet (runner, race_t, arclen, y_lat_m, confidence)
over the head-on window. race_t / arclen via the 0077 fusion-spine anchor
(race_t = patrol_vid_t - 18.65s; arclen from the official arclen<->time curve).

VALIDATE (official = validation only): per-number lateral vs stewards' Comments-on-Running —
rail #12 (wire-to-wire rail) / #8 (inside) TIGHT; wide #2 (wide barrier) / #6 (caught wide)
WIDE. The silk->number->trip chain is correct iff these read right.

DISCIPLINE: study-only; imports cv_patrol_lateral READ-ONLY; mutates no existing parquet;
official data is VALIDATION/reference only (the silk<->number map is a fixed PRE-RACE fact,
NOT an outcome; finish order / running positions are NOT used as inputs).
"""

from __future__ import annotations

import re
import sys

import cv2
import numpy as np

sys.path.insert(0, "studies")
from cv_patrol_lateral import (  # noqa: E402
    T0,
    T1,
    VIDEO,
    _silk_mask,
    _turf_hue,
    detect_jockeys,
)

# --- finish-cam window (the feed cuts to a side-on finish tracking camera) ---
FIN_T0, FIN_T1, FIN_DT = 102.5, 123.0, 0.15
OCR_SCALES = (1.6, 2.2)  # dual-scale catches small/blurred back-marker numbers
OCR_CONF = 0.45
LUT_MIN_HITS = 2  # admit a number to the LUT only with >=2 OCR detections (drop flukes)

# --- shared clock (0077 fusion spine, VALIDATION-only anchoring) ---
PATROL_OFFSET = 18.65  # race_t = patrol_vid_t - offset (winner finish-cross @ vid 117.5s)
# official sections [27.65,23.30,23.75,24.15] cumulative time at 0/412.5/825/1237.5/1650 m
SEC_T = np.array([0.0, 27.65, 50.95, 74.70, 98.85])
SEC_S = np.array([0.0, 412.5, 825.0, 1237.5, 1650.0])  # 1650m / 4 even splits

PANEL_IN = "data_cache/cv_patrol_lateral.parquet"
PANEL_OUT = "data_cache/cv_patrol_numbered.parquet"
FIG = "data_cache/figs/cv_patrol_numbered_qc.png"

# Official field (results.html) — NAMES/draw are reference only; finish order NOT an input.
FIELD = {
    3: "FAMILY FORTUNE", 12: "THE WAY WE WIN", 6: "TELECOM POWER", 2: "SETANTA",
    9: "WAH MAY WAI WAI", 8: "MANYTHANKS FOREVER", 11: "CASA LEGEND", 4: "PRINCE ALEX",
    7: "ZETTA FORCE", 10: "SMART TRIO", 5: "HAPPY BUDDIES", 1: "MASTER OF HUMOR",
}
# Stewards' Comments-on-Running (VALIDATION-only): rail/inside vs wide.
STEW_TIGHT = {12: "wire-to-wire on the rail (pos 1-1-1)", 8: "inside"}
STEW_WIDE = {2: "from a wide barrier, shifted across", 6: "caught wide in the early stages"}


NBINS = 18  # hue histogram bins (10 deg each over OpenCV's 0..180 hue)


# ----------------------------------------------------------------------------
# SILK DESCRIPTOR — saturation-weighted HUE HISTOGRAM (robust to which fragment
# dominates; LAB medians collapse to red on these overlapping low-res silks).
# A silk is multi-colour; the histogram keeps ALL its hues (blue body + red
# sleeves stay distinct), so two silks separate even when one hue is shared.
# ----------------------------------------------------------------------------
def _hue_hist(hh: np.ndarray, ss: np.ndarray, vv: np.ndarray) -> np.ndarray | None:
    """Sat-weighted hue histogram over silk pixels (already masked). L1-normalised."""
    if hh.size < 8:
        return None
    sel = (ss > 80) & (vv > 60)
    if sel.sum() < 8:
        return None
    bins = (hh[sel].astype(int) * NBINS // 180) % NBINS
    w = ss[sel].astype(float)
    hist = np.bincount(bins, weights=w, minlength=NBINS).astype(float)
    s = hist.sum()
    return hist / s if s > 0 else None


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Circular EMD between two normalised hue histograms (silks are ABSOLUTE hues, so
    NOT rotation-invariant — we do not min over shifts, which would map blue onto red).
    The two ring directions give the wrap-around EMD; take the smaller."""
    ca, cb = np.cumsum(a), np.cumsum(b)
    fwd = float(np.abs(ca - cb).sum())
    bwd = float(np.abs((1 - ca) - (1 - cb)).sum())
    return min(fwd, bwd)


def _lab_swatch(hh, ss, vv) -> np.ndarray:
    """A representative LAB colour for the QC swatch (modal-hue median pixel)."""
    sel = (ss > 80) & (vv > 60)
    if sel.sum() < 4:
        return np.array([128.0, 128.0, 128.0])
    bins = (hh[sel].astype(int) * NBINS // 180) % NBINS
    mode = np.bincount(bins, weights=ss[sel].astype(float), minlength=NBINS).argmax()
    msel = bins == mode
    h = int(np.median(hh[sel][msel]))
    s = int(np.median(ss[sel][msel]))
    v = int(np.median(vv[sel][msel]))
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


# ----------------------------------------------------------------------------
# STAGE A — finish-cam number<->silk LUT
# ----------------------------------------------------------------------------
def _silk_blobs(fr: np.ndarray):
    """Saturated non-turf silk blobs: (cx, cy, hue-hist, lab-swatch, area)."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    turf = (hh > 30) & (hh < 90) & (s > 25) & (v > 40)
    silk = ((s > 105) & (v > 75) & (~turf)).astype(np.uint8)
    silk = cv2.morphologyEx(silk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, st, ct = cv2.connectedComponentsWithStats(silk, 8)
    out = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < 35 or area > 3500:
            continue
        bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        mask = lbl[by : by + bh, bx : bx + bw] == i
        hp = hh[by : by + bh, bx : bx + bw][mask]
        sp = s[by : by + bh, bx : bx + bw][mask]
        vp = v[by : by + bh, bx : bx + bw][mask]
        hist = _hue_hist(hp, sp, vp)
        if hist is None:
            continue
        out.append((float(ct[i][0]), float(ct[i][1]), hist, _lab_swatch(hp, sp, vp), area))
    return out


def build_lut(cap, fps, ocr) -> dict[int, dict]:
    """OCR saddlecloth numbers + bind each to its nearest ABOVE silk blob, pooled.

    Per number we accumulate the silk blob's HUE HISTOGRAM (sat-weighted), then average
    across frames → a robust multi-colour silk descriptor (a LAB median collapses to red
    on these overlapping low-res silks; the histogram keeps each silk's distinct hues).
    """
    from collections import defaultdict

    hists: dict[int, list[np.ndarray]] = defaultdict(list)
    swatches: dict[int, list[np.ndarray]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    maxconf: dict[int, float] = defaultdict(float)
    for t in np.arange(FIN_T0, FIN_T1, FIN_DT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        blobs = _silk_blobs(fr)
        for scale in OCR_SCALES:
            big = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(big)
            if not res:
                continue
            for box, txt, conf in res:
                txt = txt.strip()
                if not (re.fullmatch(r"\d{1,2}", txt) and conf > OCR_CONF and 1 <= int(txt) <= 12):
                    continue
                num = int(txt)
                cx = float(np.mean([p[0] for p in box])) / scale
                cy = float(np.mean([p[1] for p in box])) / scale
                boxh = (max(p[1] for p in box) - min(p[1] for p in box)) / scale
                counts[num] += 1
                maxconf[num] = max(maxconf[num], float(conf))
                # nearest SATURATED silk blob ABOVE the number (jockey torso) within reach
                best, bd = None, 1e9
                for bx, by, hist, sw, area in blobs:
                    if by > cy - 0.2 * boxh or by < cy - 8 * boxh:
                        continue
                    if abs(bx - cx) > 2.5 * boxh:
                        continue
                    d = abs(bx - cx) + 0.4 * abs(by - cy)
                    if d < bd:
                        bd, best = d, (hist, sw)
                if best is not None:
                    hists[num].append(best[0])
                    swatches[num].append(best[1])
    lut = {}
    for num, hs in hists.items():
        # admit only numbers with enough OCR support AND a silk read (drop OCR flukes)
        if counts[num] < LUT_MIN_HITS or len(hs) < 1:
            continue
        hist = np.mean(np.array(hs), 0)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        lut[num] = {
            "hist": hist,
            "lab": np.median(np.array(swatches[num]), 0),
            "n_ocr": counts[num],
            "n_silk": len(hs),
            "maxconf": maxconf[num],
        }
    return lut


# ----------------------------------------------------------------------------
# STAGE B — head-on track silk signatures + assignment
# ----------------------------------------------------------------------------
def headon_track_silks(cap, fps, panel) -> dict[int, dict]:
    """Re-sample each robust head-on track's camera-facing silk HUE HISTOGRAM (signature).

    The 0076 panel gives, per track, (t, off_m, depth). At each frame we sample EVERY
    detected jockey's silk hue histogram (sat-weighted, _silk_mask machinery) tagged with
    its off_m+depth, then assign per-track by nearest (off, depth). The signature is the
    mean histogram — robust to which fragment dominates (a LAB median collapses to red).
    """
    from collections import defaultdict

    # group panel rows by track and by frame-time
    by_track: dict[int, list] = defaultdict(list)
    for tr, t, off, dep, robust in zip(
        panel["track"], panel["t"], panel["off_m"], panel["depth"], panel["robust"]
    ):
        by_track[int(tr)].append((round(float(t), 3), float(off), float(dep), bool(robust)))

    # we need image positions; recompute detections per frame ONCE and match to off via the
    # same homography the panel used. Simpler + robust: at each frame, sample the silk LAB of
    # EVERY detected jockey and tag with its off_m; then assign per track by off_m+depth+t.
    times = sorted({round(t, 3) for t in panel["t"]})
    frame_dets: dict[float, list] = {}
    # rebuild the homography exactly as the driver does (read-only reuse)
    from cv_patrol_lateral import build_homography, fit_fixed_rail, project

    rail = fit_fixed_rail(cap, fps)
    # mpp: reuse the driver's calibrate (mid-window field span ~15 m)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(((T0 + T1) / 2) * fps))
    ok, fr = cap.read()
    tl0 = _turf_hue(fr)
    js0 = detect_jockeys(fr, tl0)
    cxs = np.array([j["cx"] for j in js0])
    ygs = np.array([j["y_ground"] for j in js0])
    perp = cxs - (rail.a * ygs + rail.b)
    span_px = float(np.max(perp) - np.min(perp)) if len(perp) else 1.0
    mpp_near = (span_px / 15.0) if span_px > 0 else 60.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hmat = build_homography(rail, h, w, mpp_near)

    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        tl = _turf_hue(fr)
        js = detect_jockeys(fr, tl)
        m = _silk_mask(fr, tl)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        pts = np.array([[j["cx"], j["y_ground"]] for j in js], float)
        proj = project(hmat, pts) if len(pts) else np.empty((0, 2))
        dets = []
        for j, p in zip(js, proj):
            # silk hue-hist = saturated silk pixels in the jockey's silk bbox
            x0 = int(max(0, j["cx"] - j["bw"]))
            x1 = int(min(w, j["cx"] + j["bw"]))
            y0 = int(max(0, j["cy"] - j["bh"]))
            y1 = int(min(h, j["cy"] + j["bh"] * 0.6))
            sm = m[y0:y1, x0:x1] > 0
            if sm.sum() < 8:
                continue
            hp = hsv[y0:y1, x0:x1, 0][sm]
            sp = hsv[y0:y1, x0:x1, 1][sm]
            vp = hsv[y0:y1, x0:x1, 2][sm]
            hist = _hue_hist(hp, sp, vp)
            if hist is None:
                continue
            dets.append((float(p[0]), float(p[1]), hist, _lab_swatch(hp, sp, vp)))
        frame_dets[t] = dets

    # assign each panel detection to the nearest frame det by (off, depth)
    sig: dict[int, list] = defaultdict(list)
    sw: dict[int, list] = defaultdict(list)
    for tr, rows in by_track.items():
        if not any(r[3] for r in rows):  # robust tracks only
            continue
        for t, off, dep, _rob in rows:
            dets = frame_dets.get(t, [])
            best, bd = None, 1e9
            for doff, ddep, hist, lab in dets:
                d = abs(doff - off) + 3.0 * abs(ddep - dep)
                if d < bd:
                    bd, best = d, (hist, lab)
            if best is not None and bd < 1.5:
                sig[tr].append(best[0])
                sw[tr].append(best[1])
    out = {}
    for tr, hs in sig.items():
        if len(hs) < 5:
            continue
        hist = np.mean(np.array(hs), 0)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        out[tr] = {"hist": hist, "lab": np.median(np.array(sw[tr]), 0), "n": len(hs)}
    return out


def assign(track_sig: dict[int, dict], lut: dict[int, dict]):
    """One-to-one Hungarian assignment track->number on circular hue-histogram EMD."""
    from scipy.optimize import linear_sum_assignment

    tracks = sorted(track_sig)
    nums = sorted(lut)
    cost = np.zeros((len(tracks), len(nums)))
    for i, tr in enumerate(tracks):
        for j, nm in enumerate(nums):
            cost[i, j] = _hist_dist(track_sig[tr]["hist"], lut[nm]["hist"])
    ri, ci = linear_sum_assignment(cost)
    out = []
    for i, j in zip(ri, ci):
        tr, nm = tracks[i], nums[j]
        d = cost[i, j]
        # confidence = margin to this track's 2nd-best number (gap), normalised
        row = np.sort(cost[i])
        margin = float(row[1] - row[0]) if len(row) > 1 else 0.0
        out.append({"track": tr, "runner": nm, "dist": float(d), "margin": margin})
    return out, tracks, nums, cost


# ----------------------------------------------------------------------------
# race_t / arclen
# ----------------------------------------------------------------------------
def vid_to_race_t(vid_t: float) -> float:
    return vid_t - PATROL_OFFSET


def race_t_to_arclen(race_t: float) -> float:
    return float(np.interp(race_t, SEC_T, SEC_S))


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------
def main():
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq
    from rapidocr_onnxruntime import RapidOCR

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ocr = RapidOCR()

    print("=== STAGE A — finish-cam number<->silk LUT ===")
    lut = build_lut(cap, fps, ocr)
    nums = sorted(lut)
    print(f"OCR'd {len(nums)}/12 numbers: {nums}")
    for nm in nums:
        e = lut[nm]
        bgr = cv2.cvtColor(e["lab"].astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
        nm_label = FIELD.get(nm, "?")
        top = (-e["hist"]).argsort()[:3]
        hues = [f"{int(b*180//NBINS)}deg:{e['hist'][b]:.2f}" for b in top]
        print(f"  #{nm:>2} {nm_label:<18} RGB=({bgr[2]},{bgr[1]},{bgr[0]}) "
              f"top-hues[{','.join(hues)}] n_ocr={e['n_ocr']} n_silk={e['n_silk']}")

    print("\n=== STAGE B — head-on track silk signatures ===")
    panel = pq.read_table(PANEL_IN).to_pydict()
    track_sig = headon_track_silks(cap, fps, panel)
    print(f"{len(track_sig)} robust head-on tracks with silk signatures")

    print("\n=== ASSIGNMENT (Hungarian, hue-weighted LAB) ===")
    assigns, tracks, anums, cost = assign(track_sig, lut)
    # per-track median off_m for the lateral read
    from collections import defaultdict

    off_by_track = defaultdict(list)
    for tr, off, rob in zip(panel["track"], panel["off_m"], panel["robust"]):
        if rob:
            off_by_track[int(tr)].append(float(off))
    med_off = {tr: float(np.median(v)) for tr, v in off_by_track.items()}

    # confidence: margin normalised to the cost spread; flag ambiguous if small
    margins = [a["margin"] for a in assigns]
    mscale = np.median(margins) if margins else 1.0
    for a in assigns:
        a["conf"] = float(min(1.0, a["margin"] / (2 * mscale + 1e-6)))
        a["med_off"] = med_off.get(a["track"], float("nan"))
    assigns.sort(key=lambda a: a["med_off"])
    print(f"{'runner':>6} {'name':<18} {'track':>5} {'off_m':>6} {'dist':>6} "
          f"{'margin':>7} {'conf':>5}")
    for a in assigns:
        amb = "  <-- AMBIGUOUS" if a["conf"] < 0.35 else ""
        print(f"  #{a['runner']:>3} {FIELD.get(a['runner'],'?'):<18} {a['track']:>5} "
              f"{a['med_off']:>6.1f} {a['dist']:>6.1f} {a['margin']:>7.1f} "
              f"{a['conf']:>5.2f}{amb}")

    confident = [a for a in assigns if a["conf"] >= 0.35]
    print(f"\nConfidently numbered: {len(confident)}/12  "
          f"({sorted(a['runner'] for a in confident)})")
    ambiguous = [a for a in assigns if a["conf"] < 0.35]
    if ambiguous:
        print(f"Ambiguous: {sorted(a['runner'] for a in ambiguous)}")

    # ---- VALIDATION vs stewards (validation only) ----
    print("\n=== VALIDATION vs stewards' Comments-on-Running (VALIDATION ONLY) ===")
    offs = np.array([a["med_off"] for a in assigns])
    tight_thr = float(np.percentile(offs, 40))
    wide_thr = float(np.percentile(offs, 60))
    runner_off = {a["runner"]: a["med_off"] for a in assigns}

    def read(off):
        return "TIGHT" if off <= tight_thr else ("WIDE" if off >= wide_thr else "MID")

    print(f"field lateral span {offs.min():.1f}-{offs.max():.1f} m "
          f"(tight<= {tight_thr:.1f}, wide>= {wide_thr:.1f})")
    ok_n = 0
    tot = 0
    for nm, note in {**STEW_TIGHT, **STEW_WIDE}.items():
        if nm not in runner_off:
            print(f"  #{nm:>2} {FIELD.get(nm,'?'):<18} NOT NUMBERED ({note})")
            continue
        off = runner_off[nm]
        got = read(off)
        want = "TIGHT" if nm in STEW_TIGHT else "WIDE"
        hit = got == want or (want == "TIGHT" and got != "WIDE") or (
            want == "WIDE" and got != "TIGHT"
        )
        tot += 1
        ok_n += int(hit)
        print(f"  #{nm:>2} {FIELD.get(nm,'?'):<18} off={off:>5.1f}m -> {got:<5} "
              f"(stewards: {want}) {'OK' if hit else 'MISS'}  [{note}]")
    rail_off = np.mean([runner_off[n] for n in STEW_TIGHT if n in runner_off])
    wide_off = np.mean([runner_off[n] for n in STEW_WIDE if n in runner_off])
    gap = wide_off - rail_off
    print(f"\nGAP: mean wide({list(STEW_WIDE)}) {wide_off:.1f}m  -  "
          f"mean rail({list(STEW_TIGHT)}) {rail_off:.1f}m  =  {gap:+.1f}m")
    print(f"stewards agreement: {ok_n}/{tot}")

    # ---- write numbered panel (race_t, arclen, y_lat_m, confidence) ----
    conf_by_track = {a["track"]: a["conf"] for a in assigns}
    run_by_track = {a["track"]: a["runner"] for a in assigns}
    col = {"runner": [], "race_t": [], "arclen": [], "y_lat_m": [], "confidence": []}
    for tr, t, off, rob in zip(panel["track"], panel["t"], panel["off_m"], panel["robust"]):
        tr = int(tr)
        if not rob or tr not in run_by_track:
            continue
        rt = vid_to_race_t(float(t))
        col["runner"].append(run_by_track[tr])
        col["race_t"].append(rt)
        col["arclen"].append(race_t_to_arclen(rt))
        col["y_lat_m"].append(float(off))
        col["confidence"].append(conf_by_track[tr])
    pq.write_table(pa.table(col), PANEL_OUT)
    print(f"\npanel: {PANEL_OUT} ({len(col['runner'])} rows, "
          f"{len(set(col['runner']))} numbered runners)")

    make_qc(lut, track_sig, assigns, panel, off_by_track, tight_thr, wide_thr, runner_off)
    cap.release()
    os.makedirs("data_cache/figs", exist_ok=True)
    return assigns


def make_qc(lut, track_sig, assigns, panel, off_by_track, tight_thr, wide_thr, runner_off):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("data_cache/figs", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (i) LUT silk swatches per number
    a0 = ax[0, 0]
    nums = sorted(lut)
    for k, nm in enumerate(nums):
        bgr = cv2.cvtColor(lut[nm]["lab"].astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
        rgb = bgr[0, 0][::-1] / 255.0
        a0.add_patch(plt.Rectangle((0, k), 1, 0.9, color=rgb))
        a0.text(1.1, k + 0.45, f"#{nm}  {FIELD.get(nm,'?')}", va="center", fontsize=10)
    a0.set_xlim(0, 4)
    a0.set_ylim(0, len(nums))
    a0.axis("off")
    a0.set_title("(i) LUT — finish-cam silk colour per number")

    # (ii) head-on track silk swatches + assigned number
    a1 = ax[0, 1]
    srt = sorted(assigns, key=lambda a: a["med_off"])
    for k, a in enumerate(srt):
        sig = track_sig.get(a["track"])
        if sig is None:
            continue
        bgr = cv2.cvtColor(sig["lab"].astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)
        rgb = bgr[0, 0][::-1] / 255.0
        a1.add_patch(plt.Rectangle((0, k), 1, 0.9, color=rgb))
        amb = "*" if a["conf"] < 0.35 else ""
        a1.text(1.1, k + 0.45, f"trk{a['track']} -> #{a['runner']}{amb}  "
                f"off={a['med_off']:.1f}m conf={a['conf']:.2f}", va="center", fontsize=9)
    a1.set_xlim(0, 6)
    a1.set_ylim(0, len(srt))
    a1.axis("off")
    a1.set_title("(ii) head-on track silk -> assigned number (*ambiguous)")

    # (iii) numbered lateral tracks vs time
    a2 = ax[1, 0]
    run_by_track = {a["track"]: a["runner"] for a in assigns}
    from collections import defaultdict

    tline = defaultdict(list)
    for tr, t, off, rob in zip(panel["track"], panel["t"], panel["off_m"], panel["robust"]):
        if rob and int(tr) in run_by_track:
            tline[int(tr)].append((float(t) - PATROL_OFFSET, float(off)))
    for tr, pts in tline.items():
        pts.sort()
        ts = [p[0] for p in pts]
        offs = [p[1] for p in pts]
        a2.plot(ts, offs, "-o", ms=2.5, lw=1.3, alpha=0.85)
        a2.text(ts[-1], offs[-1], f"#{run_by_track[tr]}", fontsize=9, weight="bold")
    a2.set_xlabel("race_t (s)")
    a2.set_ylabel("lateral offset from rail (m)")
    a2.set_title("(iii) NUMBERED head-on lateral tracks")
    a2.grid(alpha=0.3)

    # (iv) tight/wide vs stewards
    a3 = ax[1, 1]
    a3.axis("off")
    lines = ["TIGHT/WIDE vs stewards (validation only)\n"]
    lines.append(f"  thresholds: tight<= {tight_thr:.1f}m  wide>= {wide_thr:.1f}m\n")
    for nm, note in {**STEW_TIGHT, **STEW_WIDE}.items():
        if nm not in runner_off:
            lines.append(f"  #{nm} NOT NUMBERED")
            continue
        off = runner_off[nm]
        got = "TIGHT" if off <= tight_thr else ("WIDE" if off >= wide_thr else "MID")
        want = "TIGHT" if nm in STEW_TIGHT else "WIDE"
        lines.append(f"  #{nm:>2} {FIELD.get(nm,'?')[:14]:<14} {off:>5.1f}m -> {got:<5} "
                     f"(want {want})")
    rail_off = np.mean([runner_off[n] for n in STEW_TIGHT if n in runner_off])
    wide_off = np.mean([runner_off[n] for n in STEW_WIDE if n in runner_off])
    lines.append(f"\n  GAP wide-rail = {wide_off-rail_off:+.1f} m")
    a3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("Patrol head-on lateral — NUMBERED by silk (study #86, ST 2026-06-03 R1)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    plt.close(fig)
    print(f"QC figure: {FIG}")


if __name__ == "__main__":
    main()

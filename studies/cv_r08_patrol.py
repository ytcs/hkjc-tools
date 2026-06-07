"""R08 PATROL head-on lateral — REUSE cv_patrol + the R03 auto-detect (#83/0076/#91) on HV R08 (#95).

HELD-OUT route generalization. Imports the R1 patrol GEOMETRY unchanged (_fit_white_rail /
detect_jockeys / build_homography / project / _turf_hue) AND the R03 auto-detect driver
machinery (autodetect_headon / fit_fixed_rail_window / collect_detections / calibrate_mpp /
build_numbered) verbatim — only the SCAN RANGE + clock anchor are re-pointed to the route
(the home straight arrives ~108s into the route vs ~69s on the sprint).

NO algorithm/threshold is tuned for R08. The head-on window is AUTO-DETECTED by the same
rail-slope-stability scan that found R03's 60-66s; we only widen the scan to the route's later
home straight.

Writes data_cache/cv_r08_patrol_lateral.parquet (anon) + cv_r08_patrol_numbered.parquet.
Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_patrol.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "studies")
import cv_patrol_lateral as P1  # noqa: E402,N812  (R1 patrol geometry, reused)
# REUSE the R03 auto-detect driver machinery verbatim (only re-point the scan range/clock below).
from cv_r03_patrol import (  # noqa: E402
    autodetect_headon,
    build_homography,
    calibrate_mpp,
    collect_detections,
    detect_jockeys,
    fit_fixed_rail_window,
    link_tracks,
    project,
)
from cv_r03_patrol import _turf_hue  # noqa: E402
from cv_r08_common import (  # noqa: E402
    TRACK_WIDTH_M,
    WINNER_FINISH_RACE_T,
    lateral_truth,
    parse_official,
    race_t_to_arclen,
)

VIDEO = "data_cache/video/patrol_20260603_08.mp4"
PANEL = "data_cache/cv_r08_patrol_lateral.parquet"
NUMBERED = "data_cache/cv_r08_patrol_numbered.parquet"
FIG = "data_cache/figs/cv_r08_patrol_qc.png"

P1.COURSE_WIDTH_M = TRACK_WIDTH_M

# Patrol shared-clock anchor (R08): the winner crosses near vid ~? on the 148.8s patrol feed; the
# route winner-time is 108.62s. Frame-anchored below. The head-on window sits BEFORE the line.
PATROL_WINNER_CROSS_VID = 120.7   # frame-anchored winner #6 line-cross (route; verified: #6
# surges clear ~vid 120, field past the post by 122 on the 148.7s patrol feed)
PATROL_OFFSET = PATROL_WINNER_CROSS_VID - WINNER_FINISH_RACE_T


def _validate_numbered(df, rail_t, wide_t):
    if df is None:
        return float("nan")
    import polars as pl
    print("\n[VALIDATE numbered vs stewards rail/wide]")
    med = {n: float(np.median(df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()))
           for n in df["runner"].unique().to_list()}
    rv = [med[n] for n in rail_t if n in med]
    wv = [med[n] for n in wide_t if n in med]
    for n in sorted(med):
        tag = "rail/tight" if n in rail_t else ("wide" if n in wide_t else "mid")
        print(f"    #{n:2d} [{tag:10s}] {med[n]:5.1f}m")
    gap = float("nan")
    if rv and wv:
        gap = float(np.mean(wv) - np.mean(rv))
        print(f"  GAP wide-rail = {gap:+.1f}m "
              f"(rail {sorted(rail_t & set(med))} {np.mean(rv):.1f}m, "
              f"wide {sorted(wide_t & set(med))} {np.mean(wv):.1f}m)  SIGN "
              f"{'CORRECT' if gap > 0 else 'WRONG'}")
    return gap


def build_numbered(cap, fps, tracks, rail, mpp, t0, t1):
    """Bridge anon head-on tracks to numbers via silk colour vs a finish-cam LUT (REUSE R03)."""
    import re

    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        print("[identity] RapidOCR unavailable — numbered segment skipped")
        return None
    ocr = RapidOCR()

    # finish-cam window: just after the head-on, the approach to the line (route timing).
    fin_t0, fin_t1, fdt = t1 + 4.0, min(t1 + 30.0, 145.0), 0.2
    lut: dict[int, list] = {}
    for t in np.arange(fin_t0, fin_t1, fdt):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        hh = fr.shape[0]
        crop = fr[int(hh * 0.25): int(hh * 0.85), :]
        for scale in (1.6, 2.2):
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(big)
            if not res:
                continue
            for box, txt, conf in res:
                txt = txt.strip()
                if re.fullmatch(r"\d{1,2}", txt) and conf > 0.45 and 1 <= int(txt) <= 11:
                    num = int(txt)
                    cx = float(np.mean([p[0] for p in box])) / scale
                    cy = float(np.mean([p[1] for p in box])) / scale + int(hh * 0.25)
                    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                    y0 = max(0, int(cy - 40))
                    y1 = max(0, int(cy - 5))
                    x0 = max(0, int(cx - 18))
                    x1 = min(fr.shape[1], int(cx + 18))
                    patch = hsv[y0:y1, x0:x1]
                    if patch.size:
                        lut.setdefault(num, []).append(patch.reshape(-1, 3))
    numbered_lut = {n: np.vstack(v) for n, v in lut.items() if len(v) >= 2}
    print(f"[identity] finish-cam silk-LUT numbers (>=2 OCR hits): {sorted(numbered_lut)} "
          f"({len(numbered_lut)}/11)")
    if len(numbered_lut) < 3:
        print("[GENERALIZE] patrol identity DEGRADED on R08 — too few legible LUT numbers; "
              "numbered segment falls back to anonymous tracks (rail/wide class only).")
        return None

    from cv_patrol_identity import _hist_dist, _hue_hist

    def sig(pixels):
        return _hue_hist(pixels[:, 0], pixels[:, 1], pixels[:, 2])

    lut_sig = {n: sig(px) for n, px in numbered_lut.items()}
    lut_sig = {n: s for n, s in lut_sig.items() if s is not None}

    track_px: dict[int, list] = {}
    robust = [(ti, pts) for ti, pts in enumerate(tracks) if len(pts) >= 20]
    for t in np.arange(t0, t1, 0.4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        tl = _turf_hue(fr)
        js = detect_jockeys(fr, tl)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        proj = (project(build_homography(rail, fr.shape[0], fr.shape[1], mpp),
                        np.array([[j["cx"], j["y_ground"]] for j in js], float)) if js else [])
        for j, p in zip(js, proj, strict=False):
            off_m, dep = float(p[0]), float(p[1])
            best_ti, bd = None, 1e9
            for ti, pts in robust:
                near = [pp for pp in pts if abs(pp[0] - t) < 0.6]
                if not near:
                    continue
                d = min(abs(off_m - pp[1]) + 4.0 * abs(dep - pp[2]) for pp in near)
                if d < bd:
                    bd, best_ti = d, ti
            if best_ti is not None and bd < 2.5:
                y0 = max(0, int(j["cy"] - 12))
                y1 = min(fr.shape[0], int(j["cy"] + 12))
                x0 = max(0, int(j["cx"] - 14))
                x1 = min(fr.shape[1], int(j["cx"] + 14))
                patch = hsv[y0:y1, x0:x1]
                if patch.size:
                    track_px.setdefault(best_ti, []).append(patch.reshape(-1, 3))
    track_sig = {}
    for ti, lst in track_px.items():
        s = sig(np.vstack(lst))
        if s is not None:
            track_sig[ti] = s
    if not track_sig or not lut_sig:
        print("[identity] no track/LUT signatures — anonymous only")
        return None

    from scipy.optimize import linear_sum_assignment
    tis = sorted(track_sig)
    nums = sorted(lut_sig)
    cost = np.array([[_hist_dist(track_sig[ti], lut_sig[n]) for n in nums] for ti in tis])
    ri, ci = linear_sum_assignment(cost)
    assign = {}
    for r, c in zip(ri, ci, strict=True):
        ti, num = tis[r], nums[c]
        row = cost[r]
        second = np.partition(row, 1)[1] if len(row) > 1 else row[0]
        margin = (second - row[c]) / (np.median(cost) + 1e-6)
        assign[ti] = (num, float(np.clip(margin, 0.0, 1.0)))
    print(f"[identity] assigned {len(assign)} tracks -> numbers "
          f"{sorted(n for n, _ in assign.values())}")

    rows = []
    for ti, (num, conf) in assign.items():
        for t, off_m, _dep in tracks[ti]:
            race_t = t - PATROL_OFFSET
            arclen = float(race_t_to_arclen(race_t))
            rows.append(dict(runner=num, race_t=round(race_t, 3), arclen=round(arclen, 2),
                             y_lat_m=round(off_m, 3), confidence=round(conf, 3)))
    if rows:
        import polars as pl
        df = pl.DataFrame(rows).sort(["runner", "race_t"])
        df.write_parquet(NUMBERED)
        print(f"[panel] {NUMBERED} ({df.height} rows, {df['runner'].n_unique()} runners, "
              f"arclen {df['arclen'].min():.0f}-{df['arclen'].max():.0f}m)")
        return df
    return None


def _write_anon(tracks):
    import pyarrow as pa
    import pyarrow.parquet as pq
    col = {"track": [], "t": [], "off_m": [], "depth": [], "robust": []}
    for ti, pts in enumerate(tracks):
        robust = len(pts) >= 20
        for t, off, dep in pts:
            col["track"].append(ti)
            col["t"].append(t)
            col["off_m"].append(off)
            col["depth"].append(dep)
            col["robust"].append(robust)
    pq.write_table(pa.table(col), PANEL)
    print(f"[panel] {PANEL} ({len(col['t'])} rows, {len(tracks)} tracks)")


def main():
    off = parse_official()
    rail_t, wide_t = lateral_truth(off)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 80)
    print("R08 PATROL #95 — head-on lateral on HV 1800m ROUTE (auto-detect window, no tuning)")
    print("=" * 80)

    # [GENERALIZE] AUTO-DETECT the head-on window. The route home straight arrives ~108s in (vs
    # R03's ~63s) -> widen the scan to the route's later straight. Same rail-slope-stability scan.
    t0, t1 = autodetect_headon(cap, fps, scan_t0=95.0, scan_t1=120.0, step=1.0)
    print(f"[GENERALIZE] auto-detected head-on window t={t0:.1f}-{t1:.1f}s "
          f"(R03 was 60-66s; route home-straight arrives ~40s later)")

    rail = fit_fixed_rail_window(cap, fps, t0, t1)
    print(f"[rail] x = {rail.a:.4f}*y + {rail.b:.1f}  slope CV {rail.slope_cv * 100:.1f}% "
          f"(R1 3.0%, R03 9.7%; <~5% == fixed-camera head-on confirmed)")
    if rail.slope_cv > 0.12:
        print("[GENERALIZE-WARN] rail slope CV high — head-on window may be mis-detected / the "
              "route may lack a clean fixed head-on segment.")

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mpp = calibrate_mpp(cap, fps, rail, t0, t1)
    Hmat = build_homography(rail, h, w, mpp)

    rows = collect_detections(cap, fps, Hmat, t0, t1)
    nfr = len(set(round(r[0], 3) for r in rows))
    print(f"[detect] {len(rows)} ground-contacts over {nfr} frames "
          f"(avg {len(rows) / max(nfr, 1):.1f}/frame)")

    tracks = link_tracks(rows)
    from cv_patrol_run import dispersion_ratio, robust_dispersion
    within_m, between_m, ratio, _medians = dispersion_ratio(tracks)
    rb = robust_dispersion(rows, tracks)
    long_tracks = [t for t in tracks if len(t) >= 20]
    print(f"[tracks] {len(tracks)} (>=8 dets); {len(long_tracks)}/11 robust (>=20)")
    print("\n[HEADLINE — within/between lateral dispersion ratio]")
    print(f"  (A) frame jitter  {rb['jitter']:.2f}/{rb['between']:.2f} = {rb['ratio_jitter']:.3f}")
    print(f"  (B) 1.5s window   {rb['win15']:.2f}/{rb['between']:.2f} = {rb['ratio_win15']:.3f}")
    print(f"  (C) full track    {within_m:.2f}/{between_m:.2f} = {ratio:.3f}")
    print("  R1 ref 0.06/0.16/0.35; R03 0.055/0.10/0.18. want << 1")
    worst = max(rb["ratio_jitter"], rb["ratio_win15"], ratio)
    print(f"  VERDICT: {'STABLE (all<0.87)' if worst < 0.87 else 'PARTIAL/DEGRADED'}")

    _write_anon(tracks)
    numbered = build_numbered(cap, fps, tracks, rail, mpp, t0, t1)
    _validate_numbered(numbered, rail_t, wide_t)
    _qc(cap, fps, rail, rows, tracks, rb, ratio, t0, t1, h, w)
    cap.release()


def _qc(cap, fps, rail, rows, tracks, rb, ratio, t0, t1, h, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("data_cache/figs", exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(((t0 + t1) / 2) * fps))
    ok, fr = cap.read()
    vis = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else np.zeros((h, w, 3), np.uint8)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a0 = ax[0, 0]
    a0.imshow(vis)
    ys = np.arange(int(0.30 * h), h, 4)
    a0.plot(rail.a * ys + rail.b, ys, "m-", lw=2, label="fixed rail")
    a0.set_title(f"(i) head-on t={t0:.0f}-{t1:.0f}s, rail CV {rail.slope_cv * 100:.1f}%")
    a0.legend()
    a0.set_xlim(0, w)
    a0.set_ylim(h, 0)
    a2 = ax[1, 0]
    for pts in tracks:
        ts = [p[0] for p in pts]
        offs = [p[1] for p in pts]
        a2.plot(ts, offs, "-o" if len(pts) >= 20 else "-", ms=2,
                lw=1.4 if len(pts) >= 20 else 0.6, alpha=0.85 if len(pts) >= 20 else 0.25)
    a2.set_xlabel("t (s)")
    a2.set_ylabel("lateral off rail (m)")
    a2.set_title("per-track lateral vs time")
    a2.grid(alpha=0.3)
    a3 = ax[1, 1]
    a3.axis("off")
    a3.text(0.02, 0.98,
            f"HEADLINE within/between ratio\n  jitter {rb['ratio_jitter']:.3f}\n"
            f"  1.5s   {rb['ratio_win15']:.3f}\n  full   {ratio:.3f}\n\n"
            f"R03: 0.055/0.10/0.18\nhead-on t={t0:.0f}-{t1:.0f}s\nrail CV {rail.slope_cv * 100:.1f}%",
            va="top", fontsize=12, family="monospace")
    fig.suptitle("R08 PATROL head-on lateral (#95, HV 2026-06-03 R8 1800m route)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    plt.close(fig)
    print(f"[fig] {FIG}")


if __name__ == "__main__":
    main()

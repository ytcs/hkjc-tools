"""R08 IDENTITY v2 (#95) — run the HARDENED #94 consensus on HV R08 1800m, NO tuning.

HELD-OUT route generalization of the SINGLE FIXED identity consensus (cv_identity.
consensus_assign: own-silk-LUT 1.0 / RF-anchor 0.45 / reliability-gated clock-order 0.6 /
DRAW-order 0.7) + the draw->y_lat(0) fusion anchor (#93). The weights are used AS-IS — the same
values that no-regression-on-R01 / kept-the-fix-on-R03 (journal 0090). NOTHING is tuned for R08.

Mirrors cv_identity_v2_rerun.py's R03 path (build the RF anchor + own patrol finish-cam LUT,
rebuild the R08 patrol head-on tracks, consensus_assign, write the numbered patrol panel, then
fuse with the draw anchor) but standalone for the route + field-of-11.

DISCIPLINE: study-only; imports cv_identity / cv_r08_* / cv_r03_patrol READ-ONLY; src/hkjc
UNTOUCHED; banked R01/R03 parquets UNMUTATED; all outputs R08-suffixed; official VALIDATION-only
(draw = pre-race INITIAL CONDITION, allowed input).

    # build the cached RF/own anchors once (slow OCR), then run consensus+fusion:
    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_identity.py --stage anchor
    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_identity.py --stage all
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, "studies")

from cv_identity import (  # noqa: E402
    Identity,
    augment_silk_lut_from_patrol,
    build_rf_anchor,
    draw_to_metres,
    silk_distinctness,
    silk_hist,
)
from cv_r08_common import (  # noqa: E402
    FIELD_SIZE,
    TRACK_WIDTH_M,
    WINNER_FINISH_RACE_T,
    lateral_truth,
    parse_official,
    race_t_to_arclen,
)

RF_VIDEO = "data_cache/video/full_20260603_08.mp4"
PATROL_VIDEO = "data_cache/video/patrol_20260603_08.mp4"
IDENT_CACHE = "data_cache/cv_identity_R08.npz"
OWN_LUT_CACHE = "data_cache/cv_own_lut_R08_patrol.npz"
PATROL_NUMBERED_V2 = "data_cache/cv_r08_patrol_numbered.parquet"  # the fusion reads this name

# clock anchors (frame-verified; same as the per-feed drivers)
RF_WINNER_CROSS_VID = 109.7
PATROL_WINNER_CROSS_VID = 120.7
RF_OFFSET = RF_WINNER_CROSS_VID - WINNER_FINISH_RACE_T


# ============================================================================= anchor build (slow)
def build_anchor():
    """Solve identity ONCE from RF (saddlecloth LUT + legend rank backbone) + patrol augment."""
    off = parse_official()
    name2num = {v: k for k, v in off.horse.items()}

    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except Exception:
        ocr = None

    print("=" * 80)
    print("R08 IDENTITY ANCHOR (#95) — HV 2026-06-03 R08 1800m route")
    print("=" * 80)
    print(f"winner #{off.finish_order[0]} {off.horse[off.finish_order[0]]}")

    ident = build_rf_anchor(
        RF_VIDEO, name2num, RF_OFFSET,
        # R08 legible approach to the line (the finish cam blurs at the cross; route winner ~vid 109.7)
        fin_window=(107.2, 109.6, 0.04),
        legend_window=(4.0, 107.0, 2.0),
        field_size=FIELD_SIZE, ocr=ocr,
    )
    ident = augment_silk_lut_from_patrol(
        ident, PATROL_VIDEO, fin_window=(113.0, 124.0, 0.2),
        field_size=FIELD_SIZE, ocr=ocr,
    )
    ident.save(IDENT_CACHE)
    print(f"[identity] cached -> {IDENT_CACHE}")

    # own patrol finish-cam LUT (the feed's dedicated crisp side-on number source)
    blank = Identity(field_size=FIELD_SIZE)
    blank = augment_silk_lut_from_patrol(
        blank, PATROL_VIDEO, fin_window=(113.0, 124.0, 0.2), field_size=FIELD_SIZE, ocr=ocr)
    own = dict(blank.silk_lut)
    nums = sorted(own)
    np.savez(OWN_LUT_CACHE, nums=np.array(nums, dtype=int),
             hists=np.array([own[n] for n in nums]))
    print(f"[own LUT] cached {sorted(own)} ({len(own)}/{FIELD_SIZE}) -> {OWN_LUT_CACHE}")

    # rank backbone validation vs official RP (validation only)
    from scipy.stats import spearmanr
    from cv_r08_common import OFFICIAL_SECTIONS, _D_BOUNDS
    sec_t = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])
    print("\n[rank backbone vs official RP — validation only]")
    for name, arc, ci in [("c1", 200.0, 0), ("c2", 600.0, 1), ("c3", 1000.0, 2),
                          ("c4", 1400.0, 3), ("late", 1700.0, 4)]:
        rt = float(np.interp(arc, _D_BOUNDS, sec_t))
        rk = ident.rank_at(rt)
        if not rk:
            continue
        common = [n for n in rk if n in off.rp and len(off.rp[n]) > ci]
        if len(common) >= 3:
            rho, _ = spearmanr([rk[n] for n in common], [off.rp[n][ci] for n in common])
            print(f"   {name:5s} race_t {rt:5.1f}s  rho(rank,RP)={rho:+.2f}  n={len(common)}")


# ============================================================================= patrol tracks (R08)
def _load_own_cache():
    if os.path.exists(OWN_LUT_CACHE):
        z = np.load(OWN_LUT_CACHE, allow_pickle=True)
        return {int(n): h for n, h in zip(z["nums"], z["hists"], strict=True)}
    return None


def patrol_tracks_r08():
    """Rebuild R08 patrol head-on tracks + per-track silk sig, lateral, depth, race_t (REUSE R03)."""
    import cv_r08_patrol as RP  # noqa: N812
    from cv_patrol_lateral import _turf_hue
    from cv_r03_patrol import (
        autodetect_headon,
        build_homography,
        calibrate_mpp,
        collect_detections,
        detect_jockeys,
        fit_fixed_rail_window,
        link_tracks,
        project,
    )

    cap = cv2.VideoCapture(RP.VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t0, t1 = autodetect_headon(cap, fps, scan_t0=95.0, scan_t1=120.0, step=1.0)
    rail = fit_fixed_rail_window(cap, fps, t0, t1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mpp = calibrate_mpp(cap, fps, rail, t0, t1)
    hmat = build_homography(rail, h, w, mpp)
    rows = collect_detections(cap, fps, hmat, t0, t1)
    tracks = link_tracks(rows)
    print(f"[R08 patrol] head-on t={t0:.1f}-{t1:.1f}s, rail CV {rail.slope_cv * 100:.1f}%; "
          f"{len(tracks)} tracks")

    robust = [(ti, pts) for ti, pts in enumerate(tracks) if len(pts) >= 20]
    track_px: dict[int, list] = {}
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
                y0, y1 = max(0, int(j["cy"] - 12)), min(fr.shape[0], int(j["cy"] + 12))
                x0, x1 = max(0, int(j["cx"] - 14)), min(fr.shape[1], int(j["cx"] + 14))
                patch = hsv[y0:y1, x0:x1]
                if patch.size:
                    track_px.setdefault(best_ti, []).append(patch.reshape(-1, 3))
    cap.release()

    sig, lat, depth, rt = {}, {}, {}, {}
    poff = PATROL_WINNER_CROSS_VID - WINNER_FINISH_RACE_T
    for ti, pts in robust:
        if ti not in track_px:
            continue
        s = silk_hist(np.vstack(track_px[ti]))
        if s is None:
            continue
        sig[ti] = s
        lat[ti] = float(np.median([p[1] for p in pts]))
        depth[ti] = float(np.median([p[2] for p in pts]))
        rt[ti] = float(np.median([p[0] for p in pts])) - poff
    return tracks, sig, lat, depth, rt, poff


# ============================================================================= consensus + panel
def run_consensus():
    off = parse_official()
    rail_t, wide_t = lateral_truth(off)
    draw = dict(off.draw)
    winner = off.finish_order[0]

    print("\n" + "=" * 80)
    print("R08 IDENTITY v2 CONSENSUS (#95) — SAME FIXED WEIGHTS as R01/R03 (no tuning)")
    print("=" * 80)
    print(f"draw (1=rail): {dict(sorted(draw.items()))}")
    print(f"rail/tight truth {sorted(rail_t)} | wide truth {sorted(wide_t)} | winner #{winner}")

    ident = Identity.load(IDENT_CACHE)
    ident.draw = draw
    print(f"[identity] RF anchor LUT {ident.known_numbers()} ({len(ident.silk_lut)}/{FIELD_SIZE}); "
          f"legend tp {len(ident.rank_seq)}")

    own = _load_own_cache() or {}
    distinct = silk_distinctness(own) if own else {}
    print(f"[own LUT] patrol finish-cam own LUT {sorted(own)} ({len(own)}/{FIELD_SIZE}); "
          f"distinctness {{{', '.join(f'{n}:{distinct[n]:.2f}' for n in sorted(own))}}}")

    tracks, sig, lat, depth, rt, poff = patrol_tracks_r08()

    assign = ident.consensus_assign(
        track_sig=sig, track_order=depth, track_lat=lat, track_race_t=rt,
        own_lut=own or None, own_distinct=distinct or None)
    print(f"[consensus] assigned runners {sorted(n for n, _ in assign.values())} "
          f"({len(assign)} tracks)")

    rows = []
    for ti, (num, conf) in assign.items():
        for t, off_m, _dep in tracks[ti]:
            race_t = t - poff
            rows.append(dict(runner=int(num), race_t=round(race_t, 3),
                             arclen=round(float(race_t_to_arclen(race_t)), 2),
                             y_lat_m=round(off_m, 3), confidence=round(conf, 3)))
    df = pl.DataFrame(rows).sort(["runner", "race_t"]) if rows else None
    gap = float("nan")
    if df is not None:
        df.write_parquet(PATROL_NUMBERED_V2)
        print(f"[panel] {PATROL_NUMBERED_V2} ({df.height} rows, {df['runner'].n_unique()} runners)")
        med = {n: float(np.median(df.filter(pl.col("runner") == n)["y_lat_m"].to_numpy()))
               for n in df["runner"].unique().to_list()}
        print("\n[VALIDATE patrol consensus vs stewards rail/wide]")
        for n in sorted(med):
            tag = "rail" if n in rail_t else ("wide" if n in wide_t else "mid")
            print(f"    #{n:2d} [{tag:4s}] {med[n]:6.2f}m")
        rv = [med[n] for n in rail_t if n in med]
        wv = [med[n] for n in wide_t if n in med]
        if rv and wv:
            gap = float(np.mean(wv) - np.mean(rv))
            print(f"  GAP wide-rail = {gap:+.2f}m  (rail {sorted(rail_t & set(med))} "
                  f"{np.mean(rv):.1f}m, wide {sorted(wide_t & set(med))} {np.mean(wv):.1f}m)  "
                  f"SIGN {'CORRECT' if gap > 0 else 'WRONG'}")

    d2m = draw_to_metres(draw, TRACK_WIDTH_M)
    print(f"\n[draw->metres] stall_width {TRACK_WIDTH_M / 14:.2f}m; "
          f"y_lat(0) span {min(d2m.values()):.1f}-{max(d2m.values()):.1f}m")
    if df is not None:
        pat_span = (df["y_lat_m"].min(), df["y_lat_m"].max())
        print(f"   cross-check: patrol metric field span {pat_span[0]:.1f}-{pat_span[1]:.1f}m "
              f"(draw map span {max(d2m.values()) - min(d2m.values()):.1f}m vs patrol "
              f"{pat_span[1] - pat_span[0]:.1f}m)")

    return dict(gap=gap, assign=assign, d2m=d2m, rail_t=rail_t, wide_t=wide_t, winner=winner)


# ============================================================================= fusion + draw anchor
def fusion_r08(res):
    import cv_r08_fusion as FUS  # noqa: N812
    from cv_fusion_4feed import Obs

    d2m = res["d2m"]
    orig_load_patrol = FUS.load_patrol_r08
    SD_DRAW = 1.5

    def load_patrol_with_draw():
        obs = orig_load_patrol()
        for n, y in d2m.items():
            obs.append(Obs(runner=int(n), race_t=0.0, arclen=0.0, ylat=float(y),
                           feed="draw_anchor", sd_arclen=999.0, sd_ylat=SD_DRAW))
        print(f"[draw anchor] added {len(d2m)} gate-jump y_lat(0) obs (sd {SD_DRAW}m)")
        return obs

    FUS.load_patrol_r08 = load_patrol_with_draw
    try:
        FUS.run()
    finally:
        FUS.load_patrol_r08 = orig_load_patrol


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["anchor", "consensus", "fusion", "all"])
    args = ap.parse_args()

    if args.stage == "anchor":
        build_anchor()
        sys.exit(0)
    if not os.path.exists(IDENT_CACHE):
        print("[run] no cached anchor — building it first (slow OCR pass)")
        build_anchor()
    res = run_consensus()
    if args.stage in ("fusion", "all"):
        fusion_r08(res)
    print("\n" + "=" * 80)
    print(f"R08 IDENTITY v2 (#95) — patrol consensus gap {res['gap']:+.2f}m "
          f"({'CORRECT' if res['gap'] > 0 else 'WRONG'} sign), {len(res['assign'])} tracks")
    print("=" * 80)

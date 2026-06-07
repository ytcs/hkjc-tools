"""R08 AERIAL render feed — REUSE the hardened render-inversion (#82/0075) on HV R08 1800m (#95).

HELD-OUT route generalization. Imports the R1 render-inversion GEOMETRY functions unchanged
(extract_render / faithfulness / _smooth / the odometer arclen) and re-points ONLY the
R1-baked race PARAMETERS at R08 (DIST 1650->1800, FIELD_SIZE 12->11, video path, CTX). NO
algorithm/threshold is tuned for R08.

THE ROUTE STRESS: 1800m has the most/sharpest bends -> the aerial render's oblique foreshortening
is worst here. We extract per-frame y_lat (used by the fusion foreshortening recon to quantify the
bend-vs-straight degradation). Every place the auto-detection (split-screen layout / odometer /
legend) succeeds or FAILS is printed as a [GENERALIZE] note.

Writes data_cache/cv_r08_aerial_render.parquet (R08-suffixed; R01/R03 parquets untouched).
Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_r08_aerial.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "studies")
import cv_s0d_aerial_render as R1  # noqa: E402,N812  (the render-inversion toolkit, reused)
from cv_r08_common import (  # noqa: E402
    DIST,
    FIELD_SIZE,
    TRACK_WIDTH_M,
    lateral_truth,
    parse_official,
)

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OUT_PANEL = "data_cache/cv_r08_aerial_render.parquet"
FIG_GEOM = "data_cache/figs/cv_r08_aerial_geom.png"

# [GENERALIZE 1] Re-point the R1 module constants at R08 (no edits to the R1 file).
R1.AERIAL_VIDEO = AERIAL_VIDEO
R1.RESULTS_HTML = None
R1.OUT_PANEL = OUT_PANEL
R1.FIG_GEOM = FIG_GEOM
R1.FIG_TRIP = "data_cache/figs/cv_r08_aerial_trip.png"
R1.DIST = DIST
R1.C_WIDTH_M = TRACK_WIDTH_M
R1.OFFICIAL_SECTIONS = None
R1.CTX = dict(venue="HV", date="2026-06-03", race_no=8, distance_m=DIST,
              surface="turf", course="C", field_size=FIELD_SIZE)


def main() -> None:
    stride = int(os.environ.get("R08_STRIDE", "5"))
    ocr_every = int(os.environ.get("R08_OCR_EVERY", "3"))
    marker_every = int(os.environ.get("R08_MARKER_EVERY", "8"))

    off = parse_official()
    official = {
        n: dict(pla=off.place_of[n], horse=off.horse[n], dr=off.draw[n], rp=off.rp[n])
        for n in off.finish_order
    }
    rail_t, wide_t = lateral_truth(off)
    print("=" * 80)
    print("R08 AERIAL #95 — render-inversion on HV 1800m ROUTE (reusing cv_s0d toolkit, no tuning)")
    print("=" * 80)
    print("OFFICIAL finish:", off.finish_order, "winner #", off.finish_order[0])

    (dets_by_frame, rail_by_frame, gc_by_det, legend_by_frame, odo_by_frame,
     marker_by_frame, fc_by_frame, valid_set, diag, fps,
     link_tracklets, assign_by_legend) = R1.extract_render(stride, ocr_every, marker_every)
    print("\n[recon diag]")
    for k, v in diag.items():
        print(f"  {k:22s}: {v}")
    if not dets_by_frame:
        print("[GENERALIZE-FAIL] NO DETECTIONS — split-screen layout detection failed on R08")
        return

    rframes = sorted(f for f in rail_by_frame if rail_by_frame[f].valid)
    print(f"[GENERALIZE] aerial-active (split-screen) frames {diag['frames_aerial_active']} "
          f"(single-cam lead-in excluded); rail valid {len(rframes)}")

    tracks = link_tracklets(dets_by_frame)
    runners = (assign_by_legend(tracks, legend_by_frame, FIELD_SIZE, valid_set)
               if legend_by_frame else [])
    bound = sorted(r.number for r in runners)
    print(f"[identity] legend frames {len(legend_by_frame)}; bound {len(bound)}/{FIELD_SIZE}: {bound}")

    from hkjc.vision.mode_b.pipeline_b import _odometer_arclength
    arclen_by_frame = _odometer_arclength(odo_by_frame, DIST)
    print(f"[arclen] odometer arclen in {len(arclen_by_frame)} frames")

    half = 960
    all_perp = []
    for r in runners:
        for d in r.dets:
            rail = rail_by_frame.get(d.frame)
            fc = fc_by_frame.get(d.frame)
            if rail is None or not rail.valid or fc is None:
                continue
            gx, gy, _u = gc_by_det.get((d.frame, round(d.cx), round(d.cy)),
                                       (d.cx - half, d.cy, False))
            _s, perp = rail.project(gx, gy, (fc[0] - half, fc[1]))
            all_perp.append(abs(perp))
    p98 = float(np.percentile(all_perp, 98)) if all_perp else 200.0
    m_per_px = TRACK_WIDTH_M / max(1.0, p98)
    print(f"[scale] lateral m/px = {m_per_px:.4f} (C-width {TRACK_WIDTH_M}m / p98 perp {p98:.0f}px)")

    rows = []
    for r in runners:
        for d in r.dets:
            rail = rail_by_frame.get(d.frame)
            fc = fc_by_frame.get(d.frame)
            if rail is None or not rail.valid or fc is None:
                continue
            bcx, bcy = d.cx - half, d.cy
            fcx, fcy = fc[0] - half, fc[1]
            gx, gy, used = gc_by_det.get((d.frame, round(d.cx), round(d.cy)), (bcx, bcy, False))
            _s_body, perp_body = rail.project(bcx, bcy, (fcx, fcy))
            _s_gc, perp_gc = rail.project(gx, gy, (fcx, fcy))
            y_body = abs(perp_body) * m_per_px
            y_gc = abs(perp_gc) * m_per_px
            if y_gc > TRACK_WIDTH_M + 5 or y_body > TRACK_WIDTH_M + 5:
                continue
            xa = arclen_by_frame.get(d.frame)
            rows.append(dict(
                runner=r.number, frame=d.frame, t=round(d.t, 3),
                x_arclen=float(xa) if xa is not None else float("nan"),
                y_lat=round(max(0.0, y_gc), 3), y_lat_body=round(max(0.0, y_body), 3),
                used_shadow=bool(used),
                silk_l=round(d.lab[0], 1), silk_a=round(d.lab[1], 1), silk_b=round(d.lab[2], 1),
            ))
    if not rows:
        print("[GENERALIZE-FAIL] NO ROWS")
        return
    df = pl.DataFrame(rows).sort(["runner", "frame"])
    df = R1._smooth(df, "y_lat").pipe(R1._smooth, "y_lat_body")
    os.makedirs(os.path.dirname(OUT_PANEL), exist_ok=True)
    df.write_parquet(OUT_PANEL)
    n_arc = int(df["x_arclen"].is_finite().sum())
    print(f"\n[panel] wrote {OUT_PANEL} ({df.height} rows, {df['runner'].n_unique()} runners, "
          f"{n_arc} with arclen)")

    fa = R1.faithfulness(df, official, "y_lat")
    if "ratio" in fa:
        print("\n[FAITHFULNESS — 4b battery]")
        print(f"  (a) within {fa['within']:.2f}m  between {fa['between']:.2f}m  RATIO {fa['ratio']:.2f}"
              f"  (R1 1.62-1.83 full / 1.29 body; R03 1.59)")
        print(f"      lag-1 autocorr median {fa['ac1_median']:+.2f}  frac>0.3 {fa['ac1_frac_pos']:.2f}"
              f"  (R1 +0.90, R03 +0.93)")
        print(f"  (b) corr(median y_lat, mean rp) = {fa['corr_med_rpmean']:+.2f}  (R1 +0.45, R03 -0.36)")
        body = df.filter(pl.col("x_arclen") < 0.92 * DIST)
        med = {n: float(np.nanmedian(body.filter(pl.col("runner") == n)["y_lat"].to_numpy()))
               for n in bound}
        rv = [med[n] for n in rail_t if n in med and np.isfinite(med[n])]
        wv = [med[n] for n in wide_t if n in med and np.isfinite(med[n])]
        if rv and wv:
            print(f"  (c) rail/tight {sorted(rail_t & set(med))} mean {np.mean(rv):.1f}m | "
                  f"wide {sorted(wide_t & set(med))} mean {np.mean(wv):.1f}m | "
                  f"GAP {np.mean(wv) - np.mean(rv):+.1f}m")
    print("\nDONE.")


if __name__ == "__main__":
    main()

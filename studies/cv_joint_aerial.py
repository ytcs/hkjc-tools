"""JOINT RECON — Stage 1 (aerial), step A: EXTRACTION LAYER + landmark inventory.

The CV backbone revamp (#101): replace 4 separate per-feed recons + late fusion with
EXTRACTION (raw observations, no metric commitment) -> ONE joint recon over a single RIGID
rail. This is the aerial extraction layer: one decode pass producing the raw observations the
pose fit needs, with NO arclen/y_lat commitment yet (the old code's fatal step).

Raw observations emitted per (aerial-active) frame, all PANEL-pixel coords:
  * distance-marker poles (value_m, cx, cy)  -- KNOWN-arclen landmarks that pin the homography
  * rail polyline (px, py)                    -- the rigid rail's image trace
  * odometer arclen (m)                       -- absolute field-position anchor
  * field centroid + per-horse ground contacts (identity-tagged via the legend)
  * legend running order

This step just RUNS it and reports the landmark inventory (how many marker landmarks, where,
odometer/rail coverage, IDed horses) so the pose fit (step B) can be designed against reality.
Caches to NPZ for the pose fit. Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_joint_aerial.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "studies")
import cv_s0d_aerial_render as R1  # noqa: E402,N812  (render-inversion extraction toolkit)
from cv_r08_common import DIST, FIELD_SIZE, TRACK_WIDTH_M, parse_official  # noqa: E402

AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
OUT_CACHE = "data_cache/cv_joint_aerial_R08_obs.npz"
HALF = 960  # full image is 1920 wide; panel = right half

# Re-point the R1 extraction toolkit at R08 (no edits to R1's algorithm).
R1.AERIAL_VIDEO = AERIAL_VIDEO
R1.RESULTS_HTML = None
R1.DIST = DIST
R1.C_WIDTH_M = TRACK_WIDTH_M
R1.OFFICIAL_SECTIONS = None
R1.CTX = dict(venue="HV", date="2026-06-03", race_no=8, distance_m=DIST,
              surface="turf", course="C", field_size=FIELD_SIZE)


def main() -> None:
    stride = int(os.environ.get("R08_STRIDE", "5"))
    ocr_every = int(os.environ.get("R08_OCR_EVERY", "3"))
    marker_every = int(os.environ.get("R08_MARKER_EVERY", "4"))  # denser: markers are gold

    off = parse_official()
    print("=" * 80)
    print("JOINT RECON Stage-1A — AERIAL extraction + landmark inventory (R08 1800m, HV 'C')")
    print("=" * 80)
    print("OFFICIAL finish:", off.finish_order, "winner #", off.finish_order[0])

    (dets_by_frame, rail_by_frame, gc_by_det, legend_by_frame, odo_by_frame,
     marker_by_frame, fc_by_frame, valid_set, diag, fps,
     link_tracklets, assign_by_legend) = R1.extract_render(stride, ocr_every, marker_every)
    print("\n[diag]", {k: v for k, v in diag.items()})
    if not dets_by_frame:
        print("[FAIL] no detections")
        return

    tracks = link_tracklets(dets_by_frame)
    runners = (assign_by_legend(tracks, legend_by_frame, FIELD_SIZE, valid_set)
               if legend_by_frame else [])
    bound = sorted(r.number for r in runners)
    print(f"[identity] bound {len(bound)}/{FIELD_SIZE}: {bound}")

    # ---- LANDMARK INVENTORY (the pose-fit's pinning constraints) -----------------------
    from hkjc.vision.mode_b.pipeline_b import _odometer_arclength
    arclen_by_frame = _odometer_arclength(odo_by_frame, DIST)

    rail_valid = [i for i, r in rail_by_frame.items() if r.valid]
    print("\n--- LANDMARK INVENTORY ---")
    print(f"  aerial-active frames     : {diag['frames_aerial_active']}")
    print(f"  rail polyline valid      : {len(rail_valid)} frames")
    print(f"  odometer arclen          : {len(arclen_by_frame)} frames "
          f"(range {min(arclen_by_frame.values()):.0f}..{max(arclen_by_frame.values()):.0f} m)"
          if arclen_by_frame else "  odometer: NONE")
    # marker landmarks: (frame, value_m, cx, cy)
    marks = []
    for i, mk in marker_by_frame.items():
        for (v, cx, cy) in mk:
            marks.append((i, v, cx, cy))
    vals = sorted({v for (_i, v, _x, _y) in marks})
    print(f"  distance-marker LANDMARKS : {len(marks)} reads over {len(marker_by_frame)} frames; "
          f"values seen (m-to-go): {vals}")
    # how many frames have BOTH a marker landmark AND a valid rail (fully-pinnable frames)
    pinnable = sorted(set(marker_by_frame) & set(rail_valid) & set(arclen_by_frame))
    print(f"  fully-pinnable frames     : {len(pinnable)} (marker + rail + odometer all present)")
    # marker landmark distribution: value -> n reads, and the frames' odometer arclen when seen
    print("  marker landmark detail (value_m : n_reads : odo-arclen-at-read):")
    for v in vals:
        fs = [(i, cx, cy) for (i, vv, cx, cy) in marks if vv == v]
        odos = [arclen_by_frame[i] for (i, _c, _y) in fs if i in arclen_by_frame]
        odotxt = (f"{min(odos):.0f}..{max(odos):.0f}m" if odos else "no-odo")
        print(f"     {v:4d}M : {len(fs):3d} reads : odo {odotxt}")

    # ---- cache raw observations for the pose fit (step B) ------------------------------
    # per-frame arrays; rails as object array of (N,2); markers as object array of (M,3)
    frames = sorted(rail_by_frame)
    rails = np.empty(len(frames), object)
    mks = np.empty(len(frames), object)
    odo = np.full(len(frames), np.nan)
    fcx = np.full(len(frames), np.nan)
    fcy = np.full(len(frames), np.nan)
    for j, i in enumerate(frames):
        r = rail_by_frame[i]
        rails[j] = (np.column_stack([r.px, r.py]) if r.valid else np.empty((0, 2)))
        mks[j] = np.array([[v, cx, cy] for (v, cx, cy) in marker_by_frame.get(i, [])], float).reshape(-1, 3)
        if i in arclen_by_frame:
            odo[j] = arclen_by_frame[i]
        if i in fc_by_frame:
            fcx[j], fcy[j] = fc_by_frame[i][0] - HALF, fc_by_frame[i][1]
    # per-horse ground contacts (panel coords): runner, frame, gx, gy
    hrows = []
    for r in runners:
        for d in r.dets:
            gx, gy, used = gc_by_det.get((d.frame, round(d.cx), round(d.cy)),
                                         (d.cx - HALF, d.cy, False))
            hrows.append([r.number, d.frame, gx, gy, float(used)])
    hrows = np.array(hrows, float).reshape(-1, 5)
    np.savez(OUT_CACHE, frames=np.array(frames), odo=odo, fcx=fcx, fcy=fcy,
             rails=rails, markers=mks, horse_gc=hrows, fps=fps,
             dist=DIST, track_width=TRACK_WIDTH_M, allow_pickle=True)
    print(f"\n[cache] wrote {OUT_CACHE} ({len(frames)} frames, {len(hrows)} horse ground-contacts)")
    print("DONE — landmark inventory above tells the pose fit (step B) what it has to work with.")


if __name__ == "__main__":
    main()

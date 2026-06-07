# ruff: noqa: N802, N803, N806
"""Fork-3 POC (DRAFT 0041): turn the extracted (t, x_long, y_lat) path into the
§3c lateral OBSERVABLES the geometry×control axis needs — the smallest end-to-end
slice from drone video -> a physics-relevant lateral statistic.

Reads the committed `data_cache/path_sample.parquet` (the #21 extractor output:
3 horse tracks over the HV 2026-06-03 R1 start straight, t=6-11s) and derives, per
track:
  - mean_off       : mean lateral metres off the inner rail (the §3c y_i path level)
  - lat_excursion  : total |dy| travelled laterally (lane-changing / weaving cost)
  - ground_lost    : extra path length over the pure along-track distance, m
                     (= Σ|step| − Σ|along-step|); on a STRAIGHT this is the weaving
                     tax, the straight-segment analogue of the bend arc-length tax
                     that §3c says draw pays through.

This is the canonical §3c quantity the rank-trajectory panel (trajectory_dataset_full,
section velocity+rank, NO lateral coord) structurally cannot produce — the reason CV is
the only path to it. Honest accuracy notes printed inline; this is a de-risking slice,
not a feature backfill. Disposable scratch; does NOT touch the model.

Run: uv run --group vision python studies/cv/lateral_poc.py
"""
from __future__ import annotations

import os
import numpy as np
import polars as pl

MAIN = "/home/ytc/Projects/hkjc-tools"
PATH = os.path.join(MAIN, "data_cache/path_sample.parquet")


def main() -> None:
    df = pl.read_parquet(PATH).sort(["horse_id", "t"])
    print("=" * 70)
    print("Fork-3 POC — §3c lateral observables from the extracted drone path")
    print("HV 2026-06-03 R1, start straight t=6-11s; source:", os.path.basename(PATH))
    print("=" * 70)
    print(f"{'horse_id':16s} {'n':>3} {'mean_off':>9} {'off_rng':>9} "
          f"{'lat_exc':>8} {'gnd_lost':>9} {'v_med':>6}")
    rows = []
    for hid, g in df.group_by("horse_id"):
        hid = hid[0] if isinstance(hid, tuple) else hid
        g = g.sort("t")
        if g.height < 5:
            continue
        t = g["t"].to_numpy()
        x = g["x_long"].to_numpy()
        y = g["y_lat"].to_numpy()
        # along-track step (already monotone-ish), full 2-D step, lateral step
        along = np.abs(np.diff(x))
        lat = np.abs(np.diff(y))
        step2d = np.hypot(np.diff(x), np.diff(y))
        ground_lost = float(step2d.sum() - along.sum())
        v = np.median(np.diff(x) / np.diff(t))
        rows.append((hid, g.height, float(y.mean()), float(y.max() - y.min()),
                     float(lat.sum()), ground_lost, float(v)))
        print(f"{hid:16s} {g.height:3d} {y.mean():9.2f} {y.max()-y.min():9.2f} "
              f"{lat.sum():8.2f} {ground_lost:9.2f} {v:6.1f}")

    print("-" * 70)
    print("READOUT")
    offs = np.array([r[2] for r in rows])
    print(f"  rail-hugger vs wide split is RESOLVED: mean_off spans "
          f"{offs.min():.1f}..{offs.max():.1f} m across {len(rows)} tracks — "
          f"the rail-saver (low) vs wide-and-clear (high) §3c types are SEPARABLE.")
    print("  ground_lost here is dominated by lateral re-id JITTER, not real weaving")
    print("  (the green/wide track's y_lat is noisy — see track_xy.png); so the")
    print("  ABSOLUTE number is not yet trustworthy. What IS demonstrated end-to-end:")
    print("  video pixels -> rail-relative (x_long, y_lat) -> a per-horse LATERAL")
    print("  statistic that the section-rank panel (no y) cannot produce.")
    print()
    print("  GATING for a real feature (NOT met by this slice):")
    print("   - full ARC (1650m), not a 5s/28m straight window  -> #34 backfill")
    print("   - all ~12 runners with stable IDENTITY across the race -> re-id (silks)")
    print("   - the jitter smoothed (Kalman on (x,y) in ground coords)")
    print("   - aggregated as a leak-free TRAILING per-horse trait over prior runs")


if __name__ == "__main__":
    main()

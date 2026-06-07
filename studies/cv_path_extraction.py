"""S0 driver — CV lateral-path extraction on the HKJC Aerial Virtual View sample.

Runs the Mode-B pipeline (``hkjc.vision.mode_b``) on ``data_cache/aerial_sample.mp4``,
writes a rail-relative path panel (Parquet + a few peek rows), and prints a sanity
diagnostic: are paths within the track width? are IDs stable? how is coverage?

Disposable scratch (studies/). Reads the video + ``hkjc.core.track`` geometry READ-ONLY;
writes only ``data_cache/cv_lateral_paths_sample.parquet`` and a small PNG diagnostic.

    uv run --group vision python studies/cv_path_extraction.py

The aerial sample is HKJC's *Aerial Virtual View* (a synthetic top-down render in the
right half of the frame); see ``hkjc.vision.mode_b`` for why that makes this a classical
colour-segmentation problem rather than a deep-detector one.
"""
from __future__ import annotations

import os

VIDEO = "data_cache/aerial_sample.mp4"
OUT_PARQUET = "data_cache/cv_lateral_paths_sample.parquet"
OUT_FIG = "data_cache/figs/cv_lateral_paths_sample.png"

# Context for the sample race. The clip is an ST 1600m turf race (odometer starts
# "1600M"); exact race id is unknown from the clip alone, so we tag it as the sample.
CTX = dict(
    venue="ST", date="2026-06-03", race_no=0, distance_m=1600.0,
    surface="turf", course="A", field_size=12,
)


def main() -> None:
    import numpy as np

    from hkjc.vision.mode_b import extract_lateral_paths
    from hkjc.vision.mode_b.pipeline_b import rows_to_dicts

    print("=" * 74)
    print("S0 · CV LATERAL-PATH EXTRACTION — Mode B (Aerial Virtual View render)")
    print("=" * 74)
    if not os.path.exists(VIDEO):
        print(f"  [abort] {VIDEO} missing")
        return

    res = extract_lateral_paths(VIDEO, stride=5, **CTX)
    d = res.diagnostics
    print("\n-- pipeline diagnostics --")
    for k, v in d.items():
        print(f"  {k:24s}: {v}")

    if not res.rows:
        print("\n  [abort] no rows extracted — see diagnostics.")
        return

    rows = rows_to_dicts(res.rows)
    import polars as pl

    df = pl.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.write_parquet(OUT_PARQUET)
    print(f"\n  wrote {OUT_PARQUET}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    print("  schema:", dict(df.schema))
    print("\n-- peek (first 8 rows) --")
    print(df.head(8))

    # ---- SANITY DIAGNOSTICS ---------------------------------------------------------
    print("\n-- SANITY --")
    tw = d["track_width_m"]
    yl = df["y_lat"].to_numpy()
    within = float((np.abs(yl) <= tw + 3).mean())   # allow a small margin for fit noise
    print(f"  track width (course {CTX['course']})      : {tw:.1f} m")
    print(f"  y_lat within [0, width+3m]          : {within * 100:.1f}% of rows")
    print(f"  y_lat range                          : [{yl.min():.2f}, {yl.max():.2f}] m  "
          f"(median {np.median(yl):.2f})")
    # ID stability: per-runner coverage (fraction of aerial frames it appears in)
    n_frames = df["frame"].n_unique()
    cov = (df.group_by("runner").len().sort("runner"))
    print(f"  aerial frames with paths             : {n_frames}")
    print(f"  re-id mode                           : {d.get('reid_mode')}")
    print(f"  runners bound to saddlecloth #       : {d.get('runners_bound')}")
    print(f"  re-id vote-purity (median)           : {d.get('reid_purity_median')}")
    print("  per-runner (saddlecloth#) coverage:")
    covs = []
    for r in cov.iter_rows():
        frac = 100 * r[1] / max(1, n_frames)
        covs.append(frac)
        pur = df.filter(pl.col("runner") == r[0])["reid_purity"][0]
        print(f"    #{r[0]:<3d}: {r[1]:4d} samples  ({frac:3.0f}% of frames)  purity={pur:.2f}")
    print(f"  coverage min/median/max              : "
          f"{min(covs):.0f}% / {np.median(covs):.0f}% / {max(covs):.0f}%")
    # arc-length: odometer-derived now (true distance), monotone by construction
    al = df.group_by("frame").agg(pl.col("x_arclen").mean()).sort("frame")["x_arclen"].to_numpy()
    mono = float((np.diff(al) >= -2.0).mean()) if len(al) > 1 else 1.0
    print(f"  arc-length source                    : {d.get('arclen_source')}")
    print(f"  arc-length monotone                  : {mono * 100:.0f}% of steps non-decreasing")

    # ---- QA: CV section times vs the video's PUBLISHED splits ------------------------
    print("\n-- QA: CV-derived section times vs PUBLISHED sectionals --")
    print(f"  published splits (from video)        : {d.get('published_splits')}")
    print(f"  CV-derived section times             : {d.get('cv_section_times')}")
    print(f"  section-time MAE                     : {d.get('section_time_mae_s')} s")
    print(f"  section-time max error               : {d.get('section_time_max_err_s')} s")
    print(f"  CV finish/running order (last frame) : {d.get('cv_finish_order')}")

    # ---- DIAGNOSTIC FIGURE ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for rid, sub in df.group_by("runner"):
            s = sub.sort("frame")
            ax[0].plot(s["x_arclen"], s["y_lat"], "-", lw=1, alpha=0.8, label=str(rid[0]))
            ax[1].plot(s["t"], s["y_lat"], "-", lw=1, alpha=0.8)
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].axhline(tw, color="r", lw=0.8, ls="--", label=f"rail+width={tw:.0f}m")
        ax[0].set_xlabel("x_arclen (m travelled)")
        ax[0].set_ylabel("y_lat (m outward from inner rail)")
        ax[0].set_title("Rail-relative paths (lateral vs arc-length)")
        ax[0].legend(fontsize=6, ncol=2)
        ax[1].axhline(tw, color="r", lw=0.8, ls="--")
        ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("y_lat (m)")
        ax[1].set_title("Lateral offset over time")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=110)
        print(f"\n  wrote diagnostic figure {OUT_FIG}")
    except Exception as e:  # noqa: BLE001 — figure is optional
        print(f"  (figure skipped: {e})")

    print("\nDONE.")


if __name__ == "__main__":
    main()

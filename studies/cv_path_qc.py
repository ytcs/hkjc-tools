"""QC visual for the reconstructed CV lateral paths (journal 0064) — for human quality-check
BEFORE scaling the AVV scrape (S0c). Shows the SECTION-LEVEL median rail-offset profile per runner
(the clean signal the re-id validated) over the faded raw per-frame points (the ~10%-swap noise),
labelled by saddlecloth number, with the track bounds. Reads the panel only; writes one PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PANEL = Path("data_cache/cv_lateral_paths_sample.parquet")
OUT = Path("data_cache/figs/cv_path_qc.png")
OUT_GRID = Path("data_cache/figs/cv_path_qc_grid.png")
TRACK_W = 30.5  # m, drawn track width (0 = inner rail)
N_BINS = 8      # arc-length bins for the section-median profile


def _section_median(x, y, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(y[(x >= lo) & (x < hi)]) if ((x >= lo) & (x < hi)).sum() else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])
    return centers, med


def grid(d: pl.DataFrame, runners, dist) -> None:
    """Per-runner small multiples — each reconstruction isolated for clean QC."""
    edges = np.linspace(0, dist, N_BINS + 1)
    cmap = plt.get_cmap("tab10")
    ncol = 3
    nrow = int(np.ceil(len(runners) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k, r in enumerate(runners):
        ax = axes[k]
        sub = d.filter(pl.col("runner") == r)
        x, y = sub["x_arclen"].to_numpy(), sub["y_lat"].to_numpy()
        purity = float(sub["reid_purity"][0]) if "reid_purity" in sub.columns else float("nan")
        col = cmap(k % 10)
        ax.scatter(x, y, s=7, color=col, alpha=0.12, linewidths=0)
        cx, med = _section_median(x, y, edges)
        ok = np.isfinite(med)
        ax.plot(cx[ok], med[ok], "-o", color=col, lw=2.4, ms=5)
        ax.axhline(0, color="k", lw=1.2)
        ax.axhline(TRACK_W, color="k", lw=0.8, ls="--")
        ax.set_ylim(-1.5, TRACK_W + 1.5)
        ax.set_title(f"#{r}  ·  purity {purity:.2f}  ·  {sub.height} pts", fontsize=10)
        ax.grid(alpha=0.2)
    for j in range(len(runners), len(axes)):
        axes[j].axis("off")
    fig.suptitle("CV-reconstructed lateral path PER RUNNER (section-median line over raw points) — "
                 "ST 2026-06-03 r0 · y=0 inner rail, dashed=outer (30.5 m)", fontsize=11)
    fig.supxlabel("arc-length along track (m)")
    fig.supylabel("rail offset y_lat (m)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_GRID, dpi=130)
    print(f"wrote {OUT_GRID}")


def main() -> None:
    d = pl.read_parquet(PANEL).filter(
        pl.col("x_arclen").is_finite() & pl.col("y_lat").is_finite()
    )
    runners = sorted(d["runner"].unique().to_list())
    dist = float(d["x_arclen"].max())
    edges = np.linspace(0, dist, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for k, r in enumerate(runners):
        sub = d.filter(pl.col("runner") == r)
        x = sub["x_arclen"].to_numpy()
        y = sub["y_lat"].to_numpy()
        purity = float(sub["reid_purity"][0]) if "reid_purity" in sub.columns else float("nan")
        col = cmap(k % 10)
        # faded raw per-frame points (the noise)
        ax.scatter(x, y, s=6, color=col, alpha=0.08, linewidths=0)
        # section-level median profile (the signal)
        med = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            med.append(np.median(y[m]) if m.sum() else np.nan)
        med = np.array(med)
        ok = np.isfinite(med)
        ax.plot(centers[ok], med[ok], "-o", color=col, lw=2.4, ms=5,
                label=f"#{r}  (purity {purity:.2f})")
        # label at the last valid section
        if ok.any():
            xe, ye = centers[ok][-1], med[ok][-1]
            ax.annotate(f"#{r}", (xe, ye), color=col, fontweight="bold",
                        fontsize=11, xytext=(6, 0), textcoords="offset points")

    ax.axhline(0, color="k", lw=1.5, ls="-")
    ax.axhline(TRACK_W, color="k", lw=1.0, ls="--")
    ax.text(dist * 0.01, 0.6, "inner rail (y=0)", fontsize=8, color="k")
    ax.text(dist * 0.01, TRACK_W - 1.4, f"outer rail (y={TRACK_W} m)", fontsize=8, color="k")
    ax.set_ylim(-1.5, TRACK_W + 1.5)
    ax.set_xlabel("arc-length along track (m, from start)")
    ax.set_ylabel("rail offset y_lat (m)   0 = on the rail")
    ax.set_title(
        "CV-reconstructed lateral paths — ST 2026-06-03 r0 (sample)\n"
        "bold = section-median profile (the re-id'd signal) · faded = raw per-frame points (noise) · "
        "QA: CV sectionals vs published MAE 0.12 s, finish order exact (journal 0064)",
        fontsize=10,
    )
    ax.legend(title="saddlecloth # (re-id purity)", ncol=2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}  ({len(runners)} runners, {d.height} raw points, {N_BINS}-section medians)")
    grid(d, runners, dist)


if __name__ == "__main__":
    main()

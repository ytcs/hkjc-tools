"""Validate the course-B inner-draw market mispricing on multi-season ST data.

Tests whether the crowd systematically under-prices inner draws at specific rail
positions (the 2026 panel flagged course B, ~+11pp inner−outer market residual). With
many seasons we can (a) confirm the empirical draw bias by rail position, (b) measure
the *market* residual by course (is it mispriced?), (c) compute the ROI of backing
inner draws by course with race-clustered bootstrap CIs, and (d) check per-season
robustness — the difference between a real edge and a one-season fluke.

Run: uv run python experiments/validate_course_draw.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data_cache")


def load() -> pl.DataFrame:
    df = pl.read_parquet(CACHE / "st_history.parquet")
    df = df.with_columns(
        # 'TURF - "C+3" Course' -> 'C+3'; 'ALL WEATHER TRACK' -> 'AWT'
        pl.when(pl.col("Course").cast(pl.Utf8).str.contains("ALL WEATHER")).then(pl.lit("AWT"))
        .otherwise(pl.col("Course").cast(pl.Utf8).str.extract(r'"([A-C][+0-9]*)"', 1)).alias("course"),
        pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
        pl.col("Pla").cast(pl.Int64, strict=False).alias("pla"),
        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("odds"),
        (pl.col("date").str.slice(0, 4)).alias("season"),
        (pl.col("date") + "_" + pl.col("RaceNumber").cast(pl.Utf8)).alias("rid"),
    ).filter(pl.col("draw").is_not_null() & pl.col("odds").is_not_null() & (pl.col("odds") > 1))
    df = df.with_columns((pl.col("pla") == 1).cast(pl.Int64).alias("won"),
                         pl.len().over("rid").alias("field"))
    # de-vig market win prob within race
    inv = 1.0 / pl.col("odds")
    df = df.with_columns((inv / inv.sum().over("rid")).alias("p_mkt"))
    return df.with_columns((pl.col("won") - pl.col("p_mkt")).alias("mres"))


def _boot_roi(odds, won, rids, iters=2000, seed=0):
    """Race-clustered bootstrap of ROI for a set of bets (resample races)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(rids)
    idx_by = {r: np.where(rids == r)[0] for r in uniq}
    rois = []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[r] for r in pick])
        rois.append((odds[sel] * won[sel]).mean() - 1.0)
    return np.percentile(rois, [5, 50, 95])


def main():
    df = load()
    nraces = df.select("rid").unique().height
    print(f"ST history: {df.height} runner-rows, {nraces} races, "
          f"seasons {sorted(df['season'].unique().to_list())}")
    courses = ["A", "A+2", "A+3", "B", "B+2", "B+3", "C", "C+3"]

    print("\n=== Empirical draw bias + market residual by rail position ===")
    print(f"{'course':7s} {'races':>6s} {'inWin':>6s} {'outWin':>7s} {'edge':>6s} "
          f"{'mres_in':>8s} {'mres_out':>9s} {'in-out':>7s}")
    for c in courses:
        s = df.filter(pl.col("course") == c)
        nr = s.select("rid").unique().height
        if nr < 15:
            continue
        inn = s.filter(pl.col("draw") <= 4)
        out = s.filter(pl.col("draw") >= pl.col("field") - 3)
        print(f"{c:7s} {nr:6d} {inn['won'].mean():6.3f} {out['won'].mean():7.3f} "
              f"{inn['won'].mean()-out['won'].mean():+6.3f} {inn['mres'].mean():+8.3f} "
              f"{out['mres'].mean():+9.3f} {inn['mres'].mean()-out['mres'].mean():+7.3f}")

    print("\n=== ROI of flat-staking inner draws (1-4) by rail position (race-clustered 90% CI) ===")
    for c in courses:
        s = df.filter((pl.col("course") == c) & (pl.col("draw") <= 4))
        nr = s.select("rid").unique().height
        if nr < 15:
            continue
        lo, mid, hi = _boot_roi(s["odds"].to_numpy(), s["won"].to_numpy(), s["rid"].to_numpy())
        flag = "  <== +EV (CI>0)" if lo > 0 else ""
        print(f"  {c:5s} bets={s.height:5d} races={nr:4d}  ROI={mid:+.1%}  90%CI[{lo:+.1%},{hi:+.1%}]{flag}")

    print("\n=== Per-season robustness: inner-draw ROI on the flagged course(s) ===")
    for c in ["A", "B"]:
        print(f"  course {c}:")
        for ssn in sorted(df["season"].unique().to_list()):
            s = df.filter((pl.col("course") == c) & (pl.col("draw") <= 4) & (pl.col("season") == ssn))
            nr = s.select("rid").unique().height
            if nr < 8:
                continue
            roi = (s["odds"].to_numpy() * s["won"].to_numpy()).mean() - 1.0
            print(f"    {ssn} races={nr:3d} bets={s.height:4d}  inner-draw ROI={roi:+.1%}")


if __name__ == "__main__":
    main()

"""Learn the freshness curve φ(days-since-last-race) from data, before coupling.

Freshness is an engine-TUNE state: short turnaround = residual fatigue (depressed
v_c), optimal spell = peak, long layoff = detraining + "needs the run". We isolate its
MARGINAL signal beyond form (long-run ability) by binning the relative-finish residual
(rf − form) against days-since. Leak-free: days-since and form both use only prior runs.

Run: uv run python experiments/freshness_eda.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data_cache")


def _norm(col):
    return pl.col(col).cast(pl.Utf8).str.replace_all("/", "-")


def build() -> pl.DataFrame:
    e = pl.read_parquet(CACHE / "energetics_dataset.parquet").with_columns(_norm("date").alias("date"))
    w = pl.read_parquet(CACHE / "wpq_dataset.parquet").with_columns(_norm("date").alias("date"))
    h = (e.select("date", "venue", "race_no", "runner", "horse_id")
         .join(w.select("date", "venue", "race_no", "runner", "finish_pos", "field_size"),
               on=["date", "venue", "race_no", "runner"])
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       (1.0 - (pl.col("finish_pos") - 1) / (pl.col("field_size") - 1)).alias("rf"))
         .sort(["horse_id", "d"]))
    # leak-free expanding form (ability) and days since previous run
    h = h.with_columns(
        ((pl.col("rf").cum_sum().over("horse_id") - pl.col("rf"))
         / (pl.col("rf").cum_count().over("horse_id") - 1)).alias("form"),
        (pl.col("d") - pl.col("d").shift(1).over("horse_id")).dt.total_days().alias("days_since"))
    return h.filter(pl.col("form").is_finite() & pl.col("days_since").is_not_null())


def main():
    h = build()
    h = h.with_columns((pl.col("rf") - pl.col("form")).alias("resid"))
    print(f"{h.height} runs with prior history  (days_since range "
          f"{h['days_since'].min()}–{h['days_since'].max()}, median {h['days_since'].median():.0f})\n")

    bins = [(0, 13), (14, 20), (21, 27), (28, 34), (35, 41), (42, 55), (56, 90), (91, 200), (201, 10000)]
    labels = ["≤13", "14-20", "21-27", "28-34", "35-41", "42-55", "56-90", "91-200", "201+"]
    print(f"{'days-since':>10s} {'n':>6s} {'mean rf':>9s} {'resid(rf-form)':>15s} {'win%':>7s}")
    overall_rf = h["rf"].mean()
    for (lo, hi), lab in zip(bins, labels):
        b = h.filter((pl.col("days_since") >= lo) & (pl.col("days_since") <= hi))
        if b.is_empty():
            continue
        win = (b["finish_pos"] == 1).mean()
        print(f"{lab:>10s} {b.height:>6d} {b['rf'].mean():>9.4f} {b['resid'].mean():>+15.4f} {win:>6.1%}")
    print(f"\noverall mean rf = {overall_rf:.4f}  (resid centred near 0 by construction; "
          f"the SHAPE across bins is φ)")


if __name__ == "__main__":
    main()

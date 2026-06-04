"""Data anchor for the policy map: does REAL win rate by running style shift with distance?

The sim (pace_geometry_litmus, best_response) predicts the front-runner win-edge FALLS with
distance — closers need a route to convert. Here we check it in real HKJC results: bin every
run by its realised early speed (rel_early) tercile WITHIN its race (front / mid / closer)
and by distance band, and report win rate as a multiple of fair share (win% × field_size, so
field-size differences across bands cancel). Signature to confirm: front tercile > 1 and
FALLING with distance; closer tercile RISING toward routes; the front−closer gap shrinking.

Run: uv run python experiments/policy_data_style_distance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data_cache")


def build() -> pl.DataFrame:
    s = (pl.read_parquet(CACHE / "sectional_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("d")))
    e = (pl.read_parquet(CACHE / "energetics_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")))
    w = (pl.read_parquet(CACHE / "wpq_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y/%m/%d", strict=False).alias("d")))
    keys = ["d", "venue", "race_no", "runner"]
    df = (s.select(keys + ["rel_early", "won"])
          .join(e.select(keys + ["distance"]), on=keys)
          .join(w.select(keys + ["field_size"]), on=keys))
    rid = ["d", "venue", "race_no"]
    # style = within-race tercile of realised early speed (3 = front, 1 = closer)
    df = df.with_columns(
        ((pl.col("rel_early").rank("ordinal").over(rid) - 1) /
         (pl.col("field_size") - 1)).alias("epos"))             # 0=slowest-early … 1=fastest-early
    df = df.with_columns(
        pl.when(pl.col("epos") >= 2 / 3).then(pl.lit("front"))
          .when(pl.col("epos") >= 1 / 3).then(pl.lit("mid"))
          .otherwise(pl.lit("closer")).alias("style"),
        pl.when(pl.col("distance") <= 1200).then(pl.lit("sprint ≤1200"))
          .when(pl.col("distance") <= 1650).then(pl.lit("mid 1400-1650"))
          .otherwise(pl.lit("route ≥1800")).alias("band"))
    return df


def main():
    df = build()
    print(f"{df.height} runs with sectional style + distance\n")
    bands = ["sprint ≤1200", "mid 1400-1650", "route ≥1800"]
    styles = ["front", "mid", "closer"]
    print(f"{'band':>16s} {'n':>6s}  " + "".join(f"{s:>10s}" for s in styles) + "   front−closer")
    for band in bands:
        b = df.filter(pl.col("band") == band)
        if b.is_empty():
            continue
        cells = {}
        for st in styles:
            g = b.filter(pl.col("style") == st)
            # win rate × field_size = multiple of fair share (1.0 = neutral)
            tf = float((g["won"] * g["field_size"]).mean()) if g.height else float("nan")
            cells[st] = tf
        gap = cells["front"] - cells["closer"]
        print(f"{band:>16s} {b.height:>6d}  " + "".join(f"{cells[s]:>10.2f}" for s in styles)
              + f"   {gap:>+8.2f}")
    print("\n(values = win rate ÷ fair share; 1.0 = wins at chance. Sim predicts the front−closer")
    print(" gap shrinks from sprint→route. Realised early speed is ability-confounded, so read the")
    print(" DISTANCE TREND within a style, not the absolute level.)")


if __name__ == "__main__":
    main()

"""Extend wpq_dataset to the full 5-season history — the OTHER ABM data cap.

load_st inner-joins energetics x wpq, and wpq (draw/odds/result/field) was also frozen at the
3-month window (2026-03..05). Unlike energetics it needs NO scrape: every field is already in
results_rich (Dr, ActWt, DeclarHorseWt, WinOdds, Pla, field counts). This rebuilds it over all
3248 races with the exact wpq schema. Output -> wpq_dataset_full.parquet.

Run: uv run python experiments/build_wpq_full.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

CACHE = Path("data_cache")


def main():
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("Dr").cast(pl.Int64, strict=False).alias("draw"),
                        pl.col("ActWt").cast(pl.Float64, strict=False).alias("act_wt"),
                        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("decl_wt"),
                        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("market_win_odds"),
                        pl.col("Pla").cast(pl.Int64, strict=False).alias("finish_pos"))
          .drop_nulls(["runner", "market_win_odds", "finish_pos"]))
    rid = ["date", "Venue", "RaceNumber"]
    rr = rr.with_columns((1.0 / pl.col("market_win_odds")).alias("_inv"))
    out = rr.with_columns(
        pl.col("Venue").alias("venue"),
        pl.col("RaceNumber").cast(pl.Int64).alias("race_no"),
        (pl.col("_inv") / pl.col("_inv").sum().over(rid)).alias("market_win_prob"),
        (pl.col("finish_pos") == 1).cast(pl.Int64).alias("won"),
        (pl.col("finish_pos") <= 3).cast(pl.Int64).alias("placed"),
        pl.len().over(rid).cast(pl.UInt32).alias("field_size"))
    out = out.with_columns(pl.col("market_win_prob").log().alias("log_market_prob"))
    cols = ["date", "venue", "race_no", "runner", "market_win_odds", "draw", "act_wt", "decl_wt",
            "market_win_prob", "log_market_prob", "finish_pos", "won", "placed", "field_size"]
    out = out.select(cols)
    out.write_parquet(CACHE / "wpq_dataset_full.parquet")
    nr = out.select(["date", "venue", "race_no"]).unique().height
    print(f"wrote wpq_dataset_full.parquet: {out.height} rows, {nr} races, "
          f"{out['date'].n_unique()} dates  span {out['date'].min()} -> {out['date'].max()}")


if __name__ == "__main__":
    main()

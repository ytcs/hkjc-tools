"""Fetch per-run official ratings for the energetics horses (for rating-segmented EB).

Each horse's form page carries its rating at every past run, so one fetch per horse
yields all its (date, rating) pairs. We cache a tidy ``(horse_id, date, rtg)`` table
to join onto the per-run energetics dataset, so the hierarchy can segment horses by
class instead of pooling one mono-population.

Run: uv run python experiments/build_rating_data.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.historical import get_horse_data

CACHE = Path("data_cache")


def fetch_one(horse_id: str):
    try:
        df = get_horse_data(horse_id)
        if df.height == 0 or "Rtg" not in df.columns:
            return None
        return df.select(
            pl.lit(horse_id).alias("horse_id"),
            pl.col("Date").alias("date"),
            pl.col("Rtg").cast(pl.Int64, strict=False).alias("rtg"),
        ).drop_nulls("rtg")
    except Exception:
        return None


def main():
    horses = (pl.read_parquet(CACHE / "energetics_dataset.parquet")["horse_id"]
              .unique().to_list())
    print(f"fetching ratings for {len(horses)} horses ...")
    frames, done = [], 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_one, h): h for h in horses}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                frames.append(r)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(horses)}  (rated horses: {len(frames)})")
    out = pl.concat(frames).unique(["horse_id", "date"]) if frames else pl.DataFrame()
    out.write_parquet(CACHE / "rating_dataset.parquet")
    print(f"wrote {CACHE/'rating_dataset.parquet'}: {out.height} (horse,date) ratings "
          f"for {out['horse_id'].n_unique()} horses")


if __name__ == "__main__":
    main()

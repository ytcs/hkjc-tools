"""BOUNDED trackwork backfill (research phase) -> data_cache/trackwork.parquet.

The trackwork HISTORY lives on the per-horse archive route (trackworkresult?horseid=…),
which needs the long HK_YYYY_XXXX id; our panels key on the short brand, so we map
brand -> HK_id via the already-scraped barrier_trials panel (98% of recent panel horses
covered). We restrict to the recent ~1-year slice of panel horses (a conversion read,
not an exhaustive scrape), fetch each horse's whole dated gallop log once, ~1 req/sec,
retry-once, skip empties.

Run: uv run python studies/backfill_trackwork.py
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import polars as pl

from hkjc.data.trackwork import backfill

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler()])

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
OUT = CACHE / "trackwork.parquet"
SINCE = date(2025, 5, 1)   # recent ~1-year slice (research read, not exhaustive)
# The per-horse archive pages are large (~0.8 MB) so a full 1.5k-horse sweep is ~2 h. For the
# RESEARCH conversion read we deterministically subsample to MAX_HORSES (a representative slice;
# each fetch still returns that horse's FULL multi-year history). Set MAX_HORSES=0 for all.
MAX_HORSES = int(os.environ.get("MAX_HORSES", "500"))


def horse_ids() -> list[str]:
    """Long HK ids for recent-1yr panel horses, mapped from the barrier_trials brand→id table."""
    e = (pl.read_parquet(CACHE / "energetics_dataset_full.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-")
                       .str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")))
    recent = set(e.filter(pl.col("d") >= SINCE)["horse_id"].unique().to_list())
    bt = pl.read_parquet(CACHE / "barrier_trials.parquet")
    m = (bt.filter(pl.col("horse_id").is_not_null() & pl.col("horse_brand").is_not_null())
         .select("horse_brand", "horse_id").unique())
    mapping = dict(zip(m["horse_brand"].to_list(), m["horse_id"].to_list()))
    ids = sorted({mapping[b] for b in recent if b in mapping})
    logging.info("recent-1yr panel horses=%d  mapped to HK_id=%d (%.1f%%)",
                 len(recent), len(ids), 100 * len(ids) / max(1, len(recent)))
    if MAX_HORSES and len(ids) > MAX_HORSES:
        step = len(ids) / MAX_HORSES                       # deterministic even spread
        ids = [ids[int(i * step)] for i in range(MAX_HORSES)]
        logging.info("subsampled to %d horses for the research read", len(ids))
    return ids


def main():
    ids = horse_ids()
    df = backfill(ids, sleep=1.0)
    df.write_parquet(OUT)
    logging.info("WROTE %s  rows=%d  horses=%d  date %s..%s", OUT, df.height,
                 df["horse_brand"].n_unique(), df["session_date"].min(), df["session_date"].max())


if __name__ == "__main__":
    main()

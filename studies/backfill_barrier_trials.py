"""Run the barrier-trial backfill and persist to data_cache/barrier_trials.parquet.

Sweeps 2022-06-01 .. 2026-06-05 (covers the race panel 2022-09-11 .. 2026-05-31
with a debutant lead-in). Uses hkjc.data.barrier_trials.backfill — dropdown +
Mon-Sat probing, ~1 req/sec, retry-once, skip empties.
"""
from __future__ import annotations

import logging
from pathlib import Path

from hkjc.data.barrier_trials import backfill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

OUT = Path(__file__).resolve().parents[1] / "data_cache" / "barrier_trials.parquet"


def main():
    df = backfill("2022-06-01", "2026-06-05", sleep=1.0)
    df.write_parquet(OUT)
    logging.info("WROTE %s  rows=%d  horses=%d  dates=%d",
                 OUT, df.height, df["horse_brand"].n_unique(), df["date"].n_unique())


if __name__ == "__main__":
    main()

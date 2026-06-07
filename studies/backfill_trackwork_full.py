"""RESUMABLE full ~1.5k-horse trackwork backfill -> data_cache/trackwork.parquet.

The bounded research backfill (studies/backfill_trackwork.py) writes the parquet only once at
the very end, so a long sweep that gets killed loses everything. This variant CHECKPOINTS: it
loads the existing parquet, skips horses already present (by brand), fetches the rest one at a
time (~1 req/sec, retry-once), and rewrites the merged + de-duplicated parquet every CHECKPOINT
horses. Idempotent and resumable — re-run after an interruption and it picks up where it stopped.

Run: uv run python studies/backfill_trackwork_full.py
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl
import requests

from hkjc.data.trackwork import _HEADERS, _SCHEMA, _parse_horse_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler()])

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
OUT = CACHE / "trackwork.parquet"
TWR_URL = "https://racing.hkjc.com/en-us/local/information/trackworkresult?horseid={horseid}"
SINCE = date(2025, 5, 1)     # recent ~1-year panel slice (~1.5k horses) — the brief's universe
CHECKPOINT = 25              # rewrite the merged parquet every N successful fetches
SLEEP = 1.0


def target_ids() -> list[str]:
    """Long HK ids for recent-1yr panel horses, mapped from the barrier_trials brand→id table."""
    e = (pl.read_parquet(CACHE / "energetics_dataset_full.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-")
                       .str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")))
    recent = set(e.filter(pl.col("d") >= SINCE)["horse_id"].unique().to_list())
    bt = pl.read_parquet(CACHE / "barrier_trials.parquet")
    m = (bt.filter(pl.col("horse_id").is_not_null() & pl.col("horse_brand").is_not_null())
         .select("horse_brand", "horse_id").unique())
    mapping = dict(zip(m["horse_brand"].to_list(), m["horse_id"].to_list()))
    return sorted({mapping[b] for b in recent if b in mapping})


def main():
    ids = target_ids()
    have = pl.read_parquet(OUT) if OUT.exists() else pl.DataFrame(schema=_SCHEMA)
    done_brands = set(have["horse_brand"].unique().to_list()) if have.height else set()
    todo = [h for h in ids if (h.split("_")[-1] if "_" in h else h) not in done_brands]
    logging.info("target=%d  already have=%d  todo=%d", len(ids), len(done_brands), len(todo))

    sess = requests.Session()
    sess.headers.update(_HEADERS)
    frames = [have] if have.height else []
    n_ok, since_ckpt = 0, 0
    for i, horse_id in enumerate(todo):
        brand = horse_id.split("_")[-1] if "_" in horse_id else None
        df = None
        for attempt in (1, 2):
            try:
                r = sess.get(TWR_URL.format(horseid=horse_id), timeout=20)
                r.raise_for_status()
                df = pl.DataFrame(_parse_horse_html(r.content, horse_id, brand), schema=_SCHEMA)
                break
            except requests.RequestException as e:
                if attempt == 2:
                    logging.warning("giving up on %s: %s", horse_id, e)
                else:
                    import time
                    time.sleep(2.0)
        if df is not None and df.height:
            frames.append(df)
            n_ok += 1
            since_ckpt += 1
            logging.info("%s: %d gallops (%d/%d)", horse_id, df.height, i + 1, len(todo))
        if since_ckpt >= CHECKPOINT:
            _flush(frames)
            since_ckpt = 0
        if SLEEP:
            import time
            time.sleep(SLEEP)
    _flush(frames)
    logging.info("DONE: %d new horses fetched", n_ok)


def _flush(frames: list[pl.DataFrame]) -> None:
    if not frames:
        return
    out = (pl.concat(frames, how="vertical_relaxed")
           .unique(subset=["horse_brand", "session_date"], keep="first")
           .sort(["horse_brand", "session_date"]))
    out.write_parquet(OUT)
    logging.info("CHECKPOINT wrote %s rows=%d horses=%d", OUT, out.height,
                 out["horse_brand"].n_unique())


if __name__ == "__main__":
    main()

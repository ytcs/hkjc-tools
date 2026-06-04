"""Fetch & cache declared exotic dividends for the Sha Tin races (for the exotic backtest).

For each ST race we pull the result page's dividend table (WIN/QIN/QPL/FCT/TRI/TCE/...)
— the *realized* payout of the winning combination of each pool. Cached tidy for the
exotic-pool backtest (does the structural model find value the crowd misprices?).

Run: uv run python experiments/build_dividends_st.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.historical import get_race_dividends

CACHE = Path("data_cache")


def race_list() -> list[tuple[str, int]]:
    e = pl.read_parquet(CACHE / "energetics_dataset.parquet")
    st = e.filter(pl.col("venue") == "ST").unique(["date", "race_no"])
    # date is YYYY-MM-DD -> results page wants YYYY/MM/DD
    return [(d.replace("-", "/"), int(r)) for d, r in zip(st["date"].to_list(), st["race_no"].to_list())]


def fetch_one(date_slash: str, race_no: int):
    try:
        df = get_race_dividends(date_slash, "ST", race_no)
        if df.height == 0:
            return None
        return df.select(
            pl.lit(date_slash.replace("/", "-")).alias("date"),
            pl.lit(int(race_no)).alias("race_no"),
            "pool", "combination", "dividend", "decimal")
    except Exception:
        return None


def main():
    races = race_list()
    print(f"fetching dividends for {len(races)} ST races ...")
    frames, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, d, r): (d, r) for d, r in races}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                frames.append(r)
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(races)} (with dividends: {len(frames)})")
    out = pl.concat(frames) if frames else pl.DataFrame()
    out.write_parquet(CACHE / "dividends_st.parquet")
    pools = out.group_by("pool").len().sort("len", descending=True) if out.height else out
    print(f"wrote {CACHE/'dividends_st.parquet'}: {out.height} rows, "
          f"{out['race_no'].len() if out.height else 0} pool-rows")
    if out.height:
        print(pools)


if __name__ == "__main__":
    main()

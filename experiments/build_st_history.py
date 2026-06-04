"""Scrape a multi-season Sha Tin result history to validate the course-B draw mispricing.

The 2026 panel hinted the market under-prices inner draws specifically at the B rail
position (~11pp inner−outer market residual on ~22 races — too thin to trust). This
pulls several seasons of ST results (draw, finish, win odds, course, going, distance)
so the draw × rail-position × market test has real statistical power.

Run: uv run python experiments/build_st_history.py
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.historical import get_race_data

CACHE = Path("data_cache")
START, END = dt.date(2022, 9, 1), dt.date(2026, 6, 1)
KEEP = ["Date", "RaceNumber", "HorseNo", "Dr", "Pla", "WinOdds", "Course", "Going", "RaceDistance"]


def fetch_meeting(date: dt.date):
    """Fetch all ST races on a date (stop at the first missing race). Returns rows or []."""
    ds = date.strftime("%Y/%m/%d")
    frames = []
    for r in range(1, 12):
        try:
            df = get_race_data(ds, "ST", r)
        except Exception:
            break
        if df.height == 0:
            break
        cols = [c for c in KEEP if c in df.columns]
        frames.append(df.select(cols).with_columns(pl.lit(date.isoformat()).alias("date")))
    return frames


def main():
    days = [START + dt.timedelta(d) for d in range((END - START).days)]
    print(f"probing {len(days)} dates for ST meetings ...")
    frames, done, meetings = [], 0, 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_meeting, d): d for d in days}
        for fut in as_completed(futs):
            fs = fut.result()
            if fs:
                frames.extend(fs)
                meetings += 1
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(days)} dates, {meetings} meetings, {len(frames)} races")
    out = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    out.write_parquet(CACHE / "st_history.parquet")
    print(f"wrote {CACHE/'st_history.parquet'}: {out.height} runner-rows, "
          f"{out.select(['date','RaceNumber']).unique().height if out.height else 0} races")


if __name__ == "__main__":
    main()

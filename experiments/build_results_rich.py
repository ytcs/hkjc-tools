"""Scrape multi-season ST+HV results KEEPING the condition/skill signals st_history dropped.

st_history kept only draw/finish/odds/course. To evaluate jockey, trainer, carried weight,
body weight and running position at scale (the 140-race form scan was data-starved), we
re-scrape the same meetings via get_race_data and keep those fields + HorseID (the brand
number, joinable to our energetics horse_id). Output → results_rich.parquet, the base for
leak-free trailing jockey/trainer strike rates tested in the real walk-forward backtest.

Run: uv run python experiments/build_results_rich.py
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.historical import get_race_data

CACHE = Path("data_cache")
START, END = dt.date(2022, 9, 1), dt.date(2026, 6, 1)
KEEP = ["Date", "Venue", "RaceNumber", "HorseNo", "HorseID", "Jockey", "Trainer",
        "ActWt", "DeclarHorseWt", "Dr", "Pla", "WinOdds", "RunningPosition",
        "LBW", "FinishTime", "Going", "Course", "RaceDistance", "RaceClass"]


def fetch_meeting(date: dt.date):
    """Try both venues on a date; keep whichever has races. Returns rows or []."""
    ds = date.strftime("%Y/%m/%d")
    for venue in ("ST", "HV"):
        frames = []
        for r in range(1, 12):
            try:
                df = get_race_data(ds, venue, r)
            except Exception:
                break
            if df.height == 0:
                break
            cols = [c for c in KEEP if c in df.columns]
            frames.append(df.select(cols).with_columns(pl.lit(date.isoformat()).alias("date")))
        if frames:
            return frames
    return []


def main():
    days = [START + dt.timedelta(d) for d in range((END - START).days)]
    print(f"probing {len(days)} dates for ST/HV meetings ...")
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
    out.write_parquet(CACHE / "results_rich.parquet")
    nr = out.select(["date", "Venue", "RaceNumber"]).unique().height if out.height else 0
    print(f"wrote results_rich.parquet: {out.height} runner-rows, {nr} races, "
          f"{out['Jockey'].n_unique() if out.height else 0} jockeys")


if __name__ == "__main__":
    main()
